"""Self-check a Galileo Hackathon Track 1 task submission.

Usage::

    python tools/check_submission.py <task_dir-or-zip> --siderius <SIDERIUS checkout> \
        [--data_dir DIR] [--download] [--retrain] [--report out.json]

Steps (each PASS, FAIL or SKIP with a reason). Verdict and exit status:
PASS (0) when no step fails and the required steps 3a, 3b, 4.0 and 4a ran;
FAIL (1) when any step fails; INCOMPLETE (2) when nothing failed but a required
step was skipped (4b may be skipped).

    1   package               layout, sizes, required files, refused file types,
                              TASK_CARD.md headings, JSON schemas
    2a  zenodo_record         the Zenodo record exists and lists the declared files
    2b  data_files            --data_dir files match data_source.json and fetch_data.py
                              processes them; or --download runs fetch_data.py
    3a  siderius_admission    console init + add-task --manifest --no-gpu + check
    3b  siderius_composition  the SIDERIUS composer loads task/task.yaml and its plugins
    4.0 baseline_weights      init/final .pt weights load with torch.load(weights_only=True)
    4a  baseline_evaluate     run_baseline.py evaluate reproduces baseline_result.json
    4b  baseline_retrain      (--retrain) run_baseline.py train from init weights + seed

Participant code (fetch_data.py, run_baseline.py, task plugins) runs only in
subprocesses: own session, killed with its process group on timeout, umask 077,
a fresh empty working folder per step, output captured to a file, and a reduced
environment (PATH, LANG, LC_ALL, CUDA_VISIBLE_DEVICES, proxy and SSL_CERT_FILE
variables, a private HOME and TMPDIR, PYTHONHASHSEED=0, PYTHONDONTWRITEBYTECODE=1,
PYTHONUNBUFFERED=1). Checker-owned snippets (3b, 4.0) run with ``python -I``.
Participant code is never imported into this process, but it is NOT sandboxed:
check untrusted submissions in a disposable machine or container.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

from submission_rules import (  # noqa: E402
    KIT_VERSION,
    MAX_FILE_BYTES,
    MAX_ZIP_BYTES,
    DataSource,
    MetricsFile,
    StaticResult,
    build_zip_bytes,
    check_static,
    compare_metrics,
    md5_file,
    sha256_bytes,
    list_package_files,
    zip_member_problems,
)

Status = Literal["PASS", "FAIL", "SKIP"]
ZENODO_API = "https://zenodo.org/api/records/"
RESULT_MARKER = "SUBMISSION_CHECK_RESULT "
OUTPUT_TAIL_LINES = 40
OUTPUT_TAIL_BYTES = 64 * 1024
#: Bound on waiting for a killed process group to be reaped.
KILL_WAIT_SECONDS = 10.0
#: Steps that must run for a PASS verdict; 4b (retrain) is optional.
REQUIRED_STEPS = ("3a", "3b", "4.0", "4a")
PASSED_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "LC_ALL",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
)


@dataclass
class Step:
    id: str
    name: str
    status: Status
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """Everything later steps need from earlier ones."""

    root: Path
    work: Path
    args: argparse.Namespace
    static: StaticResult
    python: Path | None
    processed_data: Path | None = None
    declared_metric_ids: list[str] | None = None
    admission_manifest: Path | None = None


# ---------------------------------------------------------------------------
# Subprocesses
# ---------------------------------------------------------------------------


@dataclass
class Completed:
    command: list[str]
    returncode: int | None  # None: timed out and killed
    seconds_limit: float
    output_tail: str


def child_env(work: Path) -> dict[str, str]:
    """A reduced environment: no credentials or tokens from the caller's shell.

    HOME and TMPDIR are private folders inside the work folder, so participant
    code cannot read or write the caller's home configuration through them.
    """
    home = work / "child_home"
    temporary = work / "child_tmp"
    home.mkdir(mode=0o700, exist_ok=True)
    temporary.mkdir(mode=0o700, exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "TMPDIR": str(temporary),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
    }
    for name in PASSED_ENV:
        if name in os.environ:
            env[name] = os.environ[name]
    return env


def fresh_dir(work: Path, label: str) -> Path:
    """A new empty private working folder for one step."""
    return Path(tempfile.mkdtemp(prefix=f"cwd-{label}-", dir=work))


def _kill_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run(command: list[str], *, cwd: Path, timeout: float, env: dict[str, str]) -> Completed:
    """Run one subprocess in its own session; kill the whole group on timeout.

    Output goes to an anonymous file, not a pipe, so a grandchild that escapes
    the kill and keeps the descriptor open cannot block the checker. Leftover
    members of the process group are killed after a normal exit too.
    """
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            umask=0o077,
        )
        code: int | None
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(process.pid)
            try:
                process.wait(timeout=KILL_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                pass
            code = None
        _kill_group(process.pid)
        size = log.seek(0, os.SEEK_END)
        log.seek(max(0, size - OUTPUT_TAIL_BYTES))
        text = log.read().decode("utf-8", errors="replace")
    tail = "\n".join(text.splitlines()[-OUTPUT_TAIL_LINES:])
    return Completed([str(part) for part in command], code, timeout, tail)


def describe(done: Completed) -> str:
    if done.returncode is None:
        return f"timed out after {done.seconds_limit:.0f} s and was killed"
    return f"exited with status {done.returncode}"


def marker_result(done: Completed) -> dict[str, Any] | None:
    for line in reversed(done.output_tail.splitlines()):
        if line.startswith(RESULT_MARKER):
            try:
                return json.loads(line[len(RESULT_MARKER) :])
            except json.JSONDecodeError:
                return None
    return None


# ---------------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------------


def step_package(
    task_id: str | None, static: StaticResult | None, zip_bytes_size: int | None, zip_problems: list[str]
) -> Step:
    problems = list(zip_problems) + (static.problems if static is not None else [])
    notes = list(static.notes) if static is not None else []
    if zip_bytes_size is None:
        notes.append("zip size not measured (the zip was not built or not read)")
    elif zip_bytes_size > MAX_ZIP_BYTES:
        problems.append(f"the zip is {zip_bytes_size} bytes; the limit is {MAX_ZIP_BYTES} (20 MB)")
    details = {
        "task_id": task_id,
        "files": len(static.files) if static is not None else None,
        "zip_bytes": zip_bytes_size,
        "problems": problems,
        "notes": notes,
    }
    if problems:
        return Step("1", "package", "FAIL", f"{len(problems)} problem(s): {problems[0]}", details)
    return Step("1", "package", "PASS", "layout, sizes, file types, card and JSON schemas are valid", details)


# ---------------------------------------------------------------------------
# Step 2
# ---------------------------------------------------------------------------

Fetch = Callable[[str], tuple[int, str, Any]]


def fetch_json(url: str) -> tuple[int, str, Any]:
    """``(HTTP status, final URL after redirects, decoded JSON or None)``."""
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": f"siderius-hackathon-check/{KIT_VERSION}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.geturl(), json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, url, None


def step_zenodo_record(source: DataSource | None, fetch: Fetch) -> Step:
    name = "zenodo_record"
    if source is None:
        return Step("2a", name, "SKIP", "data_source.json is missing or invalid (see step 1)")
    if source.is_placeholder:
        return Step(
            "2a",
            name,
            "FAIL",
            "zenodo_record_id is the placeholder ZENODO_RECORD_ID; publish the data on Zenodo and fill in the id and DOI",
        )
    url = ZENODO_API + source.zenodo_record_id
    try:
        status, final_url, record = fetch(url)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        return Step("2a", name, "FAIL", f"could not query {url}: {error}; rerun with network access")
    details: dict[str, Any] = {"url": url}
    if status in (404, 410) or record is None:
        return Step("2a", name, "FAIL", f"{url} returned HTTP {status}; no such published record", details)
    if status != 200:
        return Step("2a", name, "FAIL", f"{url} returned HTTP {status}", details)
    problems: list[str] = []
    notes: list[str] = []
    record_id = str(record.get("id", ""))
    if record_id != source.zenodo_record_id:
        problems.append(
            f"record {source.zenodo_record_id} resolves to {record_id} ({final_url}); "
            "use the version record id, not the concept id"
        )
    if record.get("doi") and record["doi"] != source.doi:
        problems.append(f"record DOI is {record['doi']}, data_source.json says {source.doi}")
    metadata = record.get("metadata") or {}
    access_right = metadata.get("access_right")
    details["access_right"] = access_right
    expected = "open" if source.access == "public" else "restricted"
    if access_right != expected:
        problems.append(
            f"Zenodo access_right is {access_right!r}; data_source.json access {source.access!r} needs {expected!r}"
        )
    license_id = (metadata.get("license") or {}).get("id")
    details["license"] = license_id
    if license_id is None:
        notes.append("Zenodo reports no license for this record")
    elif license_id.lower() != source.license.lower():
        problems.append(f"Zenodo license is {license_id!r}, data_source.json says {source.license!r}")
    listed = {entry.get("key"): entry for entry in record.get("files") or []}
    details["zenodo_files"] = len(listed)
    if not listed and source.access == "restricted":
        notes.append("Zenodo does not list the files of a restricted record publicly; step 2b verifies them")
    else:
        for declared in source.files:
            entry = listed.get(declared.name)
            if entry is None:
                problems.append(f"{declared.name} is not in the Zenodo record")
                continue
            if entry.get("size") != declared.size:
                problems.append(f"{declared.name}: Zenodo size {entry.get('size')}, declared {declared.size}")
            if entry.get("checksum") != f"md5:{declared.md5}":
                problems.append(
                    f"{declared.name}: Zenodo checksum {entry.get('checksum')}, declared md5:{declared.md5}"
                )
        undeclared = sorted(set(listed) - {d.name for d in source.files})
        if undeclared:
            notes.append(f"Zenodo files not listed in data_source.json (not checked): {undeclared}")
    details["problems"] = problems
    details["notes"] = notes
    if problems:
        return Step("2a", name, "FAIL", problems[0], details)
    return Step(
        "2a", name, "PASS", f"record {record_id} found; declared files match Zenodo's names, sizes and md5", details
    )


def step_data_files(ctx: Context) -> Step:
    name = "data_files"
    source = ctx.static.data_source
    args = ctx.args
    if source is None:
        return Step("2b", name, "SKIP", "data_source.json is missing or invalid (see step 1)")
    if not (ctx.root / "fetch_data.py").is_file():
        return Step("2b", name, "SKIP", "fetch_data.py is missing (see step 1)")
    details: dict[str, Any] = {}
    output = ctx.work / "processed_data"
    command = [str(ctx.python or sys.executable), str(ctx.root / "fetch_data.py"), "--output_dir", str(output)]
    if args.data_dir is not None:
        source_dir = Path(args.data_dir).resolve()
        problems = []
        for declared in source.files:
            path = source_dir / declared.name
            if not path.is_file():
                problems.append(f"{declared.name} is not in --data_dir {source_dir}")
            elif path.stat().st_size != declared.size:
                problems.append(f"{declared.name}: size {path.stat().st_size}, declared {declared.size}")
            elif md5_file(path) != declared.md5:
                problems.append(f"{declared.name}: md5 differs from data_source.json")
        details["source_dir"] = str(source_dir)
        details["verified_files"] = len(source.files) - len(problems)
        if problems:
            details["problems"] = problems
            return Step("2b", name, "FAIL", problems[0], details)
        command += ["--source_dir", str(source_dir)]
        mode = "processed the --data_dir files"
    elif args.download:
        if source.access == "restricted" and not source.share_link:
            return Step(
                "2b",
                name,
                "SKIP",
                "restricted record without share_link: download skipped; pass --data_dir with the downloaded files",
            )
        mode = "downloaded and processed the Zenodo files"
    else:
        return Step("2b", name, "SKIP", "pass --data_dir DIR (files already downloaded) or --download")
    done = run(command, cwd=fresh_dir(ctx.work, "2b"), timeout=args.fetch_timeout, env=child_env(ctx.work))
    details["command"] = done.command
    details["output_tail"] = done.output_tail
    if done.returncode != 0:
        return Step("2b", name, "FAIL", f"fetch_data.py {describe(done)}", details)
    produced = (
        sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()) if output.is_dir() else []
    )
    details["processed_files"] = produced[:50]
    if not produced:
        return Step("2b", name, "FAIL", "fetch_data.py exited 0 but wrote nothing to --output_dir", details)
    ctx.processed_data = output
    return Step("2b", name, "PASS", f"fetch_data.py {mode}: {len(produced)} file(s) in the data folder", details)


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------


def siderius_identity(checkout: Path) -> dict[str, Any]:
    identity: dict[str, Any] = {"path": str(checkout)}
    try:
        identity["version"] = tomllib.loads((checkout / "pyproject.toml").read_text())["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        identity["version"] = None
    try:
        identity["git_revision"] = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        identity["git_revision"] = None
    return identity


def bwrap_status() -> str:
    """Whether the bubblewrap sandbox SIDERIUS runs use can start on this machine."""
    executable = shutil.which("bwrap")
    if executable is None:
        return "not installed"
    try:
        probe = subprocess.run([executable, "--ro-bind", "/", "/", "true"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "failed to start"
    return "usable" if probe.returncode == 0 else "cannot create namespaces"


def _private_tree(path: Path) -> None:
    for current, dirnames, filenames in os.walk(path):
        os.chmod(current, 0o700)
        for name in filenames:
            os.chmod(os.path.join(current, name), 0o600)


def step_admission(ctx: Context) -> Step:
    name = "siderius_admission"
    if ctx.python is None:
        return Step("3a", name, "SKIP", "pass --siderius <checkout> (SIDERIUS at 1ab94b51 or later, after uv sync)")
    if not sys.platform.startswith("linux"):
        return Step("3a", name, "SKIP", "the SIDERIUS console runs on Linux only")
    if not (ctx.root / "task" / "task.yaml").is_file():
        return Step("3a", name, "SKIP", "task/task.yaml is missing (see step 1)")
    checkout = Path(ctx.args.siderius).resolve()
    task_id = ctx.root.name
    base = ctx.work / "siderius"
    base.mkdir(mode=0o700)
    project = base / "project"
    project.mkdir(mode=0o700)
    credential = base / "credentials.json"
    descriptor = os.open(credential, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write("{}\n")
    env = child_env(ctx.work)
    details: dict[str, Any] = {"bwrap": bwrap_status(), "workspace": str(project / "siderius_workspace")}
    console = [str(ctx.python), "-m", "tools.candidate_console"]
    timeout = ctx.args.siderius_timeout

    init = run(
        console + ["init", "--workspace", str(project), "--credential-file", str(credential)],
        cwd=checkout,
        timeout=timeout,
        env=env,
    )
    details["init"] = asdict(init)
    if init.returncode != 0:
        return Step("3a", name, "FAIL", f"candidate_console init {describe(init)}", details)
    workspace = project / "siderius_workspace"
    config = workspace / "settings" / "console.json"
    package = workspace / "tasks" / task_id
    shutil.copytree(ctx.root / "task", package, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    data = workspace / "data" / task_id
    if ctx.processed_data is not None:
        shutil.copytree(ctx.processed_data, data, symlinks=True)
        details["data"] = "processed data from step 2b"
    else:
        data.mkdir(parents=True)
        details["data"] = "empty data folder (step 2b did not produce data)"
    _private_tree(package)
    _private_tree(data)

    add = run(
        console + ["add-task", "--config", str(config), "--manifest", str(package / "task.yaml"), "--no-gpu"],
        cwd=checkout,
        timeout=timeout,
        env=env,
    )
    details["add_task"] = asdict(add)
    if add.returncode != 0:
        return Step("3a", name, "FAIL", f"add-task {describe(add)}; see details.add_task.output_tail", details)
    check = run(console + ["check", "--config", str(config)], cwd=checkout, timeout=timeout, env=env)
    details["check"] = asdict(check)
    if check.returncode != 0:
        return Step("3a", name, "FAIL", f"check {describe(check)}; see details.check.output_tail", details)
    ctx.admission_manifest = package / "task.yaml"
    reason = f"init, add-task --manifest --no-gpu and check accepted task {task_id}"
    if details["bwrap"] != "usable":
        reason += f" (note: bubblewrap {details['bwrap']}; SIDERIUS runs need it, these commands do not)"
    return Step("3a", name, "PASS", reason, details)


COMPOSE_SCRIPT = f"""
import json, sys
from enum import Enum
from workflows.task_composition import compose_run_task_bindings
c = compose_run_task_bindings(sys.argv[1])
out = {{
    "task_data_path_id": c.task_data_path_id,
    "primary_metric": {{"id": c.metric.spec.id, "direction": c.metric.spec.direction}},
    "secondary_metrics": [{{"id": m.spec.id, "direction": m.spec.direction}} for m in c.secondary_metrics],
    "task_health": str(c.task_health_binding.value) if isinstance(c.task_health_binding, Enum) else "declared",
    "semantic_fingerprint": c.semantic_fingerprint,
}}
print({RESULT_MARKER!r} + json.dumps(out, sort_keys=True))
"""


def step_composition(ctx: Context) -> Step:
    name = "siderius_composition"
    if ctx.python is None:
        return Step("3b", name, "SKIP", "pass --siderius <checkout>")
    manifest = ctx.admission_manifest or ctx.root / "task" / "task.yaml"
    if not Path(manifest).is_file():
        return Step("3b", name, "SKIP", "task/task.yaml is missing (see step 1)")
    done = run(
        [str(ctx.python), "-I", "-c", COMPOSE_SCRIPT, str(manifest)],
        cwd=fresh_dir(ctx.work, "3b"),
        timeout=ctx.args.siderius_timeout,
        env=child_env(ctx.work),
    )
    details: dict[str, Any] = {
        "command": done.command[:3] + ["<compose script>", str(manifest)],
        "output_tail": done.output_tail,
    }
    result = marker_result(done)
    if done.returncode != 0 or result is None:
        return Step("3b", name, "FAIL", f"composing task/task.yaml {describe(done)}; see details.output_tail", details)
    details["composition"] = result
    ctx.declared_metric_ids = [result["primary_metric"]["id"]] + [m["id"] for m in result["secondary_metrics"]]
    return Step(
        "3b",
        name,
        "PASS",
        f"composed: primary metric {result['primary_metric']['id']} ({result['primary_metric']['direction']}), "
        f"{len(result['secondary_metrics'])} secondary, Health {result['task_health']}",
        details,
    )


# ---------------------------------------------------------------------------
# Step 4
# ---------------------------------------------------------------------------

WEIGHTS_SCRIPT = f"""
import json, sys
import torch
out = {{}}
for path in sys.argv[1:]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise SystemExit(path + ": expected a non-empty flat state dict {{name: tensor}}")
    out[path] = {{"tensors": len(state), "parameters": int(sum(v.numel() for v in state.values()))}}
print({RESULT_MARKER!r} + json.dumps(out, sort_keys=True))
"""


def step_weights(ctx: Context) -> Step:
    name = "baseline_weights"
    if ctx.python is None:
        return Step("4.0", name, "SKIP", "pass --siderius <checkout> (its interpreter provides torch)")
    static = ctx.static
    if static.init_weights is None or static.final_weights is None:
        return Step("4.0", name, "SKIP", "init or final weights are missing (see step 1)")
    init, final = ctx.root / static.init_weights, ctx.root / static.final_weights
    details: dict[str, Any] = {"init": static.init_weights, "final": static.final_weights}
    if init.read_bytes() == final.read_bytes():
        return Step("4.0", name, "FAIL", "final weights are byte-identical to the initial weights", details)
    done = run(
        [str(ctx.python), "-I", "-c", WEIGHTS_SCRIPT, str(init), str(final)],
        cwd=fresh_dir(ctx.work, "4.0"),
        timeout=ctx.args.siderius_timeout,
        env=child_env(ctx.work),
    )
    result = marker_result(done)
    details["output_tail"] = done.output_tail
    if done.returncode != 0 or result is None:
        return Step(
            "4.0", name, "FAIL", f"loading the weights safely {describe(done)}; see details.output_tail", details
        )
    details["loaded"] = {Path(k).name: v for k, v in result.items()}
    return Step(
        "4.0", name, "PASS", "both weight files load as state dicts with torch.load(weights_only=True)", details
    )


def _read_metrics(path: Path) -> tuple[dict[str, float] | None, str | None]:
    if not path.is_file():
        return None, f"{path.name} was not written"
    try:
        return MetricsFile.model_validate_json(b'{"metrics": ' + path.read_bytes() + b"}").metrics, None
    except ValueError as error:
        return None, f"{path.name} is not a {{metric id: number}} object: {error}"


def _compare(
    ctx: Context, step_id: str, name: str, metrics_path: Path, done: Completed, details: dict[str, Any], what: str
) -> Step:
    reported = ctx.static.baseline_result
    assert reported is not None
    details["output_tail"] = done.output_tail
    if done.returncode != 0:
        return Step(step_id, name, "FAIL", f"run_baseline.py {describe(done)}; see details.output_tail", details)
    metrics, problem = _read_metrics(metrics_path)
    if metrics is None:
        return Step(step_id, name, "FAIL", problem or "no metrics", details)
    details["metrics"] = metrics
    problems: list[str] = []
    if ctx.declared_metric_ids is not None and set(metrics) != set(ctx.declared_metric_ids):
        problems.append(
            f"metrics.json ids {sorted(metrics)} differ from the task's declared metric ids {sorted(ctx.declared_metric_ids)}"
        )
    ok, rows, comparison_problems = compare_metrics(reported, metrics)
    problems += comparison_problems
    details["comparison"] = [asdict(row) for row in rows]
    if problems:
        details["problems"] = problems
        return Step(step_id, name, "FAIL", problems[0], details)
    return Step(step_id, name, "PASS", f"{what} reproduces every reported metric within its tolerance", details)


def _baseline_ready(ctx: Context, step_id: str, name: str) -> Step | None:
    if ctx.python is None:
        return Step(step_id, name, "SKIP", "pass --siderius <checkout> (the baseline runs with its interpreter)")
    if ctx.static.baseline_result is None or not (ctx.root / "baseline" / "run_baseline.py").is_file():
        return Step(step_id, name, "SKIP", "run_baseline.py or a valid baseline_result.json is missing (see step 1)")
    if ctx.processed_data is None:
        return Step(step_id, name, "SKIP", "no processed data: step 2b needs --data_dir or --download")
    return None


def step_evaluate(ctx: Context) -> Step:
    name = "baseline_evaluate"
    if (skip := _baseline_ready(ctx, "4a", name)) is not None:
        return skip
    if ctx.static.final_weights is None:
        return Step("4a", name, "SKIP", "final weights are missing (see step 1)")
    output = ctx.work / "baseline_evaluate"
    command = [
        str(ctx.python),
        str(ctx.root / "baseline" / "run_baseline.py"),
        "evaluate",
        "--data_dir",
        str(ctx.processed_data),
        "--weights",
        str(ctx.root / ctx.static.final_weights),
        "--output_dir",
        str(output),
    ]
    done = run(command, cwd=fresh_dir(ctx.work, "4a"), timeout=ctx.args.eval_timeout, env=child_env(ctx.work))
    return _compare(
        ctx, "4a", name, output / "metrics.json", done, {"command": done.command}, "evaluating the final weights"
    )


def step_retrain(ctx: Context) -> Step:
    name = "baseline_retrain"
    if not ctx.args.retrain:
        return Step("4b", name, "SKIP", "pass --retrain to train from the initial weights and seed")
    if (skip := _baseline_ready(ctx, "4b", name)) is not None:
        return skip
    if ctx.static.init_weights is None:
        return Step("4b", name, "SKIP", "initial weights are missing (see step 1)")
    reported = ctx.static.baseline_result
    assert reported is not None
    output = ctx.work / "baseline_retrain"
    command = [
        str(ctx.python),
        str(ctx.root / "baseline" / "run_baseline.py"),
        "train",
        "--data_dir",
        str(ctx.processed_data),
        "--init_weights",
        str(ctx.root / ctx.static.init_weights),
        "--seed",
        str(reported.seed),
        "--output_dir",
        str(output),
    ]
    done = run(command, cwd=fresh_dir(ctx.work, "4b"), timeout=ctx.args.train_timeout, env=child_env(ctx.work))
    details: dict[str, Any] = {"command": done.command}
    if done.returncode == 0:
        finals = sorted(p.name for p in output.glob("final_weights.*")) if output.is_dir() else []
        details["final_weights"] = finals
        if "final_weights.pt" not in finals:
            details["output_tail"] = done.output_tail
            return Step("4b", name, "FAIL", "train did not write final_weights.pt", details)
    return _compare(ctx, "4b", name, output / "metrics.json", done, details, f"retraining with seed {reported.seed}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class OpenedInput:
    """The submission as the checker sees it before step 1."""

    task_id: str | None
    root: Path | None  # None: not unpacked, only step 1 runs
    zip_bytes: int | None
    zip_sha256: str | None
    problems: list[str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _open_folder(path: Path) -> OpenedInput:
    root = path.resolve()
    files, _ = list_package_files(root)
    oversized = [f for f in files if (root / f).stat().st_size > MAX_FILE_BYTES]
    if oversized:
        # Step 1 reports each oversized file; do not read them into memory.
        return OpenedInput(root.name, root, None, None, [])
    payload = build_zip_bytes(root, files)
    return OpenedInput(root.name, root, len(payload), sha256_bytes(payload), [])


def _copy_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path) -> None:
    """Extract one member, never writing more than its declared size."""
    written = 0
    with archive.open(info) as source, destination.open("xb") as sink:
        while chunk := source.read(1 << 20):
            written += len(chunk)
            if written > info.file_size:
                raise zipfile.BadZipFile(f"{info.filename} expands beyond its declared size")
            sink.write(chunk)


def _open_zip(path: Path, work: Path) -> OpenedInput:
    size = path.stat().st_size
    task_id = None
    try:
        digest = _sha256_file(path)
        if size > MAX_ZIP_BYTES:
            problem = f"the zip is {size} bytes; the limit is {MAX_ZIP_BYTES} (20 MB); not unpacked"
            return OpenedInput(None, None, size, digest, [problem])
        with zipfile.ZipFile(path) as archive:
            task_id, problems = zip_member_problems(archive)
            if task_id is None or problems:
                problems.append("the zip was not unpacked; fix the problems above and rebuild it")
                return OpenedInput(task_id, None, size, digest, problems)
            target = work / "unpacked"
            target.mkdir(mode=0o700)
            for info in archive.infolist():
                destination = target / info.filename
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                _copy_member(archive, info, destination)
    except (OSError, zipfile.BadZipFile, EOFError, ValueError) as error:
        problem = f"the zip could not be read or unpacked: {type(error).__name__}: {error}"
        return OpenedInput(task_id, None, size, None, [problem])
    return OpenedInput(task_id, target / task_id, size, digest, [])


def open_input(path: Path, work: Path) -> OpenedInput:
    """Measure a folder as the zip make_submission.py would build, or bounded-unpack a zip."""
    if path.is_dir():
        return _open_folder(path)
    if not path.is_file() or not zipfile.is_zipfile(path):
        return OpenedInput(None, None, None, None, [f"{path} is neither a folder nor a zip file"])
    return _open_zip(path, work)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    cli = argparse.ArgumentParser(description="Self-check a Track 1 task submission (folder or zip).")
    cli.add_argument("submission", type=Path, help="the <task_id>/ folder or <task_id>.zip")
    cli.add_argument(
        "--siderius",
        type=Path,
        help="SIDERIUS checkout at 1ab94b5101f705f2bd90ba89a21fd10bb2ed98de or later, with .venv "
        "(uv sync --group dev --frozen)",
    )
    cli.add_argument("--data_dir", type=Path, help="folder holding the Zenodo files, already downloaded")
    cli.add_argument("--download", action="store_true", help="let fetch_data.py download from Zenodo")
    cli.add_argument("--retrain", action="store_true", help="also retrain the baseline (step 4b)")
    cli.add_argument("--report", type=Path, help="write the JSON report here")
    cli.add_argument("--keep_work", action="store_true", help="keep the temporary work folder and print it")
    cli.add_argument("--fetch_timeout", type=float, default=3600)
    cli.add_argument("--siderius_timeout", type=float, default=600)
    cli.add_argument("--eval_timeout", type=float, default=1800)
    cli.add_argument("--train_timeout", type=float, default=6 * 3600)
    args = cli.parse_args(argv)
    if args.data_dir is not None and args.download:
        cli.error("--data_dir and --download are alternatives; pass one")
    return args


STEP_NAMES = (
    ("2a", "zenodo_record"),
    ("2b", "data_files"),
    ("3a", "siderius_admission"),
    ("3b", "siderius_composition"),
    ("4.0", "baseline_weights"),
    ("4a", "baseline_evaluate"),
    ("4b", "baseline_retrain"),
)


def verdict_of(steps: list[Step]) -> tuple[str, list[str], list[str]]:
    """``(PASS | FAIL | INCOMPLETE, failed step ids, skipped required step ids)``."""
    failed = [step.id for step in steps if step.status == "FAIL"]
    skipped = [step.id for step in steps if step.status == "SKIP" and step.id in REQUIRED_STEPS]
    if failed:
        return "FAIL", failed, skipped
    if skipped:
        return "INCOMPLETE", failed, skipped
    return "PASS", failed, skipped


def check(args: argparse.Namespace, fetch: Fetch | None = None) -> dict[str, Any]:
    fetch = fetch or fetch_json
    work = Path(tempfile.mkdtemp(prefix="track1-check-"))
    os.chmod(work, 0o700)
    try:
        opened = open_input(args.submission, work)
        python = None
        identity = None
        if args.siderius is not None:
            identity = siderius_identity(Path(args.siderius).resolve())
            candidate = Path(args.siderius).resolve() / ".venv" / "bin" / "python"
            if not candidate.is_file():
                raise SystemExit(f"{candidate} does not exist; run `uv sync --group dev --frozen` in the checkout")
            python = candidate
        if opened.root is None:
            steps = [step_package(opened.task_id, None, opened.zip_bytes, opened.problems)]
            steps += [Step(i, n, "SKIP", "the submission was not unpacked (see step 1)") for i, n in STEP_NAMES]
        else:
            static = check_static(opened.root)
            ctx = Context(root=opened.root, work=work, args=args, static=static, python=python)
            steps = [step_package(opened.task_id, static, opened.zip_bytes, opened.problems)]
            steps.append(step_zenodo_record(static.data_source, fetch))
            steps.append(step_data_files(ctx))
            steps.append(step_admission(ctx))
            steps.append(step_composition(ctx))
            steps.append(step_weights(ctx))
            steps.append(step_evaluate(ctx))
            steps.append(step_retrain(ctx))
        verdict, failed, skipped = verdict_of(steps)
        return {
            "kit_version": KIT_VERSION,
            "task_id": opened.task_id,
            "submission": str(args.submission),
            "zip_sha256": opened.zip_sha256,
            "zip_bytes": opened.zip_bytes,
            "siderius": identity,
            "verdict": verdict,
            "failed_steps": failed,
            "skipped_required_steps": skipped,
            "steps": [asdict(step) for step in steps],
        }
    finally:
        if args.keep_work:
            print(f"work folder kept: {work}", file=sys.stderr)
        else:
            shutil.rmtree(work, ignore_errors=True)


EXIT_STATUS = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = check(args)
    for step in report["steps"]:
        print(f"[{step['status']:4}] {step['id']:>3} {step['name']}: {step['reason']}")
        for problem in step["details"].get("problems", [])[1:]:
            print(f"             - {problem}")
    print(f"verdict: {report['verdict']}  (zip sha256 {report['zip_sha256']}, {report['zip_bytes']} bytes)")
    if report["verdict"] == "INCOMPLETE":
        print(f"required steps skipped: {', '.join(report['skipped_required_steps'])}")
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"report: {args.report}")
    return EXIT_STATUS[report["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
