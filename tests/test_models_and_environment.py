"""Model-artifact rules per framework, format checks, no unpickling, requirements and the baseline venv."""

from __future__ import annotations

import json
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

import check_submission as checker
from conftest import REPO
from submission_rules import check_static, requirements_problems

EXAMPLES = REPO / "examples"
HDF5 = b"\x89HDF\r\n\x1a\n" + b"\0" * 8

# -- helpers --------------------------------------------------------------------


def _zip_bytes(names: list[str]) -> bytes:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, b"{}")
    return buffer.getvalue()


#: Content that passes each format's sanity check.
VALID = {
    "py": b"def predict(x):\n    return [0 for _ in x]\n",
    "pt": _zip_bytes(["archive/data.pkl"]),
    "onnx": b"\x08\x07\x12\x00",
    "keras": _zip_bytes(["config.json", "model.weights.h5", "metadata.json"]),
    "weights.h5": HDF5,
    "json": json.dumps({"learner": {}, "version": [2, 1, 0]}).encode(),
    "ubj": b"{L\x00",
    "txt": b"tree\nversion=v4\n",
    "cbm": b"CBM1\x00\x00",
    "skops": _zip_bytes(["schema.json"]),
    "joblib": b"\x80\x04K\x01.",
    "pkl": b"\x80\x04K\x01.",
}


def set_baseline(root: Path, framework: str, fmt: str, *, init: bool, final_content: bytes | None = None) -> None:
    """Replace the example's model artifacts and declare framework/model_format."""
    baseline = root / "baseline"
    for path in list(baseline.iterdir()):
        if path.name.startswith(("init_model", "final_model")):
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    stems = ["final_model"] + (["init_model"] if init else [])
    for stem in stems:
        content = final_content if (final_content is not None and stem == "final_model") else VALID.get(fmt)
        if fmt == "savedmodel":
            (baseline / stem / "variables").mkdir(parents=True)
            (baseline / stem / "saved_model.pb").write_bytes(b"\x08\x01" + stem.encode())
            (baseline / stem / "variables" / "variables.index").write_bytes(stem.encode())
        else:
            assert content is not None
            # make init and final differ
            (baseline / f"{stem}.{fmt}").write_bytes(content + (b"" if stem == "final_model" else b" "))
    result_path = baseline / "baseline_result.json"
    result = json.loads(result_path.read_text())
    result.update(framework=framework, model_format=fmt)
    result_path.write_text(json.dumps(result))


def model_problems(root: Path) -> list[str]:
    return [p for p in check_static(root).problems if "model" in p or "baseline_result" in p]


# -- framework / format / pair rules --------------------------------------------


@pytest.mark.parametrize(
    ("framework", "fmt", "init"),
    [
        ("classic", "py", False),
        ("pytorch", "pt", True),
        ("pytorch", "onnx", True),
        ("tensorflow", "keras", True),
        ("tensorflow", "weights.h5", True),
        ("tensorflow", "savedmodel", True),
        ("tensorflow", "onnx", True),
        ("xgboost", "json", False),
        ("xgboost", "ubj", False),
        ("lightgbm", "txt", False),
        ("catboost", "cbm", False),
        ("sklearn", "skops", False),
        ("sklearn", "joblib", False),
        ("sklearn", "pkl", False),
        ("other", "onnx", False),
        ("other", "onnx", True),
    ],
)
def test_allowed_framework_format_pairs_pass_step_1(submission: Path, framework: str, fmt: str, init: bool) -> None:
    set_baseline(submission, framework, fmt, init=init)
    assert check_static(submission).problems == []


@pytest.mark.parametrize("framework,fmt", [("pytorch", "pt"), ("tensorflow", "keras")])
def test_neural_frameworks_need_an_initial_model(submission: Path, framework: str, fmt: str) -> None:
    set_baseline(submission, framework, fmt, init=False)
    assert f"framework '{framework}' needs baseline/init_model.{fmt}" in model_problems(submission)


@pytest.mark.parametrize("framework,fmt", [("sklearn", "skops"), ("xgboost", "json"), ("lightgbm", "txt")])
def test_classical_frameworks_must_not_ship_an_initial_model(submission: Path, framework: str, fmt: str) -> None:
    set_baseline(submission, framework, fmt, init=True)
    assert any("has no initial model; remove it" in p for p in model_problems(submission))


@pytest.mark.parametrize("framework,fmt", [("sklearn", "pt"), ("pytorch", "keras"), ("xgboost", "pkl")])
def test_format_not_allowed_for_framework_is_a_schema_error(submission: Path, framework: str, fmt: str) -> None:
    set_baseline(submission, framework, fmt, init=framework == "pytorch")
    assert any(
        p.startswith("baseline/baseline_result.json") and "is not allowed for framework" in p
        for p in check_static(submission).problems
    )


def test_declared_format_must_match_the_file(submission: Path) -> None:
    set_baseline(submission, "sklearn", "skops", init=False)
    (submission / "baseline" / "final_model.skops").rename(submission / "baseline" / "final_model.joblib")
    assert any("has format 'joblib' but baseline_result.json says 'skops'" in p for p in model_problems(submission))


def test_initial_and_final_model_must_share_a_format(submission: Path) -> None:
    (submission / "baseline" / "init_model.pt").rename(submission / "baseline" / "init_model.onnx")
    assert any("must use the same format as the final model" in p for p in model_problems(submission))


def test_folder_without_saved_model_pb_is_not_a_savedmodel(submission: Path) -> None:
    set_baseline(submission, "tensorflow", "savedmodel", init=True)
    (submission / "baseline" / "final_model" / "saved_model.pb").unlink()
    assert any("is a folder but not a SavedModel" in p for p in model_problems(submission))


# -- step 4.0 format checks (no participant objects executed) --------------------


def _model_step(submission: Path, python: Path | None = None) -> checker.Step:
    ctx = checker.Context(
        root=submission,
        work=submission.parent,
        args=checker.parse_args([str(submission)]),
        static=check_static(submission),
        python=python,
    )
    return checker.step_model(ctx)


@pytest.mark.parametrize("fmt", ["keras", "weights.h5", "json", "ubj", "txt", "cbm", "skops", "joblib", "pkl"])
def test_valid_artifacts_pass_step_4_0_without_siderius(submission: Path, fmt: str) -> None:
    framework = {"keras": "tensorflow", "weights.h5": "tensorflow", "json": "xgboost", "ubj": "xgboost",
                 "txt": "lightgbm", "cbm": "catboost"}.get(fmt, "sklearn")  # fmt: skip
    set_baseline(submission, framework, fmt, init=framework == "tensorflow")
    step = _model_step(submission)
    assert step.status == "PASS", step.reason


@pytest.mark.parametrize(
    ("framework", "fmt", "content", "message"),
    [
        ("tensorflow", "keras", _zip_bytes(["config.json"]), "config.json and model.weights.h5"),
        ("tensorflow", "weights.h5", b"not hdf5", "not an HDF5 file"),
        ("xgboost", "json", b'{"trees": []}', "no 'learner' key"),
        ("xgboost", "ubj", b"[", "not a UBJSON object"),
        ("lightgbm", "txt", b"forest", "LightGBM"),
        ("catboost", "cbm", b"XXXX", "CBM1"),
        ("sklearn", "skops", _zip_bytes(["model.npy"]), "no schema.json"),
        ("pytorch", "onnx", b"\x00\x01", "ONNX"),
        ("pytorch", "pt", b"not a zip", "torch.save zip archive"),
    ],
)
def test_malformed_artifacts_fail_step_4_0(
    submission: Path, framework: str, fmt: str, content: bytes, message: str
) -> None:
    set_baseline(submission, framework, fmt, init=framework in ("pytorch", "tensorflow"), final_content=content)
    step = _model_step(submission)
    assert step.status == "FAIL" and message in step.reason, step.reason


def test_identical_initial_and_final_models_fail(submission: Path) -> None:
    shutil.copy(submission / "baseline" / "init_model.pt", submission / "baseline" / "final_model.pt")
    step = _model_step(submission)
    assert step.status == "FAIL" and "byte-identical" in step.reason


class _PlantsAFile:
    """Unpickling this object writes ``target``: proof that a pickle was executed."""

    def __init__(self, target: Path) -> None:
        self.target = target

    def __reduce__(self):
        return (Path.write_text, (self.target, "unpickled"))


@pytest.mark.parametrize("fmt", ["pkl", "joblib"])
def test_checker_never_unpickles_joblib_or_pkl(submission: Path, tmp_path: Path, fmt: str) -> None:
    marker = tmp_path / "executed.txt"
    payload = pickle.dumps(_PlantsAFile(marker))
    set_baseline(submission, "sklearn", fmt, init=False, final_content=payload)
    report = checker.check(checker.parse_args([str(submission)]), fetch=lambda url: (404, url, None))
    model_step = next(step for step in report["steps"] if step["id"] == "4.0")
    assert model_step["status"] == "PASS"
    assert "loaded only by your run_baseline.py" in model_step["details"]["final_model"]["check"]
    # also with an interpreter available (as with --siderius)
    assert _model_step(submission, python=Path(sys.executable)).status == "PASS"
    assert not marker.exists()
    pickle.loads(payload)  # the payload is live: loading it does write the marker
    assert marker.exists()


def test_malicious_pt_is_refused_by_weights_only_load(submission: Path, tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    marker = tmp_path / "executed.txt"
    torch.save({"w": _PlantsAFile(marker)}, submission / "baseline" / "final_model.pt")
    step = _model_step(submission, python=Path(sys.executable))
    assert step.status == "FAIL" and "loading the pt files safely" in step.reason
    assert not marker.exists()


# -- requirements.txt ------------------------------------------------------------


def test_valid_requirements_are_pinned_lines() -> None:
    text = "# baseline extras\nscikit-learn==1.7.2\nxgboost[scikit-learn]==2.1.4  # trees\n\nnumpy==2.4.3; python_version >= '3.12'\n"
    pins, problems = requirements_problems(text)
    assert problems == []
    assert pins == ["scikit-learn==1.7.2", "xgboost[scikit-learn]==2.1.4", "numpy==2.4.3; python_version >= '3.12'"]


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("-e .", "options"),
        ("--index-url https://example.org/simple", "options"),
        ("--extra-index-url https://example.org/simple", "options"),
        ("-r other.txt", "options"),
        ("https://example.org/pkg-1.0.whl", "URLs"),
        ("pkg @ git+https://github.com/x/pkg", "URLs"),
        ("./local_pkg", "local paths"),
        ("/abs/pkg.whl", "local paths"),
        ("scikit-learn", "pin an exact version"),
        ("scikit-learn>=1.5", "pin an exact version"),
        ("scikit-learn==1.*", "pin an exact version"),
    ],
)
def test_requirements_refusals(line: str, message: str) -> None:
    pins, problems = requirements_problems(f"numpy==2.4.3\n{line}\n")
    assert pins == ["numpy==2.4.3"]
    assert len(problems) == 1 and "line 2" in problems[0] and message in problems[0]


def test_invalid_requirements_fail_step_1(submission: Path) -> None:
    (submission / "baseline" / "requirements.txt").write_text("torch\n")
    assert any(p.startswith("baseline/requirements.txt line 1") for p in check_static(submission).problems)


# -- step 4.1 baseline environment (uv mocked) ------------------------------------


class FakeUv:
    """Stands in for uv. ``fail_on``: a subcommand, ``install`` (pass 1) or ``pins`` (pass 2, --no-build)."""

    def __init__(self, fail_on: str | None = None, output: str = "") -> None:
        self.calls: list[list[str]] = []
        self.envs: list[dict[str, str]] = []
        self.fail_on = fail_on
        self.output = output

    def __call__(
        self, arguments: list[str], *, cwd: Path, timeout: float, env: dict[str, str], tail_lines: int = 40
    ) -> checker.Completed:
        self.calls.append(arguments)
        self.envs.append(env)
        is_install = arguments[:2] == ["pip", "install"]
        failing = (
            arguments[0] == self.fail_on
            or (self.fail_on == "install" and is_install and "--no-build" not in arguments)
            or (self.fail_on == "pins" and is_install and "--no-build" in arguments)
        )
        if failing:
            return checker.Completed(["uv", *arguments], 1, timeout, self.output)
        if arguments[0] == "build":
            out = Path(arguments[arguments.index("--out-dir") + 1])
            out.mkdir(parents=True)
            (out / "siderius-1.0.0-py3-none-any.whl").write_bytes(b"wheel")
        if arguments[0] == "venv":
            (Path(arguments[-1]) / "bin").mkdir(parents=True)
        text = (
            "scikit-learn==1.7.2\nskops==0.13.0\ntorch==2.10.0\nsiderius @ file:///w/siderius-1.0.0-py3-none-any.whl\n"
        )
        return checker.Completed(["uv", *arguments], 0, timeout, text if arguments[:2] == ["pip", "freeze"] else "")


def _env_ctx(submission: Path, tmp_path: Path, requirements: str | None) -> checker.Context:
    if requirements is not None:
        (submission / "baseline" / "requirements.txt").write_text(requirements)
    work = tmp_path / "work"
    work.mkdir()
    return checker.Context(
        root=submission,
        work=work,
        args=checker.parse_args([str(submission), "--siderius", str(tmp_path)]),
        static=check_static(submission),
        python=Path(sys.executable),
    )


@pytest.fixture
def fake_uv(monkeypatch: pytest.MonkeyPatch):
    def install(fake: FakeUv) -> FakeUv:
        monkeypatch.setattr(checker, "run_uv", fake)
        monkeypatch.setattr(checker, "copy_siderius_source", lambda checkout, target: target.mkdir())
        monkeypatch.setattr(checker.shutil, "which", lambda name: f"/usr/bin/{name}")
        return fake

    return install


def test_without_requirements_the_siderius_venv_is_used(submission: Path, tmp_path: Path, fake_uv) -> None:
    fake = fake_uv(FakeUv())
    ctx = _env_ctx(submission, tmp_path, None)
    step = checker.step_environment(ctx)
    assert step.status == "PASS" and ctx.baseline_python == Path(sys.executable) and fake.calls == []


def test_requirements_build_an_isolated_venv_with_siderius_locked(submission: Path, tmp_path: Path, fake_uv) -> None:
    fake = fake_uv(FakeUv())
    ctx = _env_ctx(submission, tmp_path, "scikit-learn==1.7.2\nskops==0.13.0\n")
    step = checker.step_environment(ctx)
    assert step.status == "PASS", step.reason
    venv_python = ctx.work / "baseline-env" / "venv" / "bin" / "python"
    assert ctx.baseline_python == venv_python
    assert [call[:2] for call in fake.calls] == [
        ["build", "--wheel"],
        ["export", "--frozen"],
        ["venv", "--python"],
        ["pip", "install"],
        ["pip", "install"],
        ["pip", "freeze"],
    ]
    assert fake.calls[2] == ["venv", "--python", sys.executable, str(ctx.work / "baseline-env" / "venv")]
    siderius_pass, pins_pass = fake.calls[3], fake.calls[4]
    # pass 1: SIDERIUS wheel with its locked constraints; no participant pins, builds allowed
    assert siderius_pass[:4] == ["pip", "install", "--python", str(venv_python)]
    assert siderius_pass[4].endswith("siderius-1.0.0-py3-none-any.whl")
    assert "-r" not in siderius_pass and "--no-build" not in siderius_pass and "-e" not in siderius_pass
    assert siderius_pass[siderius_pass.index("-c") + 1].endswith("constraints.txt")
    # pass 2: only the validated pins, wheels only
    assert pins_pass[:5] == ["pip", "install", "--python", str(venv_python), "--no-build"]
    assert not any(part.endswith(".whl") for part in pins_pass)
    assert pins_pass[pins_pass.index("-c") + 1].endswith("constraints.txt")
    written = Path(pins_pass[pins_pass.index("-r") + 1]).read_text()
    assert written == "scikit-learn==1.7.2\nskops==0.13.0\n"
    assert step.details["resolved_versions"] == {
        "scikit-learn": "1.7.2",
        "skops": "0.13.0",
        "torch": "2.10.0",
        "siderius": "siderius-1.0.0-py3-none-any.whl",
    }


def test_network_failure_skips_and_makes_the_verdict_incomplete(submission: Path, tmp_path: Path, fake_uv) -> None:
    fake_uv(FakeUv(fail_on="install", output="error: Failed to fetch: `https://pypi.org/simple/skops/`"))
    ctx = _env_ctx(submission, tmp_path, "skops==0.13.0\n")
    step = checker.step_environment(ctx)
    assert step.status == "SKIP" and "network" in step.reason and ctx.baseline_python is None
    assert checker.step_evaluate(ctx).status == "SKIP"
    passing = [checker.Step(i, "x", "PASS", "") for i in ("1", "3a", "3b", "4.0")]
    assert checker.verdict_of([*passing, step])[0] == "INCOMPLETE"


def test_resolution_conflict_fails(submission: Path, tmp_path: Path, fake_uv) -> None:
    fake_uv(FakeUv(fail_on="pins", output="No solution found when resolving dependencies: numpy==1.26.0"))
    step = checker.step_environment(_env_ctx(submission, tmp_path, "numpy==1.26.0\n"))
    assert step.status == "FAIL" and "installing baseline/requirements.txt exited" in step.reason


def test_missing_uv_or_invalid_requirements_skip(submission: Path, tmp_path: Path, fake_uv, monkeypatch) -> None:
    fake_uv(FakeUv())
    ctx = _env_ctx(submission, tmp_path, "torch\n")
    assert "invalid" in checker.step_environment(ctx).reason
    (submission / "baseline" / "requirements.txt").write_text("torch==2.10.0\n")
    ctx.static = check_static(submission)
    monkeypatch.setattr(checker.shutil, "which", lambda name: None)
    step = checker.step_environment(ctx)
    assert step.status == "SKIP" and "uv is not installed" in step.reason


# -- the three worked examples ----------------------------------------------------


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}  # fmt: skip


def test_three_examples_share_task_package_and_data_source() -> None:
    reference = EXAMPLES / "quickstart_classification"
    for name in ("quickstart_classification_sklearn", "quickstart_classification_keras"):
        example = EXAMPLES / "advanced" / name
        assert _tree(example / "task") == _tree(reference / "task"), name
        for shared in ("fetch_data.py", "data_source.json"):
            assert (example / shared).read_bytes() == (reference / shared).read_bytes(), (name, shared)


@pytest.mark.parametrize(
    ("name", "framework", "has_init"),
    [
        ("quickstart_classification", "pytorch", True),
        ("quickstart_classification_sklearn", "sklearn", False),
        ("quickstart_classification_keras", "tensorflow", True),
    ],
)
def test_examples_pass_step_1_and_4_0(name: str, framework: str, has_init: bool, tmp_path: Path) -> None:
    root = tmp_path / name
    source = EXAMPLES / name if name == "quickstart_classification" else EXAMPLES / "advanced" / name
    shutil.copytree(source, root, ignore=shutil.ignore_patterns("__pycache__"))
    static = check_static(root)
    assert static.problems == [] and static.baseline_result is not None
    assert static.baseline_result.framework == framework and (static.init_model is not None) == has_init
    python = Path(sys.executable) if framework == "pytorch" else None
    assert _model_step(root, python=python).status == "PASS"


def test_parse_freeze_handles_direct_references() -> None:
    text = "a==1\nb @ file:///x/b-2.0-py3-none-any.whl\n# comment\n"
    assert checker.parse_freeze(text) == {"a": "1", "b": "b-2.0-py3-none-any.whl"}


NO_WHEEL = (
    "  x No solution found when resolving dependencies:\n"
    "  hint: Wheels are required for `antlr4-python3-runtime` because building from source is disabled "
    "for all packages (i.e., with `--no-build`)\n"
)


def test_source_only_pin_fails_with_a_clear_message(submission: Path, tmp_path: Path, fake_uv) -> None:
    """A pin with no wheel must not be built (its setup.py would run); 4.1 FAILs and says why."""
    fake = fake_uv(FakeUv(fail_on="pins", output=NO_WHEEL))
    ctx = _env_ctx(submission, tmp_path, "antlr4-python3-runtime==4.9.3\n")
    step = checker.step_environment(ctx)
    assert step.status == "FAIL" and "has no wheel" in step.reason and "source builds are not allowed" in step.reason
    assert "--no-build" in fake.calls[-1] and ctx.baseline_python is None


def test_uv_runs_without_unrelated_caller_variables(
    submission: Path, tmp_path: Path, fake_uv, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "AWS_SECRET_ACCESS_KEY": "secret",
        "OPENAI_API_KEY": "secret",
        "GITHUB_TOKEN": "secret",
        "UV_INDEX_URL": "https://mirror.example/simple",
        "UV_CACHE_DIR": str(tmp_path / "cache"),
        "PIP_INDEX_URL": "https://mirror.example/simple",
        "HTTPS_PROXY": "http://proxy:3128",
        "SSL_CERT_FILE": "/etc/ssl/cert.pem",
    }.items():
        monkeypatch.setenv(name, value)
    fake = fake_uv(FakeUv())
    ctx = _env_ctx(submission, tmp_path, "skops==0.13.0\n")
    assert checker.step_environment(ctx).status == "PASS"
    allowed = {
        "PATH", "HOME", "LANG", "TMPDIR", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED", "PYTHONUNBUFFERED",
        "CUDA_VISIBLE_DEVICES", "LC_ALL", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy",
        "no_proxy", "SSL_CERT_FILE", "PIP_INDEX_URL",
    }  # fmt: skip
    for env in fake.envs:
        unexpected = {name for name in env if name not in allowed and not name.startswith("UV_")}
        assert unexpected == set()
        assert env["UV_INDEX_URL"] == "https://mirror.example/simple" and env["HOME"] != str(Path.home())


def test_uv_env_disables_uv_config_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails if participant code could steer uv through a planted uv.toml in the shared child HOME."""
    monkeypatch.delenv("UV_NO_CONFIG", raising=False)
    assert checker.uv_env(tmp_path)["UV_NO_CONFIG"] == "1"


def test_uv_env_points_at_the_callers_cache_when_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.setattr(checker, "default_uv_cache", lambda: "/home/someone/.cache/uv")
    assert checker.uv_env(tmp_path)["UV_CACHE_DIR"] == "/home/someone/.cache/uv"


@pytest.mark.parametrize(
    "line",
    [
        "numpy==2.4.3; python_version >= '3.12' --index-url https://evil.example/simple",
        "numpy==2.4.3 ; os_name == 'posix' -e .",
        "numpy==2.4.3; os_name == 'posix' --extra-index-url=https://evil.example",
        "numpy==2.4.3; os_name == 'posix' \\",
        "numpy==2.4.3; os_name == 'posix' and os_name == `x`",
    ],
)
def test_options_or_odd_characters_after_a_marker_are_refused(line: str) -> None:
    pins, problems = requirements_problems(line + "\n")
    assert pins == [] and len(problems) == 1


def test_tab_before_a_comment_is_allowed() -> None:
    pins, problems = requirements_problems("skops==0.13.0\t# for the tree\n")
    assert (pins, problems) == (["skops==0.13.0"], [])


def test_deeply_nested_json_model_is_a_failure_not_a_crash(submission: Path) -> None:
    set_baseline(submission, "xgboost", "json", init=False, final_content=b"[" * 200_000)
    step = _model_step(submission)
    assert step.status == "FAIL" and "nests too deeply" in step.reason


@pytest.mark.parametrize(
    ("framework", "fmt", "init", "wording"),
    [
        ("pytorch", "onnx", True, "onnx header only: onnx not installed; 4a is the real load test"),
        ("tensorflow", "keras", True, "header-only check, not loaded; 4a is the real load test"),
        ("sklearn", "pkl", False, "not loaded (pickle); 4a is the real load test"),
        ("xgboost", "json", False, "JSON structure check only"),
    ],
)
def test_model_step_says_how_deep_the_check_went(
    submission: Path, framework: str, fmt: str, init: bool, wording: str
) -> None:
    set_baseline(submission, framework, fmt, init=init)
    python = Path(sys.executable) if fmt == "onnx" else None
    pytest.importorskip("torch") if fmt == "onnx" else None
    step = _model_step(submission, python=python)
    assert step.status == "PASS" and wording in step.reason, step.reason


def test_classic_source_is_parsed_without_execution(submission: Path) -> None:
    set_baseline(submission, "classic", "py", init=False,
                 final_content=b"raise RuntimeError('must not execute during syntax check')\n")
    assert check_static(submission).problems == []
    assert _model_step(submission).status == "PASS"


def test_classic_invalid_source_and_initial_model_refused(submission: Path) -> None:
    set_baseline(submission, "classic", "py", init=False, final_content=b"def broken(:\n")
    assert _model_step(submission).status == "FAIL"
    set_baseline(submission, "classic", "py", init=True)
    assert any("initial model" in p for p in check_static(submission).problems)


def test_neural_framework_cannot_use_script_format(submission: Path) -> None:
    set_baseline(submission, "pytorch", "py", init=False)
    assert check_static(submission).problems


def test_classic_evaluation_and_rerun_compare_real_outputs(submission: Path, tmp_path: Path) -> None:
    set_baseline(submission, "classic", "py", init=False,
                 final_content=b"def predict(x): return int(x >= 0)\n")
    result_path = submission / "baseline/baseline_result.json"
    result = json.loads(result_path.read_text())
    result.update(metrics={"accuracy": 1.0}, tolerance={"accuracy": 0.0})
    result_path.write_text(json.dumps(result))
    data = tmp_path / "data"
    data.mkdir()
    (data / "rows.json").write_text("[[-2, 0], [3, 1]]")
    (submission / "baseline/run_baseline.py").write_text("""
import argparse, importlib.util, json, shutil
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('action'); p.add_argument('--data_dir'); p.add_argument('--output_dir')
p.add_argument('--model'); p.add_argument('--seed')
a=p.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
model=Path(a.model) if a.model else Path(__file__).parent/'final_model.py'
if a.action == 'train':
    shutil.copyfile(model, out/'final_model.py'); model=out/'final_model.py'
spec=importlib.util.spec_from_file_location('algorithm', model)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rows=json.loads((Path(a.data_dir)/'rows.json').read_text())
score=sum(m.predict(x)==y for x,y in rows)/len(rows)
(out/'metrics.json').write_text(json.dumps({'accuracy': score}))
""")
    work=tmp_path/'work'; work.mkdir()
    ctx=checker.Context(root=submission, work=work,
        args=checker.parse_args([str(submission), '--retrain']), static=check_static(submission),
        python=Path(sys.executable), baseline_python=Path(sys.executable),
        processed_data=data, declared_metric_ids=['accuracy'])
    assert checker.step_evaluate(ctx).status == "PASS"
    assert checker.step_retrain(ctx).status == "PASS"
    (data/'rows.json').write_text('[[-2, 1], [3, 0]]')
    assert checker.step_evaluate(ctx).status == "FAIL"
