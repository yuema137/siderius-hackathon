"""Static rules for a Track 1 task submission (checker step 1 and make_submission).

Stdlib + pydantic only. Nothing here imports or executes participant code.
SPEC.md is the agent-facing statement of every rule below; keep them in sync.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, ValidationError, field_validator, model_validator

KIT_VERSION = "1.0.0"

#: Oldest SIDERIUS a submission may be built and checked with (a git ref that must
#: be an ancestor of the --siderius checkout's HEAD). Until the v1.1.0 tag exists
#: this is the pinned commit; at release change this one line to "v1.1.0".
MIN_SIDERIUS = "1ab94b5101f705f2bd90ba89a21fd10bb2ed98de"

MAX_ZIP_BYTES = 20 * 1024 * 1024
MAX_FILE_BYTES = 5 * 1024 * 1024
#: Zip-bomb bounds applied before anything is unpacked.
MAX_EXPANDED_BYTES = 200 * 1024 * 1024
MAX_ZIP_MEMBERS = 10_000

#: The record-id placeholder the worked example ships with. A real submission
#: replaces it with the numeric Zenodo record id.
PLACEHOLDER_RECORD_ID = "ZENODO_RECORD_ID"

#: The zip's single top folder: the author's working name, NOT the task id.
FOLDER_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,80}")
#: Task ids are assigned by the submission portal when the author creates a task.
PORTAL_TASK_ID_PATTERN = re.compile(r"t1-[0-9]{3,}")
LOCAL_ID_LABEL = "local working id (the portal assigns the real id)"
PORTAL_ID_LABEL = "portal-assigned task id"


def local_task_id(folder_name: str | None) -> str:
    """A safe SIDERIUS console task id derived from the working folder name.

    Used only when no portal id (``--task_id``) is given; the portal assigns
    the real id. Unsafe characters become ``-``; at most 80 characters.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "-", folder_name or "")[:80].strip("-")
    return cleaned or "local-task"


def portal_task_id_problem(value: str) -> str | None:
    if PORTAL_TASK_ID_PATTERN.fullmatch(value):
        return None
    return f"--task_id {value!r} is not a portal task id (t1-NNN, for example t1-007)"


#: Data arrays and tables belong on Zenodo; refused outside baseline/ (and inside
#: baseline/ unless the file is the model artifact itself, e.g. final_model.pkl).
DATA_SUFFIXES = frozenset({".npz", ".npy", ".h5", ".hdf5", ".parquet", ".pkl", ".pickle", ".feather"})
#: Model-like files; refused outside baseline/.
MODEL_SUFFIXES = frozenset(
    {".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".onnx", ".keras", ".joblib", ".skops", ".cbm", ".ubj", ".pb"}
)
#: Non-model files allowed inside baseline/ besides the model artifacts.
BASELINE_CONFIG_SUFFIXES = frozenset({".py", ".json", ".yaml", ".yml", ".txt", ".md"})

FRAMEWORKS = ("pytorch", "tensorflow", "sklearn", "xgboost", "lightgbm", "catboost", "other")
#: model_format ids: the text after ``final_model.`` (``savedmodel`` is a directory).
MODEL_FORMATS = (
    "pt",
    "onnx",
    "keras",
    "weights.h5",
    "savedmodel",
    "json",
    "ubj",
    "txt",
    "cbm",
    "skops",
    "joblib",
    "pkl",
)
#: Allowed formats per framework, recommended first.
FRAMEWORK_FORMATS: dict[str, tuple[str, ...]] = {
    "pytorch": ("pt", "onnx"),
    "tensorflow": ("keras", "weights.h5", "savedmodel", "onnx"),
    "xgboost": ("json", "ubj"),
    "lightgbm": ("txt",),
    "catboost": ("cbm",),
    "sklearn": ("skops", "joblib", "pkl"),
    "other": MODEL_FORMATS,
}
#: Neural-network frameworks must ship init_model next to final_model.
NEURAL_FRAMEWORKS = frozenset({"pytorch", "tensorflow"})
#: Formats that execute code when loaded; the checker never loads them.
PICKLE_FORMATS = frozenset({"joblib", "pkl"})
MODEL_STEMS = ("init_model", "final_model")
REQUIREMENTS = "baseline/requirements.txt"

#: Archives could smuggle data or weights past the rules above; refused everywhere.
ARCHIVE_SUFFIXES = frozenset({".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".zst"})
#: Editor/cache clutter: skipped when building a zip and when checking a folder,
#: refused inside an uploaded zip.
IGNORED_NAMES = frozenset({"__pycache__", ".DS_Store", ".git", ".ipynb_checkpoints", ".pytest_cache"})
IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})

REQUIRED_FILES = (
    "TASK_CARD.md",
    "data_source.json",
    "fetch_data.py",
    "task/task.yaml",
    "baseline/run_baseline.py",
    "baseline/baseline_result.json",
)

#: Level-2 headings TASK_CARD.md must contain (any order; matched case-insensitively).
CARD_HEADINGS = (
    "Scientific question",
    "Task design",
    "Data",
    "Evaluation",
    "Baseline",
    "Novelty",
    "Difficulty",
    "Known shortcuts and leakage risks",
    "Contact",
)


# ---------------------------------------------------------------------------
# JSON schemas
# ---------------------------------------------------------------------------


class ZenodoFile(BaseModel):
    """One file of the Zenodo record, as Zenodo's API reports it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, description="file key on Zenodo (no folders)")
    size: int = Field(gt=0, description="bytes")
    md5: str = Field(pattern=r"^[0-9a-f]{32}$", description="lower-case hex; Zenodo reports md5:<hex>")

    @field_validator("name")
    @classmethod
    def _plain_name(cls, value: str) -> str:
        if "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("a Zenodo file name has no folder part")
        return value


class DataSource(BaseModel):
    """data_source.json: where the task's data lives and how to verify it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    zenodo_record_id: str = Field(description="numeric id of the published Zenodo version record (not the concept id)")
    doi: str = Field(description="10.5281/zenodo.<zenodo_record_id>")
    access: Literal["public", "restricted"]
    share_link: str | None = Field(
        default=None,
        description="restricted only: a Zenodo 'Can view' secret link for the organizers",
    )
    access_note: str | None = Field(
        default=None,
        description="restricted only: how organizers get access when there is no share link",
    )
    license: str = Field(min_length=1, description="license id as shown on Zenodo, e.g. cc-by-4.0")
    files: list[ZenodoFile] = Field(min_length=1)

    @field_validator("zenodo_record_id")
    @classmethod
    def _record_id(cls, value: str) -> str:
        if value != PLACEHOLDER_RECORD_ID and not re.fullmatch(r"[1-9][0-9]*", value):
            raise ValueError("must be the numeric Zenodo record id")
        return value

    @field_validator("share_link")
    @classmethod
    def _share_link(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("https://zenodo.org/"):
            raise ValueError("a share link starts with https://zenodo.org/")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> DataSource:
        if self.doi != f"10.5281/zenodo.{self.zenodo_record_id}":
            raise ValueError(f"doi must be 10.5281/zenodo.{self.zenodo_record_id}")
        if self.access == "public" and (self.share_link or self.access_note):
            raise ValueError("share_link and access_note are only for restricted records")
        if self.access == "restricted" and not (self.share_link or self.access_note):
            raise ValueError("a restricted record needs share_link or access_note")
        names = [entry.name for entry in self.files]
        if len(set(names)) != len(names):
            raise ValueError("files lists a name twice")
        return self

    @property
    def is_placeholder(self) -> bool:
        return self.zenodo_record_id == PLACEHOLDER_RECORD_ID


def _finite(values: dict[str, float], what: str) -> dict[str, float]:
    for key, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{what}[{key!r}] is not finite")
    return values


class Hardware(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    device: Literal["cpu", "cuda", "mps", "other"]
    description: str = Field(min_length=1, description="CPU/GPU model, count, memory")


class WallTime(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    train_seconds: float = Field(ge=0)
    evaluate_seconds: float = Field(ge=0)


class BaselineResult(BaseModel):
    """baseline/baseline_result.json: the author's reported baseline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    framework: Literal["pytorch", "tensorflow", "sklearn", "xgboost", "lightgbm", "catboost", "other"]
    model_format: Literal[
        "pt", "onnx", "keras", "weights.h5", "savedmodel", "json", "ubj", "txt", "cbm", "skops", "joblib", "pkl"
    ] = Field(description="suffix of baseline/final_model.<format>; savedmodel is the directory baseline/final_model/")
    hyperparameters: dict[str, Any] = Field(description="everything needed to retrain, with the seed")
    seed: int
    evaluation_split: str = Field(min_length=1, description="the split metrics.json is computed on")
    metrics: dict[str, StrictFloat] = Field(min_length=1, description="by declared metric id")
    tolerance: dict[str, float] = Field(
        description="absolute reproduction tolerance per metric id (evaluate and retrain)"
    )
    hardware: Hardware
    wall_time: WallTime
    framework_versions: dict[str, str] = Field(min_length=1, description="e.g. python, torch, siderius")

    @model_validator(mode="after")
    def _consistent(self) -> BaselineResult:
        _finite(self.metrics, "metrics")
        _finite(self.tolerance, "tolerance")
        if set(self.tolerance) != set(self.metrics):
            raise ValueError("tolerance must name exactly the metric ids in metrics")
        if any(value < 0 for value in self.tolerance.values()):
            raise ValueError("tolerance values must be >= 0")
        if self.model_format not in FRAMEWORK_FORMATS[self.framework]:
            raise ValueError(
                f"model_format {self.model_format!r} is not allowed for framework {self.framework!r}; "
                f"use one of {list(FRAMEWORK_FORMATS[self.framework])}"
            )
        return self


class MetricsFile(BaseModel):
    """metrics.json written by run_baseline.py: {metric id: value}."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metrics: dict[str, StrictFloat] = Field(min_length=1)

    @model_validator(mode="after")
    def _finite_values(self) -> MetricsFile:
        _finite(self.metrics, "metrics.json")
        return self


@dataclass(frozen=True)
class MetricComparison:
    metric_id: str
    reported: float
    reproduced: float | None
    tolerance: float
    within: bool


def compare_metrics(
    reported: BaselineResult, reproduced: dict[str, float]
) -> tuple[bool, list[MetricComparison], list[str]]:
    """Compare reproduced metrics with the reported ones under the declared tolerance.

    Returns ``(ok, per-metric comparisons, problems)``. Missing or extra metric
    ids are problems; a difference equal to the tolerance passes.
    """
    problems: list[str] = []
    missing = sorted(set(reported.metrics) - set(reproduced))
    extra = sorted(set(reproduced) - set(reported.metrics))
    if missing:
        problems.append(f"metrics.json lacks {missing}")
    if extra:
        problems.append(f"metrics.json has ids not in baseline_result.json: {extra}")
    rows = []
    for metric_id in sorted(reported.metrics):
        value = reproduced.get(metric_id)
        tolerance = reported.tolerance[metric_id]
        within = value is not None and abs(value - reported.metrics[metric_id]) <= tolerance
        if value is not None and not within:
            problems.append(
                f"{metric_id}: reproduced {value!r}, reported {reported.metrics[metric_id]!r}, "
                f"|difference| {abs(value - reported.metrics[metric_id]):.3g} > tolerance {tolerance!r}"
            )
        rows.append(MetricComparison(metric_id, reported.metrics[metric_id], value, tolerance, within))
    return not problems, rows, problems


# ---------------------------------------------------------------------------
# File inventory
# ---------------------------------------------------------------------------


class PackageRefused(ValueError):
    """The input cannot be read as a submission at all."""


def is_ignored(relative: str) -> bool:
    parts = PurePosixPath(relative).parts
    return any(part in IGNORED_NAMES for part in parts) or PurePosixPath(relative).suffix in IGNORED_SUFFIXES


def list_package_files(root: Path) -> tuple[list[str], list[str]]:
    """``(regular files as POSIX paths relative to root, problems)``, sorted.

    Ignored clutter is skipped. Symlinks and special files are problems, never followed.
    """
    files: list[str] = []
    problems: list[str] = []
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(current)
        for name in sorted(dirnames):
            path = base / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                problems.append(f"{relative} is a symlink; submissions hold regular files only")
        dirnames[:] = sorted(name for name in dirnames if not (base / name).is_symlink() and name not in IGNORED_NAMES)
        for name in sorted(filenames):
            path = base / name
            relative = path.relative_to(root).as_posix()
            if is_ignored(relative):
                continue
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode):
                problems.append(f"{relative} is a symlink; submissions hold regular files only")
            elif not stat.S_ISREG(mode):
                problems.append(f"{relative} is not a regular file")
            else:
                files.append(relative)
    return sorted(files), problems


def model_stem_of(relative: str) -> str | None:
    """``init_model`` / ``final_model`` when ``relative`` is part of a model artifact."""
    parts = PurePosixPath(relative).parts
    if len(parts) < 2 or parts[0] != "baseline":
        return None
    for stem in MODEL_STEMS:
        if len(parts) == 2 and parts[1].startswith(stem + "."):
            return stem
        if len(parts) > 2 and parts[1] == stem:  # SavedModel directory
            return stem
    return None


def file_rule_problems(relative: str, size: int) -> list[str]:
    """Problems for one file by path and size (shared by folder and zip checks)."""
    problems = []
    path = PurePosixPath(relative)
    suffix = path.suffix.lower()
    suffixes = {s.lower() for s in path.suffixes}
    top = path.parts[0] if path.parts else ""
    if size > MAX_FILE_BYTES:
        problems.append(f"{relative} is {size} bytes; the limit per file is {MAX_FILE_BYTES} (5 MB)")
    if suffixes & ARCHIVE_SUFFIXES:
        problems.append(f"{relative} is an archive; archives are not allowed inside a submission")
    if top != "baseline":
        if suffix in DATA_SUFFIXES:
            problems.append(f"{relative}: {suffix} data files belong on Zenodo, not in the package")
        elif suffix in MODEL_SUFFIXES:
            problems.append(
                f"{relative}: model files are allowed only as baseline/init_model.<format> and "
                "baseline/final_model.<format>"
            )
    elif model_stem_of(relative) is None and suffix not in BASELINE_CONFIG_SUFFIXES:
        problems.append(
            f"{relative}: baseline/ may hold only init_model/final_model artifacts and "
            ".py, .json, .yaml, .yml, .txt or .md files"
        )
    return problems


@dataclass(frozen=True)
class ModelArtifact:
    stem: str  # init_model | final_model
    model_format: str
    path: str  # relative file, or the SavedModel directory
    files: tuple[str, ...]  # every relative file that belongs to it


def find_model_artifact(files: list[str], stem: str) -> tuple[ModelArtifact | None, list[str]]:
    """``(the one artifact named stem, problems)``; ``(None, [])`` when absent."""
    single = [
        f
        for f in files
        if PurePosixPath(f).parent.as_posix() == "baseline" and PurePosixPath(f).name.startswith(stem + ".")
    ]
    folder = [f for f in files if f.startswith(f"baseline/{stem}/")]
    found: list[ModelArtifact] = []
    problems: list[str] = []
    for relative in single:
        model_format = PurePosixPath(relative).name[len(stem) + 1 :]
        if model_format not in MODEL_FORMATS or model_format == "savedmodel":
            problems.append(f"{relative}: unknown model format {model_format!r}; see SPEC.md section 4.3")
            continue
        found.append(ModelArtifact(stem, model_format, relative, (relative,)))
    if folder:
        if f"baseline/{stem}/saved_model.pb" not in folder:
            problems.append(f"baseline/{stem}/ is a folder but not a SavedModel (no saved_model.pb)")
        else:
            found.append(ModelArtifact(stem, "savedmodel", f"baseline/{stem}", tuple(folder)))
    if len(found) > 1:
        problems.append(f"baseline/{stem} exists in more than one format: {[a.path for a in found]}; keep one")
        return None, problems
    return (found[0] if found else None), problems


def model_pair_problems(
    result: BaselineResult | None, init: ModelArtifact | None, final: ModelArtifact | None
) -> list[str]:
    """The init/final pair rule and the agreement with baseline_result.json."""
    problems = []
    if final is None:
        problems.append("missing baseline/final_model.<format> (or the SavedModel folder baseline/final_model/)")
    if result is None:
        return problems
    if final is not None and final.model_format != result.model_format:
        problems.append(
            f"{final.path} has format {final.model_format!r} but baseline_result.json says {result.model_format!r}"
        )
    if result.framework in NEURAL_FRAMEWORKS:
        if init is None:
            problems.append(f"framework {result.framework!r} needs baseline/init_model.{result.model_format}")
        elif init.model_format != result.model_format:
            problems.append(f"{init.path} must use the same format as the final model ({result.model_format!r})")
    elif init is not None and result.framework != "other":
        problems.append(
            f"{init.path}: framework {result.framework!r} has no initial model; remove it "
            "(reproduction rests on seed and hyperparameters)"
        )
    elif init is not None and final is not None and init.model_format != final.model_format:
        problems.append(f"{init.path} must use the same format as {final.path}")
    return problems


_REQUIREMENT = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?(?:\[[A-Za-z0-9._-]+(?:,[A-Za-z0-9._-]+)*\])?"
    r"==[A-Za-z0-9][A-Za-z0-9.+!_-]*"
    r"(?:\s*;\s*[A-Za-z0-9_.<>=!~'\" ()]+)?"
)


def requirements_problems(text: str) -> tuple[list[str], list[str]]:
    """``(pinned requirement lines, problems)`` for baseline/requirements.txt.

    Every requirement is ``name[extras]==version`` with an optional environment
    marker. Options (``-e``, ``-r``, ``--index-url``, ...), URLs, direct
    references and local paths are refused: the checker installs only pinned
    packages from the default index.
    """
    pins: list[str] = []
    problems: list[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = re.split(r"\s#", raw, maxsplit=1)[0].strip()
        if not line or line.startswith("#"):
            continue
        where = f"{REQUIREMENTS} line {number} ({line!r})"
        if any(token.startswith("-") for token in line.split()):
            problems.append(f"{where}: options such as -e, -r, -c and --index-url are not allowed")
        elif "://" in line or " @ " in line or line.startswith("@"):
            problems.append(f"{where}: URLs and direct references are not allowed; pin a released version")
        elif "/" in line or "\\" in line or line.startswith("."):
            problems.append(f"{where}: local paths are not allowed")
        elif not _REQUIREMENT.fullmatch(line):
            problems.append(f"{where}: pin an exact version, for example scikit-learn==1.7.2")
        else:
            pins.append(line)
    return pins, problems


# ---------------------------------------------------------------------------
# TASK_CARD.md
# ---------------------------------------------------------------------------

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def card_sections(text: str) -> dict[str, str]:
    """``{level-2 heading: body without HTML comments}`` (heading text stripped)."""
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in _COMMENT.sub("", text).splitlines():
        match = re.match(r"^##\s+(.+?)\s*#*\s*$", line)
        if match and not line.startswith("###"):
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current, lines = match.group(1).strip(), []
        elif current is not None:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()
    return sections


def card_problems(text: str, data_source: DataSource | None) -> list[str]:
    sections = card_sections(text)
    lowered = {key.lower(): value for key, value in sections.items()}
    problems = []
    for heading in CARD_HEADINGS:
        body = lowered.get(heading.lower())
        if body is None:
            problems.append(f"TASK_CARD.md lacks the heading '## {heading}'")
        elif not body:
            problems.append(f"TASK_CARD.md '## {heading}' is empty")
    if data_source is not None and data_source.doi not in lowered.get("data", ""):
        problems.append(f"TASK_CARD.md '## Data' must state the DOI {data_source.doi}")
    return problems


# ---------------------------------------------------------------------------
# Step 1 on a folder
# ---------------------------------------------------------------------------


@dataclass
class StaticResult:
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    data_source: DataSource | None = None
    baseline_result: BaselineResult | None = None
    init_model: ModelArtifact | None = None
    final_model: ModelArtifact | None = None
    requirements: list[str] | None = None  # None: no baseline/requirements.txt


def _schema_problems(error: ValidationError, name: str) -> list[str]:
    out = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "(document)"
        out.append(f"{name}: {location}: {item['msg']}")
    return out


def load_json_model(path: Path, model: type[BaseModel], name: str) -> tuple[BaseModel | None, list[str]]:
    try:
        raw = path.read_bytes()
        json.loads(raw)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        return None, [f"{name} is not valid JSON: {error}"]
    try:
        return model.model_validate_json(raw), []
    except ValidationError as error:
        return None, _schema_problems(error, name)


def check_static(root: Path) -> StaticResult:
    """Checker step 1 on an unpacked submission folder (any safe working name)."""
    result = StaticResult()
    if not FOLDER_NAME_PATTERN.fullmatch(root.name):
        result.problems.append(
            f"folder name {root.name!r} is not a safe folder name (letters, digits, '-', '_'; at most 80)"
        )
    files, problems = list_package_files(root)
    result.files = files
    result.problems.extend(problems)
    present = set(files)
    for required in REQUIRED_FILES:
        if required not in present:
            result.problems.append(f"missing required file {required}")
    result.init_model, init_problems = find_model_artifact(files, "init_model")
    result.final_model, final_problems = find_model_artifact(files, "final_model")
    result.problems.extend(init_problems + final_problems)
    if REQUIREMENTS in present:
        try:
            pins, problems = requirements_problems((root / REQUIREMENTS).read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            pins, problems = [], [f"{REQUIREMENTS} is not UTF-8 text"]
        result.requirements = pins
        result.problems.extend(problems)
    for relative in files:
        result.problems.extend(file_rule_problems(relative, (root / relative).stat().st_size))
    if "data_source.json" in present:
        model, problems = load_json_model(root / "data_source.json", DataSource, "data_source.json")
        result.data_source = model  # type: ignore[assignment]
        result.problems.extend(problems)
        if result.data_source is not None and result.data_source.is_placeholder:
            result.notes.append(
                f"data_source.json uses the placeholder record id {PLACEHOLDER_RECORD_ID}; step 2a fails until it is replaced"
            )
    if "baseline/baseline_result.json" in present:
        model, problems = load_json_model(
            root / "baseline/baseline_result.json", BaselineResult, "baseline/baseline_result.json"
        )
        result.baseline_result = model  # type: ignore[assignment]
        result.problems.extend(problems)
    result.problems.extend(model_pair_problems(result.baseline_result, result.init_model, result.final_model))
    if "TASK_CARD.md" in present:
        try:
            text = (root / "TASK_CARD.md").read_text(encoding="utf-8")
        except UnicodeDecodeError:
            result.problems.append("TASK_CARD.md is not UTF-8 text")
        else:
            result.problems.extend(card_problems(text, result.data_source))
    return result


# ---------------------------------------------------------------------------
# Zip reading and writing
# ---------------------------------------------------------------------------

FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def build_zip_bytes(root: Path, files: list[str]) -> bytes:
    """Deterministic zip: one top folder ``<folder>/``, sorted, fixed time, mode 0644."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in sorted(files):
            info = zipfile.ZipInfo(f"{root.name}/{relative}", date_time=FIXED_TIME)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (root / relative).read_bytes(), compresslevel=9)
    return buffer.getvalue()


def zip_member_problems(archive: zipfile.ZipFile) -> tuple[str | None, list[str]]:
    """``(top folder name, problems)`` for the members of an uploaded zip, before extraction."""
    problems = []
    tops = set()
    total = 0
    members = archive.infolist()
    if len(members) > MAX_ZIP_MEMBERS:
        problems.append(f"the zip has {len(members)} members; the limit is {MAX_ZIP_MEMBERS}")
    seen: set[str] = set()
    for info in members:
        name = info.filename
        if name.rstrip("/") in seen:
            problems.append(f"zip member {name!r} appears more than once")
            continue
        seen.add(name.rstrip("/"))
        path = PurePosixPath(name)
        if name.startswith("/") or "\\" in name or ".." in path.parts or not path.parts:
            problems.append(f"zip member {name!r} has an unsafe path")
            continue
        tops.add(path.parts[0])
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            problems.append(f"zip member {name!r} is a symlink")
            continue
        if info.is_dir():
            continue
        if len(path.parts) < 2:
            problems.append(f"zip member {name!r} is outside the top folder")
            continue
        relative = PurePosixPath(*path.parts[1:]).as_posix()
        if is_ignored(relative):
            problems.append(f"zip member {name!r} is cache/editor clutter; rebuild with make_submission.py")
            continue
        total += info.file_size
        problems.extend(file_rule_problems(relative, info.file_size))
    if len(tops) != 1:
        problems.append(f"the zip must contain exactly one top folder; found {sorted(tops)}")
        return None, problems
    if total > MAX_EXPANDED_BYTES:
        problems.append(f"the zip declares {total} bytes when unpacked; the limit is {MAX_EXPANDED_BYTES} (200 MB)")
    return tops.pop(), problems


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
