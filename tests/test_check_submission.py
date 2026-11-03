"""Checker and zip-builder tests. No network: the Zenodo query is a stub."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest

import check_submission as checker
import make_submission
from submission_rules import (
    MAX_FILE_BYTES,
    MAX_ZIP_BYTES,
    BaselineResult,
    DataSource,
    build_zip_bytes,
    check_static,
    compare_metrics,
    zip_member_problems,
)


def problems_of(root: Path) -> list[str]:
    return check_static(root).problems


def test_worked_example_passes_step_1(submission: Path) -> None:
    """The shipped example must stay a valid submission; any rule change that breaks it fails here."""
    result = check_static(submission)
    assert result.problems == []
    assert result.init_weights == "baseline/init_weights.pt"
    assert result.final_weights == "baseline/final_weights.pt"


# -- layout refusals ----------------------------------------------------------


@pytest.mark.parametrize("relative", ["task/declared/features.npz", "data/table.parquet", "x.h5", "cache.pkl"])
def test_data_files_outside_baseline_are_refused(submission: Path, relative: str) -> None:
    path = submission / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0")
    assert any(relative in p and "belong on Zenodo" in p for p in problems_of(submission))


def test_small_arrays_inside_baseline_are_allowed(submission: Path) -> None:
    (submission / "baseline" / "normalization.npz").write_bytes(b"0")
    assert problems_of(submission) == []


@pytest.mark.parametrize("relative", ["task/plugins/model.pt", "baseline/checkpoint.pt", "baseline/final_weights.ckpt"])
def test_weights_outside_the_two_named_files_are_refused(submission: Path, relative: str) -> None:
    (submission / relative).write_bytes(b"0")
    assert any(relative in p and "weights are allowed only" in p for p in problems_of(submission))


def test_two_formats_of_the_same_weights_are_refused(submission: Path) -> None:
    (submission / "baseline" / "init_weights.safetensors").write_bytes(b"0")
    assert any("more than one format" in p for p in problems_of(submission))


def test_missing_weights_and_required_files_are_named(submission: Path) -> None:
    (submission / "baseline" / "final_weights.pt").unlink()
    (submission / "fetch_data.py").unlink()
    problems = problems_of(submission)
    assert "missing required file fetch_data.py" in problems
    assert any("missing baseline/final_weights" in p for p in problems)


def test_nested_archives_and_symlinks_are_refused(submission: Path) -> None:
    (submission / "task" / "extra.tar.gz").write_bytes(b"0")
    os.symlink(submission / "TASK_CARD.md", submission / "task" / "link.md")
    problems = problems_of(submission)
    assert any("extra.tar.gz is an archive" in p for p in problems)
    assert any("link.md is a symlink" in p for p in problems)


def test_cache_clutter_is_ignored_not_refused(submission: Path) -> None:
    (submission / "task" / "plugins" / "__pycache__").mkdir()
    (submission / "task" / "plugins" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"0")
    result = check_static(submission)
    assert result.problems == []
    assert not any("__pycache__" in f for f in result.files)


def test_invalid_task_id_folder_name_is_refused(tmp_path: Path, submission: Path) -> None:
    renamed = submission.rename(tmp_path / "bad name")
    assert any("not a valid task id" in p for p in problems_of(renamed))


# -- size limits ----------------------------------------------------------------


def test_file_size_limit_is_inclusive(submission: Path) -> None:
    (submission / "task" / "declared" / "split.csv").write_bytes(b"a" * MAX_FILE_BYTES)
    assert problems_of(submission) == []
    (submission / "task" / "declared" / "split.csv").write_bytes(b"a" * (MAX_FILE_BYTES + 1))
    assert any("split.csv is" in p and "limit per file" in p for p in problems_of(submission))


def test_zip_over_20_mb_fails_step_1(submission: Path) -> None:
    static = check_static(submission)
    assert checker.step_package(submission, static, MAX_ZIP_BYTES, []).status == "PASS"
    step = checker.step_package(submission, static, MAX_ZIP_BYTES + 1, [])
    assert step.status == "FAIL" and "20 MB" in step.reason


# -- TASK_CARD.md -----------------------------------------------------------------


def test_card_missing_heading_and_comment_only_section(submission: Path) -> None:
    card = submission / "TASK_CARD.md"
    text = card.read_text()
    text = text.replace("## Novelty", "## Newness")
    head, _, rest = text.partition("## Difficulty\n")
    _, _, tail = rest.partition("\n## ")
    card.write_text(head + "## Difficulty\n<!-- still the template comment -->\n\n## " + tail)
    problems = problems_of(submission)
    assert "TASK_CARD.md lacks the heading '## Novelty'" in problems
    assert "TASK_CARD.md '## Difficulty' is empty" in problems


def test_card_must_state_the_declared_doi(submission: Path) -> None:
    card = submission / "TASK_CARD.md"
    card.write_text(card.read_text().replace("10.5281/zenodo.ZENODO_RECORD_ID", "see Zenodo"))
    assert any("must state the DOI" in p for p in problems_of(submission))


# -- JSON schemas -----------------------------------------------------------------


def _edit_json(path: Path, **changes: object) -> None:
    document = json.loads(path.read_text())
    document.update(changes)
    path.write_text(json.dumps(document))


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"doi": "10.5281/zenodo.1"}, "doi must be"),
        ({"access": "restricted"}, "needs share_link or access_note"),
        ({"zenodo_record_id": "abc", "doi": "10.5281/zenodo.abc"}, "numeric Zenodo record id"),
        ({"files": [{"name": "a/b.csv", "size": 1, "md5": "0" * 32}]}, "no folder part"),
        ({"files": [{"name": "a.csv", "size": 1, "md5": "XYZ"}]}, "files.0.md5"),
        ({"surprise": 1}, "surprise"),
    ],
)
def test_data_source_schema_errors_are_reported(submission: Path, changes: dict, expected: str) -> None:
    _edit_json(submission / "data_source.json", **changes)
    assert any(p.startswith("data_source.json") and expected in p for p in problems_of(submission))


def test_restricted_record_with_share_link_is_valid() -> None:
    source = DataSource.model_validate(
        {
            "schema_version": 1,
            "zenodo_record_id": "123",
            "doi": "10.5281/zenodo.123",
            "access": "restricted",
            "share_link": "https://zenodo.org/records/123?token=abc",
            "license": "cc-by-4.0",
            "files": [{"name": "a.csv", "size": 1, "md5": "0" * 32}],
        }
    )
    assert source.access == "restricted"


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"tolerance": {"accuracy": 0.01}}, "tolerance must name exactly"),
        ({"tolerance": {"accuracy": -1.0, "balanced_accuracy": 0.0}}, ">= 0"),
        ({"metrics": {"accuracy": float("nan"), "balanced_accuracy": 0.5}}, "not finite"),
        ({"seed": "abc"}, "seed"),
    ],
)
def test_baseline_result_schema_errors_are_reported(submission: Path, changes: dict, expected: str) -> None:
    _edit_json(submission / "baseline" / "baseline_result.json", **changes)
    assert any(p.startswith("baseline/baseline_result.json") and expected in p for p in problems_of(submission))


def test_invalid_json_is_reported_not_raised(submission: Path) -> None:
    (submission / "data_source.json").write_text("{not json")
    assert any(p.startswith("data_source.json is not valid JSON") for p in problems_of(submission))


# -- tolerance comparison --------------------------------------------------------


def _reported(**tolerance: float) -> BaselineResult:
    return BaselineResult.model_validate(
        {
            "schema_version": 1,
            "seed": 1,
            "evaluation_split": "validation",
            "metrics": {"a": 0.5, "b": 10.0},
            "tolerance": tolerance,
            "hardware": {"device": "cpu", "description": "x"},
            "wall_time": {"train_seconds": 1, "evaluate_seconds": 1},
            "framework_versions": {"python": "3.12"},
        }
    )


def test_difference_equal_to_tolerance_passes_and_larger_fails() -> None:
    reported = _reported(a=0.25, b=0.0)
    ok, rows, problems = compare_metrics(reported, {"a": 0.75, "b": 10.0})
    assert ok and all(row.within for row in rows) and problems == []
    ok, _, problems = compare_metrics(reported, {"a": 0.7500001, "b": 10.0})
    assert not ok and problems[0].startswith("a: reproduced")
    ok, _, problems = compare_metrics(reported, {"a": 0.5, "b": 10.0 + 1e-12})
    assert not ok and problems[0].startswith("b: reproduced")


def test_missing_and_extra_metric_ids_fail_comparison() -> None:
    ok, _, problems = compare_metrics(_reported(a=1.0, b=1.0), {"a": 0.5, "c": 1.0})
    assert not ok
    assert "metrics.json lacks ['b']" in problems
    assert "metrics.json has ids not in baseline_result.json: ['c']" in problems


# -- Zenodo record (stubbed) -----------------------------------------------------


def _source(submission: Path, record_id: str = "4242") -> DataSource:
    document = json.loads((submission / "data_source.json").read_text())
    document.update(zenodo_record_id=record_id, doi=f"10.5281/zenodo.{record_id}")
    return DataSource.model_validate(document)


def _record(source: DataSource, **overrides: object) -> dict:
    record = {
        "id": int(source.zenodo_record_id),
        "doi": source.doi,
        "metadata": {"access_right": "open", "license": {"id": "cc-by-4.0"}},
        "files": [{"key": f.name, "size": f.size, "checksum": f"md5:{f.md5}"} for f in source.files],
    }
    record.update(overrides)
    return record


def _stub(status: int, record: dict | None):
    calls = []

    def fetch(url: str):
        calls.append(url)
        return status, url, record

    return fetch, calls


def test_zenodo_record_matching_files_passes(submission: Path) -> None:
    source = _source(submission)
    fetch, calls = _stub(200, _record(source))
    step = checker.step_zenodo_record(source, fetch)
    assert step.status == "PASS", step.reason
    assert calls == ["https://zenodo.org/api/records/4242"]


def test_zenodo_md5_or_size_mismatch_fails(submission: Path) -> None:
    source = _source(submission)
    record = _record(source)
    record["files"][1]["checksum"] = "md5:" + "f" * 32
    record["files"][2]["size"] = 1
    step = checker.step_zenodo_record(source, _stub(200, record)[0])
    assert step.status == "FAIL"
    assert any("shard_0001.csv: Zenodo checksum" in p for p in step.details["problems"])
    assert any("shard_0002.csv: Zenodo size 1" in p for p in step.details["problems"])


def test_concept_id_resolving_to_another_version_fails(submission: Path) -> None:
    source = _source(submission)
    step = checker.step_zenodo_record(source, _stub(200, _record(source, id=4243))[0])
    assert step.status == "FAIL" and "version record id" in step.reason


def test_missing_record_and_placeholder_fail(submission: Path) -> None:
    source = _source(submission)
    assert checker.step_zenodo_record(source, _stub(404, None)[0]).status == "FAIL"
    placeholder = DataSource.model_validate_json((submission / "data_source.json").read_bytes())
    fetch, calls = _stub(200, {})
    step = checker.step_zenodo_record(placeholder, fetch)
    assert step.status == "FAIL" and "placeholder" in step.reason and calls == []


def test_restricted_record_without_public_file_list_passes_with_note(submission: Path) -> None:
    document = json.loads((submission / "data_source.json").read_text())
    document.update(
        zenodo_record_id="4242",
        doi="10.5281/zenodo.4242",
        access="restricted",
        access_note="organizers were granted view access on Zenodo",
    )
    source = DataSource.model_validate(document)
    record = _record(source, files=[])
    record["metadata"]["access_right"] = "restricted"
    step = checker.step_zenodo_record(source, _stub(200, record)[0])
    assert step.status == "PASS", step.reason
    assert any("step 2b verifies them" in n for n in step.details["notes"])


def test_declared_access_must_match_zenodo(submission: Path) -> None:
    source = _source(submission)
    record = _record(source)
    record["metadata"]["access_right"] = "restricted"
    step = checker.step_zenodo_record(source, _stub(200, record)[0])
    assert step.status == "FAIL" and "access_right" in step.reason


def test_network_error_is_a_failure_with_reason(submission: Path) -> None:
    def fetch(url: str):
        raise OSError("no route to host")

    step = checker.step_zenodo_record(_source(submission), fetch)
    assert step.status == "FAIL" and "rerun with network access" in step.reason


# -- zip building and reading ---------------------------------------------------


def test_zip_is_deterministic_and_excludes_clutter(submission: Path, tmp_path: Path) -> None:
    (submission / "task" / "__pycache__").mkdir()
    (submission / "task" / "__pycache__" / "a.pyc").write_bytes(b"0")
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir(), second.mkdir()
    path_a, size_a, digest_a = make_submission.make(submission, first)
    os.utime(submission / "TASK_CARD.md", (0, 0))  # timestamps must not change the bytes
    path_b, _, digest_b = make_submission.make(submission, second)
    assert digest_a == digest_b == hashlib.sha256(path_a.read_bytes()).hexdigest()
    assert path_a.name == "quickstart_classification.zip" and size_a == path_a.stat().st_size
    with zipfile.ZipFile(path_a) as archive:
        names = archive.namelist()
    assert all(name.startswith("quickstart_classification/") for name in names)
    assert not any("__pycache__" in name for name in names)
    assert "quickstart_classification/baseline/final_weights.pt" in names


def test_make_submission_refuses_failing_folder_and_existing_output(submission: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (submission / "data.npy").write_bytes(b"0")
    with pytest.raises(ValueError, match="step 1 refuses"):
        make_submission.make(submission, out)
    assert list(out.iterdir()) == []
    (submission / "data.npy").unlink()
    make_submission.make(submission, out)
    with pytest.raises(ValueError, match="already exists"):
        make_submission.make(submission, out)
    assert make_submission.main([str(submission), "--output_dir", str(out)]) == 1


def test_built_zip_round_trips_through_the_checker(submission: Path, tmp_path: Path) -> None:
    path, size, digest = make_submission.make(submission, tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    root, zip_size, problems, zip_digest = checker.open_input(path, work)
    assert (root.name, zip_size, problems, zip_digest) == ("quickstart_classification", size, [], digest)
    assert check_static(root).files == check_static(submission).files
    folder_digest = checker.open_input(submission, work)[3]
    assert folder_digest == digest  # a folder is measured as the zip make_submission would build


def _zip(tmp_path: Path, members: dict[str, bytes], symlink: str | None = None) -> zipfile.ZipFile:
    path = tmp_path / "upload.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "/etc/passwd")
    return zipfile.ZipFile(path)


def test_uploaded_zip_member_refusals(tmp_path: Path) -> None:
    archive = _zip(
        tmp_path, {"t/../../evil.py": b"", "t/a.py": b"", "t/__pycache__/a.pyc": b"", "t/x.npy": b""}, symlink="t/link"
    )
    task_id, problems = zip_member_problems(archive)
    assert task_id == "t"
    assert any("unsafe path" in p for p in problems)
    assert any("is a symlink" in p for p in problems)
    assert any("clutter" in p for p in problems)
    assert any("x.npy" in p and "Zenodo" in p for p in problems)


def test_zip_with_two_top_folders_is_refused(tmp_path: Path) -> None:
    task_id, problems = zip_member_problems(_zip(tmp_path, {"a/x.py": b"", "b/y.py": b""}))
    assert task_id is None and any("exactly one top folder" in p for p in problems)


def test_checker_refuses_to_unpack_unsafe_zip(tmp_path: Path) -> None:
    archive = _zip(tmp_path, {"t/../../evil.py": b""})
    archive.close()
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(SystemExit, match="refusing to unpack"):
        checker.open_input(tmp_path / "upload.zip", work)
    assert not (tmp_path.parent / "evil.py").exists()


def test_build_zip_bytes_matches_listed_files_only(submission: Path) -> None:
    files = check_static(submission).files
    with zipfile.ZipFile(__import__("io").BytesIO(build_zip_bytes(submission, files))) as archive:
        assert sorted(n.split("/", 1)[1] for n in archive.namelist()) == files


# -- driver without SIDERIUS ----------------------------------------------------


def test_without_siderius_or_data_the_dependent_steps_skip(submission: Path) -> None:
    args = checker.parse_args([str(submission)])
    report = checker.check(args, fetch=_stub(404, None)[0])
    statuses = {step["id"]: step["status"] for step in report["steps"]}
    assert statuses == {
        "1": "PASS",
        "2a": "FAIL",
        "2b": "SKIP",
        "3a": "SKIP",
        "3b": "SKIP",
        "4.0": "SKIP",
        "4a": "SKIP",
        "4b": "SKIP",
    }
    assert report["verdict"] == "FAIL" and report["failed_steps"] == ["2a"]
