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
    assert result.init_model is not None and result.init_model.path == "baseline/init_model.pt"
    assert result.final_model is not None and result.final_model.path == "baseline/final_model.pt"
    assert result.requirements is None


# -- layout refusals ----------------------------------------------------------


@pytest.mark.parametrize("relative", ["task/declared/features.npz", "data/table.parquet", "x.h5", "cache.pkl"])
def test_data_files_outside_baseline_are_refused(submission: Path, relative: str) -> None:
    path = submission / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0")
    assert any(relative in p and "belong on Zenodo" in p for p in problems_of(submission))


@pytest.mark.parametrize("relative", ["baseline/normalization.npz", "baseline/stats.pkl", "baseline/table.csv"])
def test_baseline_folder_holds_only_model_artifacts_and_small_configs(submission: Path, relative: str) -> None:
    (submission / relative).write_bytes(b"0")
    assert any(relative in p and "may hold only init_model/final_model" in p for p in problems_of(submission))


@pytest.mark.parametrize("relative", ["baseline/config.yaml", "baseline/notes.md", "baseline/helpers.py"])
def test_baseline_config_and_code_files_are_allowed(submission: Path, relative: str) -> None:
    (submission / relative).write_text("x")
    assert problems_of(submission) == []


@pytest.mark.parametrize("relative", ["task/plugins/model.pt", "model.keras", "task/m.joblib", "x.onnx"])
def test_model_files_outside_baseline_are_refused(submission: Path, relative: str) -> None:
    (submission / relative).write_bytes(b"0")
    assert any(relative in p and "model files are allowed only" in p for p in problems_of(submission))


def test_unknown_or_duplicate_model_formats_are_refused(submission: Path) -> None:
    (submission / "baseline" / "final_model.safetensors").write_bytes(b"0")
    (submission / "baseline" / "init_model.onnx").write_bytes(b"0")
    problems = problems_of(submission)
    assert any("final_model.safetensors: unknown model format 'safetensors'" in p for p in problems)
    assert any("baseline/init_model exists in more than one format" in p for p in problems)


def test_missing_final_model_and_required_files_are_named(submission: Path) -> None:
    (submission / "baseline" / "final_model.pt").unlink()
    (submission / "fetch_data.py").unlink()
    problems = problems_of(submission)
    assert "missing required file fetch_data.py" in problems
    assert any(p.startswith("missing baseline/final_model.<format>") for p in problems)


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


def test_unsafe_folder_name_is_refused(tmp_path: Path, submission: Path) -> None:
    renamed = submission.rename(tmp_path / "bad name")
    assert any("'bad name' is not a safe folder name" in p for p in problems_of(renamed))


# -- size limits ----------------------------------------------------------------


def test_file_size_limit_is_inclusive(submission: Path) -> None:
    (submission / "task" / "declared" / "split.csv").write_bytes(b"a" * MAX_FILE_BYTES)
    assert problems_of(submission) == []
    (submission / "task" / "declared" / "split.csv").write_bytes(b"a" * (MAX_FILE_BYTES + 1))
    assert any("split.csv is" in p and "limit per file" in p for p in problems_of(submission))


def test_zip_over_20_mb_fails_step_1(submission: Path) -> None:
    static = check_static(submission)
    assert checker.step_package(submission.name, static, MAX_ZIP_BYTES, []).status == "PASS"
    step = checker.step_package(submission.name, static, MAX_ZIP_BYTES + 1, [])
    assert step.status == "FAIL" and "20 MiB" in step.reason


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
            "framework": "sklearn",
            "model_format": "skops",
            "hyperparameters": {},
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
    assert "quickstart_classification/baseline/final_model.pt" in names


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
    opened = checker.open_input(path, work)
    assert opened.root is not None
    assert (opened.folder, opened.zip_bytes, opened.problems, opened.zip_sha256) == (
        "quickstart_classification",
        size,
        [],
        digest,
    )
    assert check_static(opened.root).files == check_static(submission).files
    # a folder is measured as the zip make_submission would build
    assert checker.open_input(submission, work).zip_sha256 == digest


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
    opened = checker.open_input(tmp_path / "upload.zip", work)
    assert opened.root is None and any("not unpacked" in p for p in opened.problems)
    assert not (tmp_path.parent / "evil.py").exists() and list(work.iterdir()) == []


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
        "3.0": "SKIP",
        "3a": "SKIP",
        "3b": "SKIP",
        "4.0": "SKIP",
        "4.1": "SKIP",
        "4a": "SKIP",
        "4b": "SKIP",
    }
    assert report["verdict"] == "FAIL" and report["failed_steps"] == ["2a"]


# -- bounded unpacking (zip bombs, malformed zips) ------------------------------


def _write_zip(path: Path, members: dict[str, bytes], compression: int = zipfile.ZIP_DEFLATED) -> Path:
    with zipfile.ZipFile(path, "w", compression) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def _unpacked_bytes(work: Path) -> int:
    return sum(p.stat().st_size for p in work.rglob("*") if p.is_file())


def test_highly_compressed_oversized_member_is_never_unpacked(tmp_path: Path) -> None:
    """A small zip declaring a member above 5 MiB is refused before any byte is written."""
    upload = _write_zip(tmp_path / "t.zip", {"t/big.csv": b"0" * (MAX_FILE_BYTES + 1)})
    assert upload.stat().st_size < 100_000
    work = tmp_path / "work"
    work.mkdir()
    opened = checker.open_input(upload, work)
    assert opened.root is None
    assert any("big.csv is" in p and "limit per file" in p for p in opened.problems)
    assert _unpacked_bytes(work) == 0


def test_zip_over_the_upload_limit_is_not_opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upload = _write_zip(tmp_path / "t.zip", {"t/a.py": b"x" * 2000}, zipfile.ZIP_STORED)
    monkeypatch.setattr(checker, "MAX_ZIP_BYTES", 1000)
    work = tmp_path / "work"
    work.mkdir()
    opened = checker.open_input(upload, work)
    assert opened.root is None and "not unpacked" in opened.problems[0]
    assert _unpacked_bytes(work) == 0


def test_expanded_total_and_member_count_bounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import submission_rules

    upload = _write_zip(tmp_path / "t.zip", {f"t/f{i}.py": b"x" * 100 for i in range(5)})
    monkeypatch.setattr(submission_rules, "MAX_EXPANDED_BYTES", 400)
    monkeypatch.setattr(submission_rules, "MAX_ZIP_MEMBERS", 4)
    with zipfile.ZipFile(upload) as archive:
        _, problems = zip_member_problems(archive)
    assert any("declares 500 bytes when unpacked" in p for p in problems)
    assert any("has 5 members; the limit is 4" in p for p in problems)


def test_duplicate_member_names_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "t.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("t/a.py", b"first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("t/a.py", b"second")
    with zipfile.ZipFile(path) as archive:
        _, problems = zip_member_problems(archive)
    assert any("'t/a.py' appears more than once" in p for p in problems)


@pytest.mark.parametrize("case", ["file_folder_collision", "corrupt_data"])
def test_malformed_zip_is_a_step_1_failure_not_a_crash(tmp_path: Path, case: str) -> None:
    if case == "file_folder_collision":
        upload = _write_zip(tmp_path / "t.zip", {"t/a": b"file", "t/a/b.py": b"inside"})
    else:
        upload = _write_zip(tmp_path / "t.zip", {"t/a.py": b"A" * 1000}, zipfile.ZIP_STORED)
        raw = bytearray(upload.read_bytes())
        start = raw.index(b"A" * 1000)
        raw[start : start + 10] = b"B" * 10  # CRC no longer matches
        upload.write_bytes(bytes(raw))
    report = checker.check(checker.parse_args([str(upload)]), fetch=_stub(404, None)[0])
    assert report["verdict"] == "FAIL" and report["failed_steps"] == ["1"]
    assert "could not be read or unpacked" in report["steps"][0]["reason"]
    assert all(step["status"] == "SKIP" for step in report["steps"][1:])


def test_folder_with_oversized_file_is_not_read_into_a_zip(submission: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (submission / "task" / "huge.csv").write_bytes(b"0" * (MAX_FILE_BYTES + 1))

    def refuse(*_: object) -> bytes:
        raise AssertionError("build_zip_bytes must not run when a file exceeds 5 MiB")

    monkeypatch.setattr(checker, "build_zip_bytes", refuse)
    report = checker.check(checker.parse_args([str(submission)]), fetch=_stub(404, None)[0])
    assert report["zip_bytes"] is None and report["steps"][0]["status"] == "FAIL"


# -- verdict, subprocess isolation ----------------------------------------------


def _steps(**statuses: str) -> list[checker.Step]:
    ids = {"s1": "1", "s3a": "3a", "s3b": "3b", "s40": "4.0", "s41": "4.1", "s4a": "4a", "s4b": "4b"}
    return [checker.Step(ids[key], key, status, "") for key, status in statuses.items()]


def test_verdict_distinguishes_incomplete_from_pass_and_fail() -> None:
    passing = _steps(s1="PASS", s3a="PASS", s3b="PASS", s40="PASS", s41="PASS", s4a="PASS", s4b="SKIP")
    assert checker.verdict_of(passing)[0] == "PASS"
    verdict, _, skipped = checker.verdict_of(
        _steps(s1="PASS", s3a="SKIP", s3b="PASS", s40="PASS", s41="SKIP", s4a="SKIP")
    )
    assert verdict == "INCOMPLETE" and skipped == ["3a", "4.1", "4a"]
    assert checker.verdict_of(_steps(s1="FAIL", s3a="SKIP"))[0] == "FAIL"


def test_all_static_steps_passing_without_siderius_is_incomplete_exit_2(
    submission: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("data_source.json", "TASK_CARD.md"):
        path = submission / name
        path.write_text(path.read_text().replace("ZENODO_RECORD_ID", "4242"))
    source = DataSource.model_validate_json((submission / "data_source.json").read_bytes())
    monkeypatch.setattr(checker, "fetch_json", _stub(200, _record(source))[0])
    assert checker.main([str(submission)]) == 2


def test_timeout_returns_even_if_a_grandchild_escapes_the_group(tmp_path: Path) -> None:
    """An escaped grandchild holding stdout must not keep the checker waiting."""
    import sys
    import time

    script = "import subprocess, time; subprocess.Popen(['setsid', 'sleep', '20']); time.sleep(20)"
    started = time.monotonic()
    done = checker.run([sys.executable, "-c", script], cwd=tmp_path, timeout=1, env=checker.child_env(tmp_path))
    assert done.returncode is None
    assert time.monotonic() - started < 10


def test_child_env_has_private_home_and_passes_proxies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
    monkeypatch.setenv("no_proxy", "localhost")
    monkeypatch.setenv("SSL_CERT_FILE", "/etc/ssl/cert.pem")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    env = checker.child_env(tmp_path)
    assert env["HOME"] == str(tmp_path / "child_home") and (tmp_path / "child_home").is_dir()
    assert env["HTTPS_PROXY"] == "http://proxy:3128" and env["no_proxy"] == "localhost"
    assert env["SSL_CERT_FILE"] == "/etc/ssl/cert.pem" and "OPENAI_API_KEY" not in env


def test_composition_step_cannot_import_modules_planted_in_the_work_folder(submission: Path, tmp_path: Path) -> None:
    """fetch_data.py can write into the work folder; 3b must not import from it.

    Uses the base interpreter (no SIDERIUS installed), so only a planted module could make 3b pass.
    """
    import sys

    base = Path(getattr(sys, "_base_executable", sys.executable))
    if base == Path(sys.executable):
        pytest.skip("needs a base interpreter distinct from the SIDERIUS virtualenv")

    work = tmp_path / "work"
    (work / "workflows").mkdir(parents=True)
    (work / "workflows" / "__init__.py").write_text("")
    (work / "workflows" / "task_composition.py").write_text(
        "import json\n"
        "def compose_run_task_bindings(path):\n"
        f"    print({checker.RESULT_MARKER!r} + json.dumps({{'primary_metric': {{'id': 'x', 'direction': 'higher'}},"
        " 'secondary_metrics': [], 'task_health': 'none'}))\n"
        "    raise SystemExit(0)\n"
    )
    ctx = checker.Context(
        root=submission,
        work=work,
        args=checker.parse_args([str(submission)]),
        static=check_static(submission),
        python=base,
    )
    step = checker.step_composition(ctx)
    assert step.status == "FAIL", step.reason
    assert "ModuleNotFoundError" in step.details["output_tail"]


# -- portal task ids ---------------------------------------------------------------


@pytest.mark.parametrize("value", ["t1-007", "t1-123", "t1-0001"])
def test_portal_task_ids_are_accepted(value: str) -> None:
    assert checker.parse_args(["x", "--task_id", value]).task_id == value


@pytest.mark.parametrize("value", ["t1-7", "t1-07", "T1-007", "t2-007", "quickstart", "t1-007x", "t1-"])
def test_malformed_task_ids_are_refused(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        checker.parse_args(["x", "--task_id", value])
    assert exited.value.code == 2 and "is not a portal task id (t1-NNN" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("folder", "expected"),
    [
        ("galaxy_morphology", "galaxy_morphology"),
        ("my task.v2", "my-task-v2"),
        ("", "local-task"),
        (None, "local-task"),
    ],
)
def test_local_working_id_is_a_safe_console_id(folder: str | None, expected: str) -> None:
    from submission_rules import FOLDER_NAME_PATTERN, local_task_id

    assert local_task_id(folder) == expected and FOLDER_NAME_PATTERN.fullmatch(local_task_id(folder))


def test_report_labels_the_task_id(submission: Path) -> None:
    local = checker.check(checker.parse_args([str(submission)]), fetch=_stub(404, None)[0])
    assert (local["task_id"], local["folder"]) == ("quickstart_classification", "quickstart_classification")
    assert local["task_id_kind"] == "local working id (the portal assigns the real id)"
    portal = checker.check(checker.parse_args([str(submission), "--task_id", "t1-007"]), fetch=_stub(404, None)[0])
    assert (portal["task_id"], portal["task_id_kind"], portal["folder"]) == (
        "t1-007",
        "portal-assigned task id",
        "quickstart_classification",
    )


def test_admission_registers_the_task_under_the_given_id(
    submission: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 3a passes --task-id and uses tasks/<id>/ and data/<id>/, whatever the folder is called."""
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path, timeout: float, env: dict[str, str], tail_lines: int = 40):
        calls.append(command)
        if "init" in command:
            (Path(command[command.index("--workspace") + 1]) / "siderius_workspace" / "settings").mkdir(parents=True)
        return checker.Completed(command, 0, timeout, "")

    monkeypatch.setattr(checker, "run", fake_run)
    monkeypatch.setattr(checker, "bwrap_status", lambda: "usable")
    work = tmp_path / "work"
    work.mkdir()
    ctx = checker.Context(
        root=submission,
        work=work,
        args=checker.parse_args([str(submission), "--siderius", str(tmp_path), "--task_id", "t1-007"]),
        static=check_static(submission),
        python=Path("/usr/bin/python3"),
        task_id="t1-007",
    )
    step = checker.step_admission(ctx)
    assert step.status == "PASS" and "--task-id t1-007" in step.reason
    add_task = next(call for call in calls if "add-task" in call)
    assert add_task[add_task.index("--task-id") + 1] == "t1-007"
    workspace = work / "siderius" / "project" / "siderius_workspace"
    assert add_task[add_task.index("--manifest") + 1] == str(workspace / "tasks" / "t1-007" / "task.yaml")
    assert (workspace / "data" / "t1-007").is_dir() and not (workspace / "tasks" / "quickstart_classification").exists()


def test_make_submission_names_the_zip_after_the_portal_id(submission: Path, tmp_path: Path) -> None:
    path, _, _ = make_submission.make(submission, tmp_path, "t1-007")
    assert path.name == "t1-007.zip"
    with zipfile.ZipFile(path) as archive:
        assert all(name.startswith("quickstart_classification/") for name in archive.namelist())
    with pytest.raises(ValueError, match="not a portal task id"):
        make_submission.make(submission, tmp_path, "007")


# -- minimum SIDERIUS revision (step 3.0) ---------------------------------------


def _git(repo: Path, *arguments: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.org", *arguments],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def history(tmp_path: Path) -> tuple[Path, str, str]:
    """A checkout with two commits: ``(repo, older, newer)``; HEAD is ``newer``."""
    repo = tmp_path / "siderius"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "pyproject.toml").write_text('[project]\nname = "siderius"\nversion = "1.0.0"\n')
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "older")
    older = _git(repo, "rev-parse", "HEAD")
    (repo / "pyproject.toml").write_text('[project]\nname = "siderius"\nversion = "1.1.0"\n')
    _git(repo, "commit", "-q", "-am", "newer")
    return repo, older, _git(repo, "rev-parse", "HEAD")


def test_checkout_containing_the_minimum_passes(history, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, older, newer = history
    for minimum in (older, newer):
        monkeypatch.setattr(checker, "MIN_SIDERIUS", minimum)
        step = checker.step_siderius_version(repo, checker.siderius_identity(repo))
        assert step.status == "PASS", step.reason
        assert step.details["version"] == "1.1.0" and step.details["git_revision"] == newer


def test_checkout_older_than_the_minimum_fails(history, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, older, newer = history
    _git(repo, "checkout", "-q", older)
    monkeypatch.setattr(checker, "MIN_SIDERIUS", newer)
    step = checker.step_siderius_version(repo, checker.siderius_identity(repo))
    assert step.status == "FAIL" and "is older than the required" in step.reason


def test_unknown_minimum_ref_fails_with_fetch_advice(history, monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _, _ = history
    monkeypatch.setattr(checker, "MIN_SIDERIUS", "v1.1.0")
    step = checker.step_siderius_version(repo, checker.siderius_identity(repo))
    assert step.status == "FAIL" and "update your SIDERIUS checkout: git fetch --tags" in step.reason


def test_non_git_checkout_fails(tmp_path: Path) -> None:
    step = checker.step_siderius_version(tmp_path, None)
    assert step.status == "FAIL" and "not a git checkout" in step.reason
