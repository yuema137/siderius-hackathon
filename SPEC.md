# Track 1 submission specification

Scope: the exact format of a Galileo Hackathon Track 1 task submission and the
checks `tools/check_submission.py` applies. Owners: this file states the rules;
`tools/submission_rules.py` implements the static rules and JSON schemas,
`tools/check_submission.py` the steps, `tools/make_submission.py` the zip.
When this file and the code disagree, the code is the executed behavior and
the disagreement is a defect to report. Kit version 1.0.0.

A submission is a **task package only**: a scientific ML task that SIDERIUS
v1.1.0 can run in Track 2. It contains no workflow, no data and no credentials.
The data lives on Zenodo.

## 1. Layout

The upload is one zip, `<task_id>.zip`, with exactly one top folder:

```
<task_id>/
  TASK_CARD.md                    # required headings, section 5
  data_source.json                # Zenodo record and file checksums, section 2
  fetch_data.py                   # download + verify + process, section 3
  task/                           # SIDERIUS task package
    task.yaml                     #   composition manifest; paths relative to task/
    declared/ ...                 #   declarations task.yaml names
    plugins/ ...                  #   data path, metric(s), Health views, model plugin(s)
  baseline/run_baseline.py        # standalone baseline, section 4
  baseline/init_weights.(safetensors|pt)
  baseline/final_weights.(safetensors|pt)
  baseline/baseline_result.json   # section 4.2
```

- `<task_id>` matches `[A-Za-z0-9_-]{1,80}` (the SIDERIUS console task-id
  rule). It becomes the SIDERIUS task id: Track 2 copies `task/` to
  `siderius_workspace/tasks/<task_id>/` and the processed data to
  `siderius_workspace/data/<task_id>/`.
- `task/` follows the SIDERIUS
  [task composition reference](https://github.com/yuema137/SIDERIUS/blob/master/docs/reference/task-composition.md).
  Required manifest sections: `task_data_path`, `dataset_profile`, `metric`
  (declaration + implementation), `task_health` (`config:` or `none: true`),
  `task_config`. Optional and encouraged: `secondary_metrics`, `evaluation`,
  a real `task_health` roster, `parameter_rules`, `model_plugins` (the bonus).
  The task data path reads files only from the run's data folder
  (`data_dir`); it never downloads.
- Other files are allowed (helpers, split/manifest CSVs, a `README.md`)
  within the limits below.

## 1.1 Limits and refused files

| Rule | Value |
| --- | --- |
| zip size | <= 20 MB (20,971,520 bytes) |
| any single file (uncompressed) | <= 5 MB (5,242,880 bytes); equal passes |
| data arrays/tables `.npz .npy .h5 .hdf5 .parquet .pkl .pickle .feather` | refused outside `baseline/` |
| weights `.safetensors .pt .pth .ckpt .bin .onnx` | allowed only as `baseline/init_weights.{safetensors,pt}` and `baseline/final_weights.{safetensors,pt}`; exactly one format per stem |
| archives `.zip .tar .gz .tgz .bz2 .xz .7z .rar .zst` (any suffix in the name) | refused everywhere |
| symlinks, devices, FIFOs | refused |
| `__pycache__/ .git/ .DS_Store .ipynb_checkpoints/ .pytest_cache/ *.pyc *.pyo` | skipped when checking a folder and when building the zip; refused inside an uploaded zip |
| zip members | no absolute paths, no `..`, no backslashes, no symlinks; one top folder; expanded size <= 200 MB |

CSV split or manifest files (row ids per split, file lists) are allowed inside
the per-file limit. `.safetensors` is preferred for weights; a `.pt` file must
be a flat state dict `{name: tensor}` that loads with
`torch.load(path, map_location="cpu", weights_only=True)`.

## 2. `data_source.json`

Pydantic model `DataSource` (`extra="forbid"`):

| Field | Type | Rule |
| --- | --- | --- |
| `schema_version` | `1` | literal |
| `zenodo_record_id` | string | numeric id of the **published version record** (not the concept id); the worked example uses the placeholder `ZENODO_RECORD_ID` |
| `doi` | string | exactly `10.5281/zenodo.<zenodo_record_id>` |
| `access` | `"public"` or `"restricted"` | must match Zenodo's `metadata.access_right` (`open` / `restricted`) |
| `share_link` | string or null | restricted only; starts with `https://zenodo.org/`; a Zenodo secret link with *Can view* permission |
| `access_note` | string or null | restricted only; how organizers get access when there is no share link (for example: view access granted to a named organizer account) |
| `license` | string | Zenodo license id, compared case-insensitively with `metadata.license.id` |
| `files` | list, >= 1 | `{name, size, md5}`; `name` has no folder part and is unique; `size` > 0 bytes; `md5` 32 lower-case hex, equal to Zenodo's `checksum` without the `md5:` prefix |

A restricted record needs `share_link` or `access_note`; a public record has
neither. List every file `fetch_data.py` needs; Zenodo files not listed are
reported as a note and not checked.

## 3. `fetch_data.py`

```
python fetch_data.py --output_dir DIR [--source_dir SRC]
```

- Reads `data_source.json` from its own folder.
- Without `--source_dir`: downloads each listed file from Zenodo
  (`https://zenodo.org/api/records/<id>/files/<name>/content`). With
  `--source_dir`: uses the files already in `SRC` and does not download.
- Verifies every file's size and md5 against `data_source.json` before using it.
- Writes into `DIR` exactly the layout the task data path in `task/task.yaml`
  reads. `DIR` is the run's data folder; keep held-out test data out of it
  unless the task needs it there.
- Exit status 0 on success, non-zero on any failure. It must not need
  credentials other than a share link in `data_source.json`, must not write
  outside `DIR` and its own temporary folder, and should be safe to re-run.
- The checker runs it with the SIDERIUS interpreter; use the standard library
  or packages in SIDERIUS's environment (numpy, torch, PyYAML, ...).

## 4. Baseline

### 4.1 `baseline/run_baseline.py`

Runs with the SIDERIUS checkout's interpreter (`<SIDERIUS>/.venv/bin/python`),
because the task's own plugins import the framework.

```
run_baseline.py train    --data_dir D --init_weights W --seed S --output_dir O
run_baseline.py evaluate --data_dir D --weights W --output_dir O
```

- `D` is the folder `fetch_data.py --output_dir` wrote.
- `train` starts from the weights in `W`, uses seed `S` for every random choice
  it makes (batch order, dropout, augmentation), and writes
  `O/final_weights.safetensors` or `O/final_weights.pt` and `O/metrics.json`.
- `evaluate` loads `W` and writes `O/metrics.json`.
- `metrics.json` is a JSON object `{metric id: number}` with exactly the
  task's declared metric ids (primary plus `secondary_metrics`), computed on the
  evaluation split by the **task's own metric plugin(s)**, the same classes
  `task.yaml` binds. Values are finite.
- Exit status 0 on success. Load weights only with `weights_only=True` or
  safetensors. Other subcommands (for example an `init` helper) are allowed.
- A GPU is allowed if the baseline needs one; say so in `TASK_CARD.md` and
  `baseline_result.json`. The checker passes `CUDA_VISIBLE_DEVICES` through.

### 4.2 `baseline/baseline_result.json`

Pydantic model `BaselineResult` (`extra="forbid"`):

| Field | Type | Rule |
| --- | --- | --- |
| `schema_version` | `1` | literal |
| `seed` | int | passed as `--seed` in the retrain check |
| `evaluation_split` | string | the split `metrics.json` is computed on, for example `validation` |
| `metrics` | `{id: float}` | >= 1 entry, finite; the reported baseline |
| `tolerance` | `{id: float}` | same keys as `metrics`; absolute, >= 0; a reproduced value passes when `abs(reproduced - reported) <= tolerance` |
| `hardware` | `{device, description}` | `device` in `cpu, cuda, mps, other`; `description` names the CPU/GPU model and count |
| `wall_time` | `{train_seconds, evaluate_seconds}` | >= 0 |
| `framework_versions` | `{name: version}` | >= 1 entry, for example `python`, `torch`, `siderius` (version or git revision) |

The same tolerance applies to evaluating the final weights (4a) and to
retraining (4b). Choose it to cover nondeterminism of your hardware (GPU
kernels, thread count); 0 is valid for a deterministic CPU baseline.

## 5. `TASK_CARD.md`

Level-2 headings (`## `), matched case-insensitively, each with non-empty
text after HTML comments are removed:

1. `Scientific question`
2. `Task design`
3. `Data`: DOI (must contain the `doi` from `data_source.json` literally),
   size, license, access, processing
4. `Evaluation`: primary metric; optional secondary metrics, formula, Health
   checks, and why each relates to the science
5. `Baseline`: method, score, compute
6. `Novelty`
7. `Difficulty`
8. `Known shortcuts and leakage risks`
9. `Contact`

`templates/TASK_CARD.md` holds the headings with guidance.

## 6. Checker

```
python tools/check_submission.py <task_dir-or-zip> [--siderius CHECKOUT]
    [--data_dir DIR | --download] [--retrain] [--report out.json] [--keep_work]
    [--fetch_timeout S] [--siderius_timeout S] [--eval_timeout S] [--train_timeout S]
```

Run it with the SIDERIUS interpreter (it needs pydantic). Exit 0 only when no
step is `FAIL`; `SKIP` does not fail. The report JSON holds `verdict`,
`failed_steps`, `zip_sha256` (of the uploaded zip, or of the zip
`make_submission.py` would build from a folder), `zip_bytes`, the SIDERIUS
`path`/`version`/`git_revision`, and per step `id`, `name`, `status`,
`reason`, `details`.

| Step | PASS when | SKIP when |
| --- | --- | --- |
| `1` package | section 1 layout and limits hold; JSON files match their schemas; card headings present and non-empty; card states the DOI | never |
| `2a` zenodo_record | `GET https://zenodo.org/api/records/<id>` returns 200; the resolved `id` equals the declared id; DOI, access and license agree; every declared file is listed with the same size and `md5:` checksum. Restricted records whose file list is hidden pass with a note | `data_source.json` invalid |
| `2b` data_files | `--data_dir`: every declared file is there with matching size and md5, then `fetch_data.py --source_dir <data_dir>` exits 0 and writes at least one file. `--download`: `fetch_data.py` exits 0 and writes at least one file | neither flag; or `--download` on a restricted record without `share_link` |
| `3a` siderius_admission | in a private temporary folder: `candidate_console init --workspace ... --credential-file <mode-600 {}>`, `task/` copied to `tasks/<task_id>/`, processed data (or an empty folder) to `data/<task_id>/`, `add-task --config ... --manifest ... --no-gpu`, `check --config ...` all exit 0 | no `--siderius`; not Linux |
| `3b` siderius_composition | `workflows.task_composition.compose_run_task_bindings(task.yaml)` succeeds; reports primary/secondary metric ids and directions | no `--siderius` |
| `4.0` baseline_weights | both weight files load as flat state dicts (`weights_only=True` / safetensors) and are not byte-identical | no `--siderius`; weights missing |
| `4a` baseline_evaluate | `run_baseline.py evaluate` on the final weights exits 0 and its `metrics.json` keys equal the declared metric ids (from 3b) and the reported ones, each within tolerance | no `--siderius`, no processed data from 2b, or invalid `baseline_result.json` |
| `4b` baseline_retrain | with `--retrain`: `run_baseline.py train` from the initial weights and `seed` exits 0, writes `final_weights.*`, and its `metrics.json` passes the 4a comparison | no `--retrain`; same as 4a |

Facts the steps rely on:

- `add-task` and `check` (3a) validate the console entry, folders and
  permissions; they do not compose the manifest or load plugins (a manifest
  naming a non-existent metric symbol passes them; verified at SIDERIUS
  1ab94b51). Step 3b closes that gap with the composer SIDERIUS runs use.
- 3a does not start the bubblewrap run sandbox. The report records whether
  `bwrap` is usable on the machine; Track 2 runs need it.
- Participant code (`fetch_data.py`, `run_baseline.py`, plugins loaded in 3b)
  runs only as subprocesses: own session, killed with its process group on
  timeout, `umask 077`, working directory in the private temporary folder, and
  an environment reduced to `PATH`, `HOME`, `LANG`, `TMPDIR`,
  `CUDA_VISIBLE_DEVICES`, `LC_ALL`, `PYTHONHASHSEED=0`,
  `PYTHONDONTWRITEBYTECODE=1`. It is not sandboxed: check untrusted submissions
  in a disposable machine or container.
- Default timeouts: fetch 3600 s, each SIDERIUS command 600 s, evaluate 1800 s,
  retrain 21600 s.
- The temporary folder is removed at the end unless `--keep_work`.

## 7. `make_submission.py`

```
python tools/make_submission.py <task_dir> [--output_dir DIR]
```

Runs step 1 on the folder and refuses on any problem; leaves out the skipped
clutter; writes `DIR/<task_id>.zip` (never replacing an existing file);
prints the path, size and sha256. The zip is deterministic: sorted members
under `<task_id>/`, deflate level 9, timestamp 1980-01-01, mode 0644. Exit 0 on
success, 1 on refusal.

## 8. Worked example

`examples/quickstart_classification/` is a complete submission converted from
the SIDERIUS quickstart (synthetic two-class tabular data). Its data files are
the four quickstart shards generated with seed 20260824 (SHA-256 equal to the
pins in SIDERIUS `examples/quickstart/declared/data_manifest.json`); its
`data_source.json` holds their real sizes and md5 with the placeholder record
id, so step 2a fails until the organizers publish them. Its `task/` is the
SIDERIUS CSV classification template adapted to that data (file-backed labels,
not the quickstart's seed-carried scope labels), with primary `accuracy`,
secondary `balanced_accuracy`, two Health checks and the bonus model plugin.

## 9. Unresolved

- Whether Zenodo's files API honours a secret-link `token` query parameter for
  restricted records is unverified; the example `fetch_data.py` passes it on.
  Organizers can always check restricted data with `--data_dir`.
- The portal's own upload-time checks are not specified here; they are
  expected to repeat step 1 only (no participant code runs at upload).
