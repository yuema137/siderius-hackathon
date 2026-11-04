# Track 1 submission specification

Scope: the exact format of a Galileo Hackathon Track 1 task submission and the
checks `tools/check_submission.py` applies. Owners: this file states the rules;
`tools/submission_rules.py` implements the static rules and JSON schemas,
`tools/check_submission.py` the steps, `tools/make_submission.py` the zip.
When this file and the code disagree, the code is the executed behavior and
the disagreement is a defect to report. Kit version 1.0.0.

A submission is a **task package only**: a scientific ML task that SIDERIUS can
run in Track 2. The reference SIDERIUS is <https://github.com/yuema137/SIDERIUS>
at commit `1ab94b5101f705f2bd90ba89a21fd10bb2ed98de` or later (its
`pyproject.toml` still says 1.0.0); the 1.1.0 release tag replaces the commit
once published. The checker records the version and git revision it used.
A submission contains no workflow, no data and no credentials; the data lives
on Zenodo.

## 1. Task ids and layout

**Task ids are assigned by the submission portal**, not by the package. The
flow is:

1. sign in to the portal;
2. create a task with a title; the portal assigns its id, `t1-NNN` (for
   example `t1-007`; regex `^t1-[0-9]{3,}$`) and registers the title;
3. build and check the package locally (sections 6 and 7);
4. upload the zip to that task; each upload is a new version of it.

The zip's top folder name is only the author's working name; it is not the task
id and the portal ignores it. The `TASK_CARD.md` title heading is free text; the
portal's title is the registered one.

The upload is one zip with exactly one top folder:

```
<folder>/                         # any safe working name
  TASK_CARD.md                    # required headings, section 5
  data_source.json                # Zenodo record and file checksums, section 2
  fetch_data.py                   # download + verify + process, section 3
  task/                           # SIDERIUS task package
    task.yaml                     #   composition manifest; paths relative to task/
    declared/ ...                 #   declarations task.yaml names
    plugins/ ...                  #   data path, metric(s), Health views, model plugin(s)
  baseline/run_baseline.py        # standalone baseline, section 4.1
  baseline/final_model.<format>   # required; formats in section 4.3
  baseline/init_model.<format>    # required for PyTorch and TensorFlow/Keras only
  baseline/requirements.txt       # optional pinned extra packages, section 4.4
  baseline/baseline_result.json   # section 4.2
```

A TensorFlow SavedModel is a folder: `baseline/final_model/` holding
`saved_model.pb` (and `variables/`), and likewise `baseline/init_model/`.

- `<folder>` matches `[A-Za-z0-9_-]{1,80}` (safe-name rule). Track 2 registers
  the package under the portal-assigned id: it copies `task/` to
  `siderius_workspace/tasks/<task id>/` and the processed data to
  `siderius_workspace/data/<task id>/` and runs `add-task --task-id <task id>`.
- `task/` follows the SIDERIUS
  [task composition reference](https://github.com/yuema137/SIDERIUS/blob/master/docs/reference/task-composition.md).
  Required manifest sections: `task_data_path`, `dataset_profile`, `metric`
  (declaration + implementation), `task_health` (`config:` or `none: true`),
  `task_config`. Optional and encouraged: `secondary_metrics`, `evaluation`,
  a real `task_health` roster, `parameter_rules`, `model_plugins` (the bonus;
  **PyTorch only**, because the SIDERIUS trainer is PyTorch. The required
  standalone baseline may use any framework in section 4.3).
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
| model-like files `.safetensors .pt .pth .ckpt .bin .onnx .keras .joblib .skops .cbm .ubj .pb` | refused outside `baseline/` |
| `baseline/` content | only the model artifacts `init_model.<format>` / `final_model.<format>` (or the SavedModel folders `init_model/`, `final_model/`) and `.py .json .yaml .yml .txt .md` files; anything else (for example `normalization.npz`, `stats.pkl`, a CSV) is refused |
| model artifacts | one `final_model` (required) and at most one `init_model`; a known format (section 4.3); an artifact in two formats is refused |
| archives `.zip .tar .gz .tgz .bz2 .xz .7z .rar .zst` (any suffix in the name) | refused everywhere |
| symlinks, devices, FIFOs | refused |
| `__pycache__/ .git/ .DS_Store .ipynb_checkpoints/ .pytest_cache/ *.pyc *.pyo` | skipped when checking a folder and when building the zip; refused inside an uploaded zip |
| zip members | no absolute paths, no `..`, no backslashes, no symlinks, no duplicate names; one top folder; at most 10,000 members; declared expanded size <= 200 MB |

CSV split or manifest files (row ids per split, file lists) are allowed outside
`baseline/` inside the per-file limit. Model artifacts are subject to the same
5 MB per-file limit.

An uploaded zip is unpacked only when it passes every zip-level rule above:
the checker refuses to unpack a zip over 20 MB, any member declaring more than
5 MB, a declared expanded total over 200 MB, more than 10,000 members, an
unsafe or duplicate member name, or a symlink. Extraction never writes more
than a member's declared size. An unreadable or malformed zip (bad CRC, a file
and a folder with the same name) is a step 1 FAIL; the other steps SKIP.
A folder whose files include one over 5 MB is not built into a zip for
measuring; step 1 reports the file.

## 2. `data_source.json`

Pydantic model `DataSource` (`extra="forbid"`):

| Field | Type | Rule |
| --- | --- | --- |
| `schema_version` | `1` | literal |
| `zenodo_record_id` | string | numeric id of the **published version record** (not the concept id); the worked example uses the placeholder `ZENODO_RECORD_ID` |
| `doi` | string | exactly `10.5281/zenodo.<zenodo_record_id>` |
| `access` | `"public"` or `"restricted"` | must match Zenodo's `metadata.access_right` (`open` / `restricted`) |
| `share_link` | string or null | restricted only; starts with `https://zenodo.org/`; a Zenodo secret link with *Can view* permission, expiring no earlier than 2026-12-31 |
| `access_note` | string or null | restricted only; required when there is no `share_link`: states that the record is shared with the organizers' Zenodo account (and which account) |
| `license` | string | Zenodo license id, compared case-insensitively with `metadata.license.id` |
| `files` | list, >= 1 | `{name, size, md5}`; `name` has no folder part and is unique; `size` > 0 bytes; `md5` 32 lower-case hex, equal to Zenodo's `checksum` without the `md5:` prefix |

A restricted record needs one of two access routes: a *Can view* secret link
(record page, **Share**, **Links**; expiry no earlier than 2026-12-31) in
`share_link`, or the record shared with the organizers' Zenodo account,
explained in `access_note`. The schema enforces that one of the two fields is
present; the expiry date and the account sharing are checked by the organizers,
not by the checker. A public record has neither field. List every file `fetch_data.py` needs; Zenodo files not listed are
reported as a note and not checked.

## 3. `fetch_data.py`

```
python fetch_data.py --output_dir DIR [--source_dir SRC]
```

- Reads `data_source.json` from its own folder.
- Without `--source_dir`: downloads each listed file from Zenodo
  (`https://zenodo.org/api/records/<id>/files/<name>/content`). For a
  restricted record, downloading with a secret-link token is not verified by
  the kit; the fallback is `--source_dir` with files downloaded in a browser,
  which the checker uses through `--data_dir`. With
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

The standalone baseline is required and may use any framework in section 4.3.
The SIDERIUS model plugin (`task.yaml` `model_plugins`) is a separate, optional
bonus and must be PyTorch.

### 4.1 `baseline/run_baseline.py`

Runs in the baseline environment (section 4.4): the SIDERIUS checkout's
interpreter, or an isolated venv with SIDERIUS plus `requirements.txt`. The
task's own plugins import the framework, so SIDERIUS is always installed.

```
run_baseline.py train    --data_dir D --seed S --output_dir O [--init_model M]
run_baseline.py evaluate --data_dir D --model M --output_dir O
```

- `D` is the folder `fetch_data.py --output_dir` wrote.
- `train` uses seed `S` for every random choice it makes (initialization not
  read from a file, batch order, dropout, augmentation, tree randomness) and
  writes `O/final_model.<format>` (or the SavedModel folder `O/final_model/`)
  in the declared `model_format`, plus `O/metrics.json`. The checker passes
  `--init_model` exactly when `baseline/init_model.*` exists (PyTorch and
  TensorFlow/Keras); classical models are reproduced from seed and
  hyperparameters alone.
- `evaluate` loads the model `M` and writes `O/metrics.json`.
- `metrics.json` is a JSON object `{metric id: number}` with exactly the
  task's declared metric ids (primary plus `secondary_metrics`), computed on the
  evaluation split by the **task's own metric plugin(s)**, the same classes
  `task.yaml` binds. Values are finite numbers.
- Exit status 0 on success. Load models safely: `torch.load(weights_only=True)`,
  `keras.saving.load_model` in safe mode, skops with an explicit trusted-type
  list. Other subcommands (for example an `init` helper) are allowed.
- A GPU is allowed if the baseline needs one; say so in `TASK_CARD.md` and
  `baseline_result.json`. The checker passes `CUDA_VISIBLE_DEVICES` through.

### 4.2 `baseline/baseline_result.json`

Pydantic model `BaselineResult` (`extra="forbid"`):

| Field | Type | Rule |
| --- | --- | --- |
| `schema_version` | `1` | literal |
| `framework` | enum | `pytorch`, `tensorflow` (TensorFlow or Keras, any backend), `sklearn`, `xgboost`, `lightgbm`, `catboost`, `other` |
| `model_format` | enum | the text after `final_model.`: `pt`, `onnx`, `keras`, `weights.h5`, `savedmodel` (folder), `json`, `ubj`, `txt`, `cbm`, `skops`, `joblib`, `pkl`; must be allowed for `framework` (section 4.3) and match the file |
| `hyperparameters` | object | everything besides the seed needed to retrain (model settings, optimizer, epochs, ...) |
| `seed` | int | passed as `--seed` in the retrain check |
| `evaluation_split` | string | the split `metrics.json` is computed on, for example `validation` |
| `metrics` | `{id: number}` | >= 1 entry, finite; strings and booleans refused; the reported baseline |
| `tolerance` | `{id: float}` | same keys as `metrics`; absolute, >= 0; a reproduced value passes when `abs(reproduced - reported) <= tolerance` |
| `hardware` | `{device, description}` | `device` in `cpu, cuda, mps, other`; `description` names the CPU/GPU model and count |
| `wall_time` | `{train_seconds, evaluate_seconds}` | >= 0 |
| `framework_versions` | `{name: version}` | >= 1 entry, for example `python`, `torch`, `scikit-learn`, `siderius` (version or git revision) |

The same tolerance applies to evaluating the final model (4a) and to
retraining (4b). Choose it to cover nondeterminism of your hardware (GPU
kernels, thread count); 0 is valid for a deterministic CPU baseline.
Reproduction of a classical model rests on `seed` plus `hyperparameters`; a
neural network also starts from `init_model`.

### 4.3 Model formats and the init/final pair rule

| `framework` | Recommended | Also allowed | `init_model` |
| --- | --- | --- | --- |
| `pytorch` | `pt` | `onnx` | required, same format |
| `tensorflow` | `keras` | `weights.h5`, `savedmodel`, `onnx` | required, same format |
| `xgboost` | `json`, `ubj` | | not allowed |
| `lightgbm` | `txt` | | not allowed |
| `catboost` | `cbm` | | not allowed |
| `sklearn` | `skops` | `joblib`, `pkl` (discouraged: loading them executes code) | not allowed |
| `other` | any format above | | optional, same format |

Step 4.0 checks each artifact without executing participant objects:

| Format | Check |
| --- | --- |
| `pt` | a `torch.save` zip archive; then, with `--siderius`, `torch.load(path, map_location="cpu", weights_only=True)` in a `python -I` subprocess must return a non-empty flat `{name: tensor}` state dict |
| `onnx` | first byte `0x08` (ModelProto `ir_version`); with `--siderius`, `onnx.checker.check_model` in a subprocess only if `onnx` is installed in the SIDERIUS venv (it is not at 1ab94b51; then the header check is all) |
| `keras` | a valid zip whose directory lists `config.json` and `model.weights.h5`; not loaded |
| `weights.h5` | HDF5 signature `\x89HDF\r\n\x1a\n`; not loaded |
| `savedmodel` | `saved_model.pb` present and non-empty; not loaded |
| `json` (XGBoost) | parses as a JSON object with a `learner` key |
| `ubj` (XGBoost) | starts with `{` |
| `txt` (LightGBM) | starts with `tree` |
| `cbm` (CatBoost) | starts with `CBM1` |
| `skops` | a valid zip whose directory lists `schema.json`; not loaded |
| `joblib`, `pkl` | non-empty only. **Never unpickled by the checker**; loaded only by your `run_baseline.py` inside the baseline environment |

Every artifact must be non-empty, and the final model must not be
byte-identical to the initial model.

### 4.4 Baseline environment: `baseline/requirements.txt`

Optional. **It applies only to `baseline/run_baseline.py`.** The task package
in `task/` (data path, metrics, Health views, model plugin) runs in the SIDERIUS
venv in step 3b and in Track 2, so its plugins may import only packages
SIDERIUS installs.

Each non-comment line is `name[extras]==version`, optionally with an
environment marker (`numpy==2.4.3; python_version >= "3.12"`) written with the
marker characters `A-Z a-z 0-9 _ . < > = ! ~ ' " ( )` and spaces. A comment
starts at `#` after a space or tab. Refused: any whitespace-separated token
starting with `-` anywhere on the line (`-e`, `-r`, `-c`, `--index-url`,
`--extra-index-url`, also after a marker), URLs and direct references
(`pkg @ git+https://...`), local paths, and anything not pinned with `==` (no
`>=`, no wildcards). **Every pinned package and everything it depends on must
be available as a wheel** for Linux x86_64 and the SIDERIUS Python (3.12):
source-only packages are refused in step 4.1 because building them runs their
setup code.

- **Without the file**, the baseline runs with `<SIDERIUS>/.venv/bin/python`.
- **With the file**, step 4.1 builds an isolated venv with
  [uv](https://docs.astral.sh/uv/) inside the checker's work folder, never
  writing into the SIDERIUS checkout:
  1. copy the checkout's tracked files (`git ls-files`, including uncommitted
     edits) into the work folder;
  2. `uv build --wheel` from that copy (SIDERIUS is installed non-editable);
  3. `uv export --frozen --no-dev --no-emit-project --no-hashes` of the copy's
     `uv.lock` as constraints, so shared packages (torch, numpy, pydantic, ...)
     keep SIDERIUS's locked versions;
  4. `uv venv --python <SIDERIUS>/.venv/bin/python` (same Python);
  5. pass 1: `uv pip install <siderius wheel> -c <constraints>`. Only
     SIDERIUS's own locked dependencies are installed here, and only they may
     build from source (at 1ab94b51 `pylatexenc==2.10` has no wheel);
  6. pass 2: `uv pip install --no-build -r <validated pins> -c <constraints>`.
     The participant's pins and their transitive dependencies must be wheels;
     a source-only package fails 4.1 with "has no wheel ... source builds are
     not allowed".

  A pin that conflicts with SIDERIUS's locked versions fails 4.1. Every uv
  command runs with the same minimal environment as participant subprocesses
  (private `HOME` and `TMPDIR`, proxy and `SSL_CERT_FILE` variables) plus only
  the caller's `UV_*` variables and `PIP_INDEX_URL`, never the full caller
  environment. When `UV_CACHE_DIR` is unset, the checker sets it to the
  caller's uv cache (`uv cache dir`), so the private `HOME` does not force a
  cold cache. The step reports the resolved versions of the pinned packages,
  `siderius`, `torch` and `numpy`.
- **Network**: building the venv downloads from the package index. A network
  failure makes 4.1 SKIP (and therefore 4a/4b SKIP and the verdict
  INCOMPLETE); a missing `uv` SKIPs too.
- **Cost**: with a warm uv cache on the same filesystem the venv builds in
  seconds (measured: about 1 s for the install, about 25 s for the whole
  check of the scikit-learn example) and its files are hard links into the
  cache. With a cold cache uv downloads SIDERIUS's full dependency set,
  dominated by PyTorch and its CUDA libraries (several GB; the installed venv
  occupies about 7.6 GB, measured at 1ab94b51), plus your pins. Budget minutes
  and disk accordingly; the `--env_timeout` default is 3600 s.

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
python tools/check_submission.py <task_dir-or-zip> [--siderius CHECKOUT] [--task_id t1-NNN]
    [--data_dir DIR | --download] [--retrain] [--report out.json] [--keep_work]
    [--fetch_timeout S] [--siderius_timeout S] [--eval_timeout S] [--train_timeout S]
```

Run it with the SIDERIUS interpreter (it needs pydantic). Verdicts and exit
status:

| Verdict | Exit | When |
| --- | --- | --- |
| `PASS` | 0 | no step FAILs and the required steps 3a, 3b, 4.0, 4.1 and 4a ran |
| `FAIL` | 1 | any step FAILs |
| `INCOMPLETE` | 2 | no step FAILs but a required step was SKIPPED (4b may be skipped) |

`--task_id` is the id the portal assigned; it must match `^t1-[0-9]{3,}$`
(otherwise the command exits 2 with a usage error). Step 3a registers the task
with `add-task --task-id <id>`. Without it, the checker derives a safe local id
from the folder name (unsafe characters become `-`; empty becomes
`local-task`) and labels it "local working id (the portal assigns the real
id)". The report records `task_id`, `task_id_kind` (`portal-assigned task id`
or that label) and `folder`.

The report JSON holds `verdict`, `failed_steps`, `skipped_required_steps`,
`zip_sha256` (of the uploaded zip, or of the zip `make_submission.py` would
build from a folder; null when not measured), `zip_bytes`, the SIDERIUS
`path`/`version`/`git_revision`, and per step `id`, `name`, `status`,
`reason`, `details`.

| Step | PASS when | SKIP when |
| --- | --- | --- |
| `1` package | section 1 layout and limits hold; JSON files match their schemas; card headings present and non-empty; card states the DOI | never |
| `2a` zenodo_record | `GET https://zenodo.org/api/records/<id>` returns 200; the resolved `id` equals the declared id; DOI, access and license agree; every declared file is listed with the same size and `md5:` checksum. Restricted records whose file list is hidden pass with a note | `data_source.json` invalid |
| `2b` data_files | `--data_dir`: every declared file is there with matching size and md5, then `fetch_data.py --source_dir <data_dir>` exits 0 and writes at least one file. `--download`: `fetch_data.py` exits 0 and writes at least one file | neither flag; or `--download` on a restricted record without `share_link` |
| `3a` siderius_admission | in a private temporary folder: `candidate_console init --workspace ... --credential-file <mode-600 {}>`, `task/` copied to `tasks/<id>/`, processed data (or an empty folder) to `data/<id>/`, `add-task --config ... --manifest ... --task-id <id> --no-gpu`, `check --config ...` all exit 0 | no `--siderius`; not Linux |
| `3b` siderius_composition | `workflows.task_composition.compose_run_task_bindings(task.yaml)` succeeds; reports primary/secondary metric ids and directions | no `--siderius` |
| `4.0` baseline_model | every model artifact passes its section 4.3 format check; the final model differs from the initial model | final model or `baseline_result.json` missing/invalid; `.pt`/`.onnx` without `--siderius` |
| `4.1` baseline_environment | no `requirements.txt` (SIDERIUS venv used), or the isolated venv of section 4.4 was built (two passes; pins as wheels only) | no `--siderius`; invalid `requirements.txt` (step 1 fails); `uv` missing; network failure |
| `4a` baseline_evaluate | `run_baseline.py evaluate --model <final model>` in the baseline environment exits 0 and its `metrics.json` keys equal the declared metric ids (from 3b) and the reported ones, each within tolerance | no `--siderius`, no environment from 4.1, no processed data from 2b, or invalid `baseline_result.json` |
| `4b` baseline_retrain | with `--retrain`: `run_baseline.py train` with `seed` (and `--init_model` when present) exits 0, writes `final_model.<model_format>` (or `final_model/`), and its `metrics.json` passes the 4a comparison | no `--retrain`; same as 4a |

Facts the steps rely on:

- `add-task` and `check` (3a) validate the console entry, folders and
  permissions; they do not compose the manifest or load plugins (a manifest
  naming a non-existent metric symbol passes them; verified at SIDERIUS
  1ab94b51). Step 3b closes that gap with the composer SIDERIUS runs use.
- 3a does not start the bubblewrap run sandbox. The report records whether
  `bwrap` is usable on the machine; Track 2 runs need it.
- PASS on 2b, 4a and 4b means the participant's scripts ran and produced
  numbers consistent with `baseline_result.json` (and, for 4a/4b, keyed by
  the declared metric ids). It does not prove that `run_baseline.py` used the
  task's metric plugin, that `fetch_data.py` processes the data as the card
  says, or that the baseline is scientifically sound. The organizers review the
  code.
- Participant code (`fetch_data.py`, `run_baseline.py`, plugins loaded in 3b)
  runs only as subprocesses: own session; on timeout the process group is
  killed and the wait for it is bounded (10 s); leftover group members are
  killed after a normal exit too; output goes to a temporary file, not a pipe;
  `umask 077`; a fresh empty working folder per step; and an environment
  reduced to `PATH`, `LANG`, a private `HOME` and `TMPDIR` inside the work
  folder, `PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1`,
  `PYTHONUNBUFFERED=1`, plus `CUDA_VISIBLE_DEVICES`, `LC_ALL`, `HTTP_PROXY`,
  `HTTPS_PROXY`, `NO_PROXY` (both cases) and `SSL_CERT_FILE` when set. The
  checker's own snippets in 3b and 4.0 run with `python -I`. `uv` (step 4.1)
  is the checker's own tool and runs with the caller's environment. Participant code
  is not sandboxed: check untrusted submissions in a disposable machine or
  container.
- Default timeouts: fetch 3600 s, each SIDERIUS command 600 s, baseline
  environment 3600 s per uv command, evaluate 1800 s, retrain 21600 s.
- The temporary folder is removed at the end unless `--keep_work`.

## 7. `make_submission.py`

```
python tools/make_submission.py <task_dir> [--output_dir DIR] [--task_id t1-NNN]
```

Runs step 1 on the folder and refuses on any problem; leaves out the skipped
clutter; writes `DIR/<task_id>.zip` with `--task_id`, otherwise
`DIR/<folder>.zip` (never replacing an existing file); prints the path, size,
sha256 and the task id. A malformed `--task_id` is refused. The zip is
deterministic: sorted members under `<folder>/`, deflate level 9, timestamp 1980-01-01, mode 0644.
Files over 5 MB are refused by step 1 before anything is read into the zip. Exit 0 on
success, 1 on refusal.

## 8. Worked examples

Three complete submissions share one task package (`task/`, byte-identical),
`fetch_data.py` and `data_source.json`, converted from the SIDERIUS quickstart
(synthetic two-class tabular data). Their data files are the four quickstart
shards generated with seed 20260824 (SHA-256 equal to the pins in SIDERIUS
`examples/quickstart/declared/data_manifest.json`); `data_source.json` holds
their real sizes and md5 with the placeholder record id, so step 2a fails until
the organizers publish them. The task package is the SIDERIUS CSV
classification template adapted to that data (file-backed labels), with primary
`accuracy`, secondary `balanced_accuracy`, two Health checks and the PyTorch
model plugin bonus.

| Example | framework / format | init model | requirements.txt | validation accuracy |
| --- | --- | --- | --- | --- |
| `quickstart_classification` | `pytorch` / `pt` (MLP) | yes | none (SIDERIUS venv) | 0.859375 |
| `quickstart_classification_sklearn` | `sklearn` / `skops` (decision tree) | no | `scikit-learn==1.7.2`, `skops==0.13.0` | 0.8125 |
| `quickstart_classification_keras` | `tensorflow` / `keras` (MLP, Keras 3 on the PyTorch backend) | yes | `keras==3.12.0` | 0.875 |

## 9. Unresolved

- Whether Zenodo's files API honours a secret-link `token` query parameter for
  restricted records is unverified (no restricted test record was available);
  the example `fetch_data.py` passes it on. `--data_dir` is the fallback for
  restricted data.
- The format checks for `cbm`, `ubj`, `txt`, `weights.h5`, `savedmodel` and
  `onnx` are exercised only with synthetic files in the tests; no real
  CatBoost, XGBoost, LightGBM, TensorFlow or ONNX model was produced here.
  TensorFlow itself was not installed: the Keras example uses Keras 3 on the
  PyTorch backend.
- The portal's own upload-time checks are not specified here; they are
  expected to repeat step 1 only (no participant code runs at upload).
