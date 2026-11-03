# Galileo Hackathon Track 1: submit a scientific ML task

In Track 1 you design a scientific machine-learning task: a question, the data
that answers it, how answers are scored and a baseline that shows the task is
solvable. In Track 2, agents built on [SIDERIUS](https://github.com/yuema137/SIDERIUS)
try to beat your baseline. This kit tells you what to put in your submission,
checks it on your own machine and builds the zip you upload.

You submit **one zip of a task package**: the files that describe your task to
SIDERIUS, a script that fetches your data, a baseline and a short task card.
The data itself is not in the zip; it lives on [Zenodo](https://zenodo.org), a
free research data archive that gives every dataset a permanent DOI.

## Before you start

- **Linux** with **[bubblewrap](https://github.com/containers/bubblewrap)**
  (`bwrap`, the sandbox SIDERIUS runs tasks in). Check it with
  `bwrap --ro-bind / / true && echo ok`.
- **Python 3.12+** and **[uv](https://docs.astral.sh/uv/)**.
- **A SIDERIUS v1.1.0 checkout**, installed once:

  ```bash
  git clone https://github.com/yuema137/SIDERIUS.git
  cd SIDERIUS
  uv sync --group dev --frozen
  ```

  Success: `.venv/bin/python` exists. Every command below uses this
  interpreter; the examples write it as `$SIDERIUS/.venv/bin/python`, so set
  `SIDERIUS=/absolute/path/to/SIDERIUS` first.
- **This kit**, cloned anywhere.
- **A Zenodo account** to publish your data.
- **A GPU only if your baseline needs one.** No API keys are needed: the
  checker never calls a language model.

## What is in this kit

| Path | What it is |
| --- | --- |
| [SPEC.md](SPEC.md) | The exact rules: layout, file formats, size limits, every check. Your coding agent reads this. |
| [templates/TASK_CARD.md](templates/TASK_CARD.md) | The task card to fill in, with guidance under each heading. |
| [examples/quickstart_classification/](examples/quickstart_classification/) | A complete, passing submission to copy from. |
| `tools/check_submission.py` | The self-check. Run it before every upload. |
| `tools/make_submission.py` | Builds the upload zip and prints its checksum. |
| [skills/track1-task-check/SKILL.md](skills/track1-task-check/SKILL.md) | Instructions for a coding agent (Claude Code, Codex) helping you check and improve your task. |

## Steps

### 1. Design the task

Decide, before writing code:

- the scientific question and what one sample is;
- the model input and output (shapes and meaning);
- the split: which samples are for training, which for validation (the score
  Track 2 agents see) and, if you keep one, a held-out test. Samples that are
  related (same patient, run, detector, day) belong in the same split, or the
  score is too optimistic;
- the **primary metric** (required) and its direction (higher or lower is
  better). Secondary metrics and **Health checks** are optional and
  encouraged. A Health check is a rule that marks a model's output invalid
  whatever its score, for example "predicts only one class".

Write these into a copy of [templates/TASK_CARD.md](templates/TASK_CARD.md) as
you go. The card's sections are listed in [SPEC.md section 5](SPEC.md#5-task_cardmd).

### 2. Build the package

Create a folder named after your task id (letters, digits, `-`, `_`), for
example `galaxy_morphology/`, with this content. The fastest start is to copy
the worked example and change it.

```
galaxy_morphology/
  TASK_CARD.md
  data_source.json
  fetch_data.py
  task/                 task.yaml + declared/ + plugins/
  baseline/run_baseline.py
  baseline/init_weights.safetensors   (or .pt)
  baseline/final_weights.safetensors  (or .pt)
  baseline/baseline_result.json
```

1. **Publish the data on Zenodo.** Upload the files, choose a license and
   publish. Public is preferred. If the data must stay restricted, publish it
   as restricted and either create a *Can view* share link (record page,
   **Share**, **Links**) or give the organizers' account access; see
   [SPEC.md section 2](SPEC.md#2-data_sourcejson).
2. **Write `data_source.json`**: the record id and DOI from the Zenodo page,
   access, license, and each file's name, size in bytes and md5 checksum.
   Zenodo shows the md5 on the record page; locally, `md5sum FILE` and
   `stat -c %s FILE` give the same numbers.
3. **Write `fetch_data.py`**: it downloads the files (or takes them from
   `--source_dir`), checks the md5, and writes the files your task reads into
   `--output_dir`. The example's [fetch_data.py](examples/quickstart_classification/fetch_data.py)
   is a ready pattern.
4. **Write the SIDERIUS task package in `task/`.** It tells SIDERIUS how to read
   your data, the model input and output, the metric and the Health checks.
   For CSV classification data, start from the example's `task/` or the
   SIDERIUS [CSV classification template](https://github.com/yuema137/SIDERIUS/blob/master/examples/task_templates/csv_classification/README.md);
   for anything else, follow the SIDERIUS
   [task definition guide](https://github.com/yuema137/SIDERIUS/blob/master/docs/guides/define-a-task.md).
   A coding agent opened in your SIDERIUS checkout can write it with you.
5. **Write and run the baseline.** `baseline/run_baseline.py` has two
   commands, `train` and `evaluate`, and computes scores with your task's own
   metric code ([SPEC.md section 4](SPEC.md#4-baseline)). Save the initial
   weights, train with a fixed seed, save the final weights, then record the
   scores, seed, hardware, run time and a reproduction tolerance in
   `baseline/baseline_result.json`. Optionally (a bonus), also provide the
   model as a SIDERIUS model plugin in `task/plugins/`, as the example does.

Data files never go in the folder: data arrays and tables (`.npy`, `.npz`,
`.h5`, `.parquet`, `.pkl`, `.feather`) are refused outside `baseline/`. Each
file must be at most 5 MB and the zip at most 20 MB.

### 3. Check it

From the kit folder:

```bash
$SIDERIUS/.venv/bin/python tools/check_submission.py /absolute/path/galaxy_morphology \
  --siderius $SIDERIUS --download --retrain --report check-report.json
```

Use `--data_dir DIR` instead of `--download` if the Zenodo files are already
on disk in `DIR`. `--retrain` retrains your baseline from the initial weights,
so it takes as long as your training; leave it out for a quick check.

The checker prints one line per step and a verdict, for example:

```text
[PASS]   1 package: layout, sizes, file types, card and JSON schemas are valid
[PASS]  2a zenodo_record: record 1234567 found; declared files match Zenodo's names, sizes and md5
[PASS]  2b data_files: fetch_data.py downloaded and processed the Zenodo files: 2 file(s) in the data folder
[PASS]  3a siderius_admission: init, add-task --manifest --no-gpu and check accepted task galaxy_morphology
[PASS]  3b siderius_composition: composed: primary metric accuracy (higher), 1 secondary, Health declared
[PASS] 4.0 baseline_weights: both weight files load as state dicts without executing code
[PASS]  4a baseline_evaluate: evaluating the final weights reproduces every reported metric within its tolerance
[PASS]  4b baseline_retrain: retraining with seed 20261104 reproduces every reported metric within its tolerance
verdict: PASS  (zip sha256 <64 hex characters>, <size> bytes)
```

Success: `verdict: PASS` and exit status 0. `SKIP` means a step did not run
(its reason says which option enables it); it does not fail the check, but
the organizers run every step. A `FAIL` line names the file or value to fix;
`check-report.json` holds the full output of each step. The steps are
described in [SPEC.md section 6](SPEC.md#6-checker).

To have a coding agent run the check, explain failures and review your task
design, point it at [skills/track1-task-check/SKILL.md](skills/track1-task-check/SKILL.md).

### 4. Build the zip

```bash
$SIDERIUS/.venv/bin/python tools/make_submission.py /absolute/path/galaxy_morphology \
  --output_dir /absolute/path/uploads
```

It refuses a folder that fails step 1, leaves out caches such as
`__pycache__`, and prints:

```text
wrote /absolute/path/uploads/galaxy_morphology.zip
size <bytes> bytes
sha256 <64 hex characters>
```

Building the same folder again gives the same sha256. To rebuild, move the
old zip away first; it is never overwritten.

### 5. Upload

Upload the zip at `PORTAL_URL`. Keep the sha256 the tool printed; the portal's
receipt shows the sha256 it received, and the two should match. You can upload
new versions until the deadline; the latest counts.

## The worked example

[examples/quickstart_classification/](examples/quickstart_classification/) is
the SIDERIUS quickstart task (classify 256 synthetic rows with four features
into two classes) turned into a submission. Read its
[TASK_CARD.md](examples/quickstart_classification/TASK_CARD.md) first.

Its four data files are generated, not downloaded. Until the organizers
publish them on Zenodo, its `data_source.json` holds the placeholder record id
`ZENODO_RECORD_ID`, so step 2a fails with "placeholder" and every other step
passes. To check it yourself, generate the files from your SIDERIUS checkout:

```bash
mkdir -p /absolute/path/example-data
cd $SIDERIUS/examples/quickstart
../../.venv/bin/python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('qs', 'plugins/_quickstart_task.py')
qs = importlib.util.module_from_spec(spec); sys.modules['qs'] = qs; spec.loader.exec_module(qs)
print(qs.write_data_dir(sys.argv[1]))
" /absolute/path/example-data
cd -
$SIDERIUS/.venv/bin/python tools/check_submission.py examples/quickstart_classification \
  --siderius $SIDERIUS --data_dir /absolute/path/example-data --retrain
```

The generator prints the SHA-256 of `shard_0000.csv` to `shard_0003.csv`.
After the upload, the organizers replace `ZENODO_RECORD_ID` in
`data_source.json` and `TASK_CARD.md` with the record id.

## Where each choice lives

| Choice | File |
| --- | --- |
| Question, design rationale, risks, contact | `TASK_CARD.md` |
| Data location, access, checksums | `data_source.json` |
| How raw files become the data folder | `fetch_data.py` |
| Data reading, splits, model input/output, metric, Health checks, parameter limits | `task/task.yaml` and the files it names |
| Baseline model and training recipe | `baseline/run_baseline.py` (and the optional model plugin in `task/plugins/`) |
| Reported scores, seed, tolerance, hardware | `baseline/baseline_result.json` |

## Running the kit's tests

```bash
$SIDERIUS/.venv/bin/python -m pytest tests
```

The tests need no network and no SIDERIUS commands.
