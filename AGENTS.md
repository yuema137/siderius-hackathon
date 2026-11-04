# Agent guide: build and check a Track 1 submission

This file is for coding agents. Start with the participant's scientific question,
then use the technical instructions below to prepare the submission.

- Read [SPEC.md](SPEC.md) for the exact submission rules and
  [the check skill](skills/track1-task-check/SKILL.md) for the checking procedure.
- Keep the deterministic checker's verdict separate from LLM advisory notes.
  Do not invent scores, checksums, data access, or successful checks.
- Ask the participant to confirm the scientific question, evaluation, data-sharing
  rights, and public-release choice. They handle registration, Zenodo publication,
  and the final portal upload; prepare the files and explain the next action.
- Ask before large downloads or full baseline training/retraining. Report resource
  requirements before running them. The deterministic checker does not call an
  LLM; using a coding agent may require a separate subscription or API access.
- Organizer reproducibility review checks whether submitted results can be
  reproduced. It does not judge scientific merit, research quality, or awards.
  Your advisory design feedback is not an organizer verdict.
- Use the requirements and command flags supported by this checkout. Do not copy
  newer artifact names or CLI flags from an unreleased branch.

The current technical workflow follows. Keep its commands and requirements in
sync with this kit's SPEC and tools. The participant-facing [README](README.md)
intentionally omits these details.

# Technical workflow

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
- **SIDERIUS v1.1.0 or later** from <https://github.com/yuema137/SIDERIUS>
  (see step 0). The checker refuses an older checkout.
- **This kit**, cloned anywhere.
- **A Zenodo account** to publish your data.
- **A GPU only if your baseline needs one.** The deterministic checker needs no API keys and never calls a language model.
  Your coding agent has its own subscription or API requirements.
- **Network access and disk space** if your baseline needs extra packages
  (`baseline/requirements.txt`): the checker then builds a separate Python
  environment, which can download several GB the first time (see step 3).

## What is in this kit

| Path | What it is |
| --- | --- |
| [SPEC.md](SPEC.md) | The exact rules: layout, file formats, size limits, every check. Your coding agent reads this. |
| [templates/TASK_CARD.md](templates/TASK_CARD.md) | The task card to fill in, with guidance under each heading. |
| [examples/](examples/) | Start with the local PyTorch walkthrough; optional scikit-learn and Keras variants are under examples/advanced/. These are format references, not submission-ready packages; the Zenodo check intentionally fails. |
| `tools/check_submission.py` | The self-check. Run it before every upload. |
| `tools/make_submission.py` | Builds the upload zip and prints its checksum. |
| [skills/track1-task-check/SKILL.md](skills/track1-task-check/SKILL.md) | Instructions for a coding agent (Claude Code, Codex) helping you check and improve your task. |

## Steps

### 0. Install SIDERIUS

```bash
git clone https://github.com/yuema137/SIDERIUS.git
cd SIDERIUS
git checkout v1.1.0
uv sync --group dev --frozen
```

Use the published v1.1.0 release for this event. The checker requires the
`v1.1.0` tag to be an ancestor of the checkout (later revisions are accepted,
but these instructions target the release). Its
step 3.0 fails when your checkout's history does not contain it (run
`git fetch --tags` if the checker says the revision is unknown). It also
records the SIDERIUS version and git revision it ran with in its report.

Success: the sync ends without an error and `.venv/bin/python` exists. Every
later command uses this interpreter; the examples write it as
`$SIDERIUS/.venv/bin/python`, so set `SIDERIUS=/absolute/path/to/SIDERIUS`
first.

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

First, sign in to the portal at [the submission portal](https://submit.siderius.scitra.org) and create your task with a
title. The portal registers the title and assigns the task's id, for example
`t1-007`; you upload every version of the package to that task.

Then create a folder for the package with any working name (letters, digits,
`-`, `_`, starting with a letter or digit), for example `galaxy_morphology/`. The folder name is only for you;
the task id is the one the portal assigned. The fastest start is to copy the
worked example closest to your model and change it. The folder holds:

```
galaxy_morphology/
  TASK_CARD.md
  data_source.json
  fetch_data.py
  task/                 task.yaml + declared/ + plugins/
  baseline/run_baseline.py
  baseline/final_model.<format>      the trained baseline model
  baseline/init_model.<format>       neural networks only: the model before training
  baseline/requirements.txt          optional: extra packages, pinned with ==
  baseline/baseline_result.json
```

1. **Publish the data on Zenodo.** Upload the files, choose a license and
   publish. Public is preferred. If the data must stay restricted, publish it
   as restricted and do one of these:
   - create a *Can view* secret link (record page, **Share**, **Links**) that
     expires no earlier than 2026-12-31, and put it in `share_link`; or
   - share the record with the organizers' Zenodo account and say so in
     `access_note`.

   See [SPEC.md section 2](SPEC.md#2-data_sourcejson).
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
   SIDERIUS [CSV classification template](https://github.com/yuema137/SIDERIUS/blob/v1.1.0/examples/task_templates/csv_classification/README.md);
   for anything else, follow the SIDERIUS
   [task definition guide](https://github.com/yuema137/SIDERIUS/blob/v1.1.0/docs/guides/define-a-task.md).
   A coding agent opened in your SIDERIUS checkout can write it with you.
5. **Write and run the baseline.** The baseline shows the task is solvable
   and sets the score to beat. Classical algorithms and rule-based methods are also allowed: use `framework: "classic"`,
   `model_format: "py"`, and `baseline/final_model.py` with no model weights.
   Keep the same train/evaluate interface; for a fixed method, train copies the
   algorithm into its output folder and evaluates it. See SPEC.md, Script baselines.
   Use any of these frameworks; the file name is
   always `final_model.<format>`:

   | Framework | Recommended format | Also allowed |
   | --- | --- | --- |
   | PyTorch | `.pt` (a `state_dict` saved with `torch.save`) | `.onnx` |
   | TensorFlow / Keras | `.keras` | `.weights.h5`, SavedModel folder `final_model/`, `.onnx` |
   | XGBoost | `.json`, `.ubj` | |
   | LightGBM | `.txt` | |
   | CatBoost | `.cbm` | |
   | scikit-learn | `.skops` | `.joblib`, `.pkl` (discouraged: loading them runs code) |

   `baseline/run_baseline.py` has two commands, `train` and `evaluate`, and
   computes scores with your task's own metric code
   ([SPEC.md section 4](SPEC.md#4-baseline)). Train with a fixed seed and save
   the final model. For a neural network (PyTorch, TensorFlow/Keras), also save
   the model before training as `init_model.<format>`, so training can be
   repeated from the same start; other models are repeated from the seed and
   the hyperparameters alone. Record the framework, format, hyperparameters,
   seed, scores, hardware, run time and a reproduction tolerance in
   `baseline/baseline_result.json`.

   If your baseline needs packages that SIDERIUS does not install (for example
   scikit-learn or Keras), list them in `baseline/requirements.txt`, one per
   line with an exact version (`scikit-learn==1.7.2`). Each package, and every
   package it depends on, must be published as a wheel for Linux and Python
   3.12; packages that only exist as source code are refused. These packages
   are only for `run_baseline.py`: the task package in `task/` runs inside
   SIDERIUS's own environment, so its plugins may import only packages that
   SIDERIUS installs.

   Optionally (a bonus), also provide your model as a SIDERIUS model plugin in
   `task/plugins/`, as the examples do. The bonus must be PyTorch, because the
   SIDERIUS trainer is PyTorch; the required baseline above can use any
   framework in the table.

Data files never go in the folder: data arrays and tables (`.npy`, `.npz`,
`.h5`, `.parquet`, `.pkl`, `.feather`) are refused, and `baseline/` holds only
the model files plus small `.py`, `.json`, `.yaml`, `.txt` and `.md` files. Each
file must be at most 5 MiB and the zip at most 20 MiB.

### 3. Check it

From the kit folder:

```bash
$SIDERIUS/.venv/bin/python tools/check_submission.py /absolute/path/galaxy_morphology \
  --task_id t1-007 --siderius $SIDERIUS --download --retrain --report check-report.json
```

Replace `t1-007` with your task's id from the portal; the checker registers
the task in SIDERIUS under that id. Without `--task_id` it uses a local working
id made from the folder name and says so. Use `--data_dir DIR` instead of `--download` if the Zenodo files are already
on disk in `DIR`. `--retrain` retrains your baseline, so it takes as long as
your training; leave it out for a quick check.

If you have `baseline/requirements.txt`, the checker builds a separate Python
environment for your baseline with uv: SIDERIUS (with its own locked package
versions) plus your pinned packages. This needs network access. The first time,
uv downloads SIDERIUS's dependencies, mostly PyTorch and its CUDA libraries
(several GB; the environment takes about 7.6 GB of disk); later runs reuse uv's
cache and take seconds. Without the file, your baseline runs in the SIDERIUS
environment directly.

The checker prints one line per step and a verdict, for example:

```text
task id: t1-007 (portal-assigned task id); folder: galaxy_morphology
[PASS]   1 package: layout, sizes, file types, card and JSON schemas are valid
[PASS]  2a zenodo_record: record 1234567 found; declared files match Zenodo's names, sizes and md5
[PASS]  2b data_files: fetch_data.py downloaded and processed the Zenodo files: 2 file(s) in the data folder
[PASS] 3.0 siderius_version: SIDERIUS 1.1.0 at 5ccb436f88e6 includes v1.1.0
[PASS]  3a siderius_admission: init, add-task --manifest --task-id t1-007 --no-gpu and check accepted the task
[PASS]  3b siderius_composition: composed: primary metric accuracy (higher), 1 secondary, Health declared
[PASS] 4.0 baseline_model: pytorch/pt: initial and final model pass the format checks
[PASS] 4.1 baseline_environment: no baseline/requirements.txt: the baseline runs in the SIDERIUS venv
[PASS]  4a baseline_evaluate: evaluating the final model reproduces every reported metric within its tolerance
[PASS]  4b baseline_retrain: retraining with seed 20261104 from the initial model reproduces every reported metric within its tolerance
verdict: PASS  (zip sha256 <64 hex characters>, <size> bytes)
```

Success: `verdict: PASS` and exit status 0. `SKIP` means a step did not run;
its reason says which option enables it. If a required step (3.0, 3a, 3b,
4.0, 4.1 or 4a) was skipped and nothing failed, the verdict is `INCOMPLETE` (exit status
2): run again with `--siderius` and `--data_dir` or `--download`, and with
network access if you have `baseline/requirements.txt`. Only 4b
(retraining) may be skipped for a `PASS`, but the organizers run it too.
A `FAIL` verdict (exit status 1) means a step failed; its line names the file or value to fix;
`check-report.json` holds the full output of each step. The steps are
described in [SPEC.md section 6](SPEC.md#6-checker).

To have a coding agent run the check, explain failures and review your task
design, point it at [skills/track1-task-check/SKILL.md](skills/track1-task-check/SKILL.md).

### 4. Build the zip

```bash
$SIDERIUS/.venv/bin/python tools/make_submission.py /absolute/path/galaxy_morphology \
  --task_id t1-007 --output_dir /absolute/path/uploads
```

It refuses a folder that fails step 1, leaves out caches such as
`__pycache__`, and prints:

```text
wrote /absolute/path/uploads/t1-007.zip
task id: t1-007
size <bytes> bytes
sha256 <64 hex characters>
```

Building the same folder again gives the same sha256. To rebuild, move the
old zip away first; it is never overwritten.

### 5. Upload

On the portal at [the submission portal](https://submit.siderius.scitra.org), open the task you created in step 2 and upload
the zip to it. Each upload is a new version of that task. Keep the sha256 the
tool printed; the portal's receipt shows the sha256 it received, and the two
should match. You can upload new versions until the deadline; the latest
counts.

## Local walkthrough and advanced variants

[examples/](examples/) holds the SIDERIUS quickstart task (classify 256
synthetic rows with four features into two classes) as three submissions that
differ only in `baseline/`:

| Example | Baseline | Files in `baseline/` |
| --- | --- | --- |
| [quickstart_classification](examples/quickstart_classification/) | PyTorch MLP | `init_model.pt`, `final_model.pt`; no `requirements.txt` |
| [quickstart_classification_sklearn](examples/advanced/quickstart_classification_sklearn/) | scikit-learn decision tree | `final_model.skops` only; `requirements.txt` pins scikit-learn and skops |
| [quickstart_classification_keras](examples/advanced/quickstart_classification_keras/) | Keras MLP (Keras 3 on its PyTorch backend) | `init_model.keras`, `final_model.keras`; `requirements.txt` pins keras |

Read [the PyTorch example's TASK_CARD.md](examples/quickstart_classification/TASK_CARD.md) first.

Its four data files are generated, not downloaded. This is intentionally a local-only walkthrough. Its `data_source.json` holds the placeholder record id
`ZENODO_RECORD_ID`, so step 2a fails with "placeholder". This expected failure does not count as a full checker PASS; inspect the other steps separately. To check it yourself, generate the files from your SIDERIUS checkout:

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

Replace `examples/quickstart_classification` with
`examples/advanced/quickstart_classification_sklearn` or
`examples/advanced/quickstart_classification_keras` to check the others; those two build the
separate baseline environment described in step 3.

The generator prints the SHA-256 of `shard_0000.csv` to `shard_0003.csv`.
There is no planned Zenodo publication for these walkthrough files. For a real
challenge, replace the placeholder in `data_source.json` and `TASK_CARD.md`
with your own published version record, then run the complete checker.
Never bypass step 2a or report the walkthrough as submission-ready.

## Where each choice lives

| Choice | File |
| --- | --- |
| Question, design rationale, risks, contact | `TASK_CARD.md` |
| Data location, access, checksums | `data_source.json` |
| How raw files become the data folder | `fetch_data.py` |
| Data reading, splits, model input/output, metric, Health checks, parameter limits | `task/task.yaml` and the files it names |
| Baseline model and training recipe | `baseline/run_baseline.py` (and the optional PyTorch model plugin in `task/plugins/`) |
| Extra packages the baseline needs | `baseline/requirements.txt` |
| Framework, model format, hyperparameters, seed, reported scores, tolerance, hardware | `baseline/baseline_result.json` |

## Running the kit's tests

```bash
$SIDERIUS/.venv/bin/python -m pytest tests
```

The tests need no network and no SIDERIUS commands.

## License

MIT; see [LICENSE](LICENSE). The worked examples' task package is adapted from
MIT-licensed SIDERIUS code.
