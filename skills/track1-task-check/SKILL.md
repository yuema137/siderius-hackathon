---
name: track1-task-check
description: Help a Galileo Hackathon Track 1 author check a task submission before upload. Run the deterministic checker, explain each FAIL with the concrete fix, then give an advisory review of the task's scientific design (evaluation, leakage, task card, Health checks). Use when an author asks whether their Track 1 task package is ready, why the checker fails, or how to improve the task.
---

# Track 1 task check

The kit's [SPEC.md](../../SPEC.md) owns the submission rules and
`tools/check_submission.py` executes them. This skill adds a procedure and an
advisory review; it is not another rule authority. When SPEC.md, the checker
and this skill disagree, the checker's result is what the organizers see:
report the disagreement to the author instead of reconciling it yourself.

Two kinds of output, never mixed:

- **Deterministic result**: the checker's per-step PASS/FAIL/SKIP and its
  verdict. Only the checker produces it. Quote it exactly.
- **Advisory notes**: your judgment of the task design. They never change,
  override or excuse a deterministic result. A task with every step PASS can
  still get advisory notes; a FAIL stays a FAIL however good the design is.

## Before running anything

1. Identify the submission folder (`<task_id>/`) or zip, the SIDERIUS checkout
   (`--siderius`; it must have `.venv/bin/python` from
   `uv sync --group dev --frozen`) and where the data is: already downloaded
   (`--data_dir`) or to be downloaded (`--download`). Ask only for what you
   cannot find; reuse paths the author already gave.
2. Read SPEC.md sections 1 to 6 and the author's `TASK_CARD.md`,
   `data_source.json`, `task/task.yaml` and `baseline/baseline_result.json`.
3. The checker runs the author's `fetch_data.py`, `run_baseline.py` and task
   plugins as subprocesses without a sandbox. That is expected for the author's
   own code; for a submission from someone else, say so and suggest a
   disposable machine or container before running steps 2b to 4b.
4. `--retrain` reruns the full baseline training (GPU and hours if the
   baseline needs them). Ask before adding it unless the author requested it.
   `--download` downloads the whole Zenodo record.

## Run the checker

From the kit root, with the SIDERIUS interpreter:

```bash
$SIDERIUS/.venv/bin/python tools/check_submission.py <task_dir-or-zip> \
  --siderius $SIDERIUS (--data_dir DIR | --download) [--retrain] --report <path>/check-report.json
```

Read the printed lines and the report JSON (`steps[].details` holds problem
lists, command output tails and metric comparisons). Record the exact command,
exit status and verdict. Use the exit status, not your reading of the text, as
the verdict: 0 means no FAIL.

Present the result as a table of step, status and reason, then:

- **For each FAIL**: the cause in one sentence, the file and field or line to
  change, and the concrete fix. Use the mapping below. When several problems
  share a cause, say so once.
- **For each SKIP**: which option or earlier fix enables it. The organizers run
  every step with `--download --retrain`; a SKIP is not evidence of a pass.
- After the author fixes something, rerun the checker; do not declare a step
  fixed from an edit alone.

| Step | Typical FAIL | Fix |
| --- | --- | --- |
| 1 | data file in the package; weights outside `baseline/`; file > 5 MB; missing heading or empty section; card lacks the DOI; schema error naming a field | move data to Zenodo and read it via `fetch_data.py`; rename or move weights to `baseline/init_weights.*`/`final_weights.*`; split or shrink the file; fill the section; copy the DOI from `data_source.json` into `## Data`; correct the named field per SPEC.md section 2 or 4.2 |
| 2a | placeholder id; HTTP 404; resolves to another id; md5/size/license/access mismatch | publish the record and copy its version id (not the concept id) and DOI; copy name, size and md5 from Zenodo's file list; align `license`/`access` with the Zenodo record |
| 2b | md5 differs for a `--data_dir` file; `fetch_data.py` non-zero exit or wrote nothing | re-download or correct `data_source.json`; read `details.output_tail` for the traceback and fix the script; make sure it writes into `--output_dir` |
| 3a | `init` refused (permissions); `add-task`/`check` names a field or file | the checker creates private folders itself, so an `init` refusal is an environment problem to report; otherwise fix the named manifest field or file inside `task/` (every path relative to `task/`, nothing outside it) |
| 3b | `TaskCompositionError` naming a section, file or symbol | fix that `task.yaml` section; see the SIDERIUS [task composition reference](https://github.com/yuema137/SIDERIUS/blob/master/docs/reference/task-composition.md) |
| 4.0 | weights do not load with `weights_only=True`; not a flat state dict; final equals initial | save `model.state_dict()` with `torch.save` or safetensors; save the trained weights as final |
| 4a/4b | non-zero exit; `metrics.json` ids differ from declared ids; value outside tolerance | read `output_tail`; key `metrics.json` by the declared metric ids and compute them with the task's metric plugin; make training seed-deterministic, or set a tolerance that honestly covers the measured spread and state it in the card |

Never edit `baseline_result.json` metrics or tolerance just to make a check
pass. A tolerance change needs a reason the author can state in the card (for
example measured GPU run-to-run spread).

## Advisory review

Label this section **Advisory** and keep it separate from the result. Base each
note on a file and line you read; say what you could not verify. Cover:

1. **Evaluation tied to the scientific goal.** Does the primary metric measure
   what the scientific question asks (for example a detection threshold,
   physical units, a class balance that matters)? Is the direction right? Would
   a trivial predictor (constant, majority class, mean) score well, and does
   the card state that reference score? Are secondary metrics or a formula
   needed to stop a model from optimizing the wrong thing?
2. **Leakage and shortcuts.** Can a sample's identifier, file order, filename,
   metadata column or simulation artifact reveal the label? Are related samples
   (same object, run, patient, day) split across training and validation? Is
   normalization or feature selection fit on validation data? Can a model read
   held-out test data from the data folder? Does `fetch_data.py` leak the
   label into features?
3. **Health checks.** Which invalid outputs would score deceptively well
   (constant output, out-of-range values, NaN, collapsed classes, physically
   impossible predictions)? Would the declared Health roster catch each one,
   and with the right disposition (blocking vs recording)? A `task_health:
   none: true` is allowed; note what it leaves uncaught.
4. **Task card completeness.** Each section answers its template question; the
   Data section states DOI, size, license, access and processing; the Baseline
   section matches `baseline_result.json`; novelty cites prior work; difficulty
   states the headroom.
5. **Reproducibility and cost.** Seed use in `run_baseline.py`, deterministic
   settings, the GPU and wall time stated, a tolerance that matches the
   measured spread.

End with the deterministic verdict again, in one line, so the advisory notes
are not read as the result.
