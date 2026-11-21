# Track 2 technical reference

[Back to onboarding](../README.md). Follow the released package's task-specific
instructions alongside this reference. Replace every `<placeholder>` before
running a command. The root README holds the schedule, submission quota,
train-only rules, awards and report logistics.

## Environment and data

On the organizer GPU machine:

```sh
source /opt/siderius/.venv/bin/activate
ls /data/galileo/
cat /data/galileo/README.md
mkdir -p ~/work/<slug>
cp -r /data/galileo/<slug>/package ~/work/<slug>/package
ls ~/work/<slug>/package/
cd ~/work/<slug>/package/<package-folder>
python fetch_data.py --source_dir /data/galileo/<slug>/data --output_dir ~/work/<slug>/data
```

The package folder contains `TASK_CARD.md`, the task definition and `baseline/`.
Read these before training. The shared folder is read-only; do not edit it.
Data/task access follows the announced release time, not the existence of a path
in this guide. A missing mount is an access issue to raise with the organizers.

On your own machine, install Python 3.12 and uv, then:

```sh
git clone https://github.com/yuema137/SIDERIUS.git
cd SIDERIUS
git checkout v1.1.0
uv sync --group dev --frozen
uv pip install --python .venv/bin/python scikit-learn==1.9.1 skops==0.13.0
source .venv/bin/activate
```

Use Linux and a compatible NVIDIA GPU for GPU training. CPU algorithm tasks
have different needs; check your package. After the noon release, download the
selected package from [Releases](https://github.com/yuema137/siderius-hackathon/releases):

```sh
mkdir -p ~/work/<slug>
unzip <downloaded-package>.zip -d ~/work/<slug>/package
ls ~/work/<slug>/package/
cd ~/work/<slug>/package/<package-folder>
python fetch_data.py --output_dir ~/work/<slug>/data
```

Without `--source_dir`, the fetch script downloads and verifies the source data.
Keep enough free disk space (roughly twice the download size). Run the baseline
using this release's documented commands and keep the logs before editing it.
There is no single training command that works for all eight tasks.

## Choose a mode and model profile

Use **orchestration mode** for the competition: your coding agent selects the
next capability calls at runtime. **Workflow mode** runs a predefined sequence
and is the economical option when that sequence is sufficient. Keep the same
scientific checks in either mode; flexibility does not establish validity.

The organizer installation is `/opt/siderius`. In your own installation, use
your SIDERIUS checkout path instead. Copy a profile to your writable project
and point your run's configuration to that copy:

```sh
# Initial smoke test: inexpensive setup validation.
cp /opt/siderius/configs/llm/openai_smoke_luna.json ~/work/<slug>/llm-smoke.json
# Recommended profile for substantive experiments.
cp /opt/siderius/configs/llm/openai-moderate-v1.json ~/work/<slug>/llm-moderate.json
```

Copying a file does not switch an existing run: configure the caller to use the
chosen file, then check the recorded model configuration. Keep a copy alongside
your experiment log. You can change the models, but API prices vary by model
and provider. Include both capability calls and your external coding agent in
your budget; these JSON profiles do not select the external agent's model.

For orchestration, follow `src/tools/orchestration_setup/README.md` in the
SIDERIUS checkout. The setup assembles project instructions using a sandbox
profile and a run declaration; the coding agent then writes the task-specific
caller. A useful starting instruction for your agent is:

> Read the released task's TASK_CARD.md and baseline instructions, then follow
> SIDERIUS's orchestration_setup guide to prepare this writable project. Use
> llm-smoke.json for a short smoke test and llm-moderate.json for substantive
> experiments. Respect the task's train/validation split and scientific checks.
> Set explicit time and API budgets, record each hypothesis and result, and
> keep the complete training source needed to reproduce the submitted model.

The setup tool's example sandbox is an isolation demonstration with a short
budget, not a ready-made competition configuration. Set task-appropriate data
access, GPU access and runtime limits. Follow the assembled project's commands;
there is no universal caller command for all eight tasks.

## Package and check

Stage only the files needed to reproduce and score your solution:

```text
submission.json     at ZIP root
model.py            self-contained inference plugin
initial.pt          model state immediately after initialization, before training
weights.pt          final trained state_dict
cfg.json            optional constructor arguments
train/              complete original training source, nonempty
  train.py
```

Illustrative manifest, for a task that accepts a named class and configuration:

```json
{
  "kind": "model",
  "seed": 0,
  "initial_weights": "initial.pt",
  "plugin": "model.py",
  "symbol": "MyNet",
  "weights": "weights.pt",
  "plugin_config": "cfg.json",
  "train_command": "python train/train.py --data_dir $DATA_DIR --seed $SEED --initial-out initial.pt --out weights.pt",
  "train_dir": "train/"
}
```

Adapt `symbol` and `plugin_config` as required by the task table below. The
training flags shown are examples: implement the same behavior in your script. Paths are relative to the ZIP root.
PyTorch weights are loaded as a state dict with `weights_only=True`, not a
pickled model object. For algorithms without learned parameters, use
`"kind": "algorithm"`, `"seed": 0`, and `"initial_artifact": "initial_model.py"`;
include the initial implementation/configuration and the final `plugin`, and
omit `weights` and `initial_weights`. Do not create fake weight files. This
algorithm option applies only where the task contract supports it. The inference plugin
must contain its helpers: arbitrary training files are not available to it.

`train_command` runs with bash in writable `/work`. Read training data through
`$DATA_DIR` (read-only train files); `$PACKAGE_DIR` is the read-only task package,
`$OUTPUT_DIR` is `/work`, and `$SEED` is your recorded manifest seed. Do not rely on absolute paths from
your machine, network access, validation files, or a pre-trained final artifact
being present during reproduction. Include your complete training code and
state legitimate initialization requirements explicitly.

### Capture the initial model before training

All five items are required: full training code, full inference code, recorded
random seed, initial model and final trained model. Record the seed you actually used in the manifest (`0` in the example above),
an integer from 0 to 4,294,967,295. The reproduction command receives it as `$SEED`. Your training code must
apply it to Python, NumPy, PyTorch, data shuffling and any other random generator
it uses. Record deterministic settings and remaining nondeterminism.

Save `initial.pt` **immediately after constructing the model and before any
optimizer step or fitting**. Save `weights.pt` after training. Both must be in
the submitted ZIP, including for geoneutrinos (whose score still uses fresh
retraining). The model architecture and configuration must load both states.

The uploaded initial checkpoint is audit evidence, **not a trusted starting
point for retraining**. Reproduction initializes the model again from the
recorded seed; do not depend on loading your uploaded initial or final weights
inside `train_command`. Include initialization code and document any requested
pretraining separately; an uploaded checkpoint does not authorize pretraining.
File-presence checks cannot prove that a checkpoint is untrained or that source
is complete: reproducibility and scientific review remain necessary.

The scoring/retraining environment is SIDERIUS v1.1.0 with Python 3.12,
PyTorch 2.10, NumPy, SciPy, pandas, h5py, scikit-learn 1.9.1 and skops 0.13.0,
among the installed packages. No new dependencies are installed from your
`requirements.txt`; it is informational only. The sandbox has no network.

Before using an upload attempt:

1. Validate the JSON with `python -m json.tool submission.json`.
2. Check that every named path exists and the plugin obeys the exact input/output
   contract. Test inference and the applicable validity checks locally.
3. Execute the exact `train_command` in a fresh work directory using only the
   train split, with the environment variables above set to local equivalents.
   Confirm it writes the declared initial and final states, or the final plugin for an algorithm. Test inference separately. A partial run checks startup only;
   it does not prove reproduction. Do not point this check at a directory that
   exposes validation labels to training.
4. Inspect the staged files for secrets, private data, logs containing keys,
   caches and unnecessary artifacts. Do not include symlinks.
5. ZIP the contents, not their enclosing folder:

```sh
cd ~/work/<slug>/submission
zip -r ../submission.zip . -x '*__pycache__*' '*.DS_Store'
unzip -l ../submission.zip
```

Use a new archive filename or remove an old local archive first: `zip` updates
existing archives and could retain stale files. Confirm the listing includes
`submission.json`, not `submission/submission.json`. Limits: **100 MiB zipped,
1 GiB unpacked, 2,000 files**. From your laptop, copy a remote archive with:

```sh
scp team@<IP>:~/work/<slug>/submission.zip ./<slug>-submission.zip
```

Upload through the [portal](https://submit.siderius.scitra.org/track2). Save the
submission ID, exact ZIP and checksum. Ten accepted uploads are shared across
all tasks; later evaluation failure does not refund an accepted attempt.

## Task contracts

These summarize the website's published participant guide; the released task
card and baseline explain feature order and scientific interpretation. Scoring
limits are not the three-minute upload cooldown. A metric is usable only when
the submission passes the applicable validity checks and review.

### `exoplanet-orbits`

- **Metric:** `success_rate`; higher is better.
- **Interface:** Either an algorithm file with predict(X), X of shape [N, 4, 60], returning [N, 3] = [log10 P, log10 K, e] per star; or a PyTorch model [B, 4, 60] → [B, 3].
- **Manifest:** Algorithm: omit weights; the plugin defines predict (runs on CPU; if it accepts n_workers it gets 14). PyTorch: plugin + weights, symbol optional.
- **Limits:** scoring 1,800 s; retraining 3,600 s.
- **Download:** approximately 47 MB.

### `pitching-wing-wake`

- **Metric:** `forecast_nmse`; lower is better.
- **Interface:** PyTorch model [B, 14, 48, 96] → [B, 96822] float32: 20 future frames, mean field, 50-bin log spectrum, dominant frequency and 3 regime logits, in that order.
- **Manifest:** symbol required; the class is built with no arguments, so plugin_config is not allowed.
- **Limits:** scoring 1,200 s; median single-sample forward ≤ 100 ms; retraining 3,600 s.
- **Download:** approximately 3.4 GB.

### `cluster-magic-numbers`

- **Metric:** `delta2e_pearson`; higher is better.
- **Interface:** PyTorch model [B, 178] → [B, 2] (energy per atom in eV, gap in eV).
- **Manifest:** No symbol: the plugin defines PLUGIN_MODEL_CLASS and PLUGIN_CONFIG_CLASS; plugin_config optional.
- **Limits:** scoring 900 s; retraining 3,600 s.
- **Download:** approximately 42 MB.

### `cell-signalling`

- **Metric:** `lr_auprc`; higher is better.
- **Interface:** Either an algorithm file with predict(x), x of shape [5, 1024], returning one float (or predict_batch(X) for [N, 5, 1024]); or a PyTorch model [B, 5, 1024] → [B, 1].
- **Manifest:** Algorithm: omit weights. PyTorch: plugin + weights, symbol optional.
- **Limits:** scoring 1,200 s; retraining 3,600 s.
- **Download:** approximately 12 MB.

### `catalyst-binding`

- **Metric:** `mae_ood_mean_ev`; lower is better.
- **Interface:** PyTorch model [B, 228, 6] → [B, 1], the adsorption energy in eV.
- **Manifest:** No symbol: the plugin defines PLUGIN_MODEL_CLASS and PLUGIN_CONFIG_CLASS; plugin_config optional.
- **Limits:** scoring 900 s; retraining 3,600 s.
- **Download:** approximately 365 MB.

### `aneurysm-wall-stress`

- **Metric:** `wall_tawss_rel_l2`; lower is better.
- **Interface:** PyTorch model [1, 16384, 13] → [1, 16384, 2] (TAWSS in Pa, OSI), one case per call.
- **Manifest:** symbol required; built as Class(**plugin_config); plugin_config optional.
- **Limits:** scoring 900 s; ≤ 1,000 ms per case; retraining 3,600 s.
- **Download:** approximately 214 MB.

### `geoneutrinos`

- **Metric:** `obd_combined`; higher is better.
- **Interface:** predict(waveform_files, geometry_file, model_path, procs) → {event id: [score, px, py, pz, cx, cy, cz]}.
- **Manifest:** Always retrained: weights names the file your train_command writes. Include the initial and final uploaded states for audit; scoring uses the freshly retrained output. A PyTorch network inside predict is fine.
- **Limits:** scoring 1,800 s; retraining 2,400 s.
- **Download:** approximately 20.5 GB.

### `solver-selection`

- **Metric:** `gap_closed_vs_sbs`; higher is better.
- **Interface:** PyTorch model [B, 30] → [B, 11] scores; the highest score is the chosen configuration. The 30 features come in train.csv column order. Runs on CPU, one thread.
- **Manifest:** No symbol: the plugin defines PLUGIN_MODEL_CLASS and PLUGIN_CONFIG_CLASS; plugin_config optional.
- **Limits:** scoring 600 s; retraining 1,800 s.
- **Download:** approximately 0.2 MB.

Geoneutrinos always retrains before scoring and can take substantially longer.
Its public-validation scoring is not permission to fit validation labels or a
claim of hidden-set generalization. For all tasks, keep training train-only.

For help, share the task slug and redacted error with the organizers via the
[Discord](https://discord.gg/FXfcGTUb). Never share private keys, access codes,
API keys or your machine address in public.
