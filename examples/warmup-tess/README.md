# TESS warm-up: a star, a baseline, and one next idea

**CPU laptop friendly. No GPU is needed for this warm-up.** First look at real
TESS light curves and measure a deliberately simple baseline. Then optionally
ask SIDERIUS's Luna-backed interpretation and proposal capabilities what to try
next. This is a small **orchestration example**, not a full research loop:
**it never invokes Implement, Tune or Train.** SIDERIUS v1.1.0's real training
requires an NVIDIA CUDA GPU; use the team's H100 for that later.

You will learn the input (brightness over time), the target (near-core rotation
frequency in cycles/day), the score (R²), and why the next idea needs a new test.
The target is not simply the strongest Fourier peak in the light curve.

## 1. Get a useful result on a CPU laptop

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed.
Get the hackathon kit once (if you already cloned it, just enter that folder):

```bash
git clone https://github.com/yuema137/siderius-hackathon.git "$HOME/siderius-hackathon"
cd "$HOME/siderius-hackathon"
```

Then run this one preview command:

```bash
uv run --no-project --with numpy --with pyarrow python examples/warmup-tess/preview.py \
  --output-dir /tmp/tess-warmup
```

This downloads about **5 MB of public data**: the pinned train/validation
identity manifest and validation light curves. The first `uv` run also installs
NumPy and PyArrow (tens of MB). It does not install SIDERIUS, download the held-out
test split, use a key, or train a neural network. Downloads are checksum-verified
and capped at 32 MiB per file. Windows users can run these commands in WSL.

Open `/tmp/tess-warmup/preview.html` in your browser. It shows three actual light
curves, the input/target explanation and the measured baseline. Read
`/tmp/tess-warmup/baseline.json` for the numbers.

**Expected result:** predict the training mean, approximately **1.08968
cycles/day**, for every validation event. On all 442 validation curves:
**R² ≈ −0.009278, RMSE ≈ 0.56444 cycles/day**. Training and validation stars are
disjoint. The conclusion is deliberately simple: ignoring the light curve gives
a weak baseline; a useful next idea must use information in the curve.
This is not a competition submission or a published neural-model score.

Already downloaded it? Repeat without network:

```bash
uv run --no-project --with numpy --with pyarrow python examples/warmup-tess/preview.py \
  --output-dir /tmp/tess-warmup --offline
```

## 2. Ask Luna to interpret the baseline and propose one model

### On an organizer-prepared machine

The organizer-prepared installation is at `/opt/siderius`. From the hackathon
repository, select it once:

```bash
SIDERIUS=/opt/siderius
```

### On your own laptop

This second step needs the framework environment, which is **larger than the
5 MB preview**. Install it once; skip this step if it is already installed:

```bash
git clone --branch v1.1.0 --depth 1 https://github.com/yuema137/SIDERIUS.git "$HOME/SIDERIUS"
cd "$HOME/SIDERIUS"
uv sync --group dev --frozen
SIDERIUS="$HOME/SIDERIUS"
cd "$HOME/siderius-hackathon"
```

Use the safe key-entry instructions in the main README. The launcher only checks
whether `OPENAI_API_KEY` is present; never paste the key into a command, JSON,
report or chat. External coding-agent access and these internal Luna API calls
are separate services and budgets.

Then run this **one live command** from the hackathon repository:

```bash
"$SIDERIUS/.venv/bin/python" examples/warmup-tess/orchestrate.py \
  --preview-dir /tmp/tess-warmup --output-dir /tmp/tess-warmup-luna --live
```

The script automatically fetches the pinned official TESS task definitions;
you do not need to find or configure another repository. It invokes the native
`ResultInterpretationAgent`, transfers the typed result with
`local_full_context`, and invokes `MLModelProposalAgent`. Both use
`gpt-6-luna`, low reasoning. No generated proposal code is executed.

Bounds: **240 seconds total for capability calls**, at most **4 bridge
requests**, no transport retries, **8,192 output tokens per request**, and a
100,000-character prompt ceiling. These are usage limits, not a guaranteed
currency price; provider prices differ. Native schema-repair attempts, if any, consume the same four-request cap.
A limit can stop an incomplete run.
Use a fresh output directory for each attempt; do not erase failed evidence.

Look for:

- `status.json`: what really started and returned, or why the run is incomplete.
- `interpretation-input.json`, `interpretation.json`: measured baseline evidence
  and the native interpretation.
- `proposal-input.json`, `proposal.json`: the typed handoff and a proposed model.

A proposed improvement is **not** a measured improvement. Read the native status
and diagnostic fields. Ask your coding agent: “Explain the baseline result,
the proposed change, and one comparison that could show the idea is wrong.”
This short caller demonstrates composable capability calls; it does not claim
to be an autonomous, adaptively scheduled exploration campaign.

For a no-API native-input check, omit `--live` and use a new output directory.
That check still needs the installed SIDERIUS environment and official task
source, but it does not contact the model provider. It does not certify a live
Luna run.

## Validation status

The real-data CPU preview was run successfully, including checksums, split
identity, disjoint stars, the baseline above, and an offline rerun. Missing-key
refusal was also exercised and made zero API calls.

The exact SIDERIUS v1.1.0 frozen environment was installed and the pinned
TESS composition plus native `InterpretationInput` ran successfully on CPU.
Nine tests passed: native typed handoff into `ProposalInput` with an explicitly
synthetic interpretation fixture; shared request/time/prompt budget guards with
a mocked transport; and missing-key/corrupt-cache/offline-missing-data guards.
These tests do **not** claim the two native agents completed a real Luna run.

**The final repaired caller completed a real Luna run:** 3 requests in about
37.3 seconds, with 9,109 prompt + 3,625 completion tokens (12,734 total).
All three transport receipts report `ok`. The proposal correctly cites the
measured constant baseline, uses one rotation-frequency scalar per light curve,
and describes a compact CNN as a hypothesis to compare—not a measured gain.
See [the reviewed example](EXAMPLE-RESULT.md) and `VALIDATION.json`.

Earlier live review exposed two defects: empty intermediate reasoning with a
smaller output budget, and a denoising-specific per-time-step phrase in the
native legacy commit prompt. The caller now refuses empty reasoning, carries
original measured evidence directly into the commit, and corrects that exact
phrase for this TESS-only example. See [qualification notes](CONTENT-REVIEW.md).

Successful orchestration is not proof that every suggestion is scientifically
correct. The returned architecture is untrained; screening durations, parameter
estimates and comparisons need checking. One attribution to the “interpretation”
actually concerns background task evidence, and the structured prediction field
is empty. Read the output critically before formal experiments on the H100.

Reproduce the offline guards and native schema/handoff test after installation:

```bash
"$SIDERIUS/.venv/bin/python" -m unittest discover -s examples/warmup-tess -p 'test_*.py' -v
```

## Data provenance

Source: [official PhyTS TESS task](https://github.com/yuema137/siderius-exp/tree/c30ebb8a9f54dca8de89d016f9ebf313aa489daf/tasks/phyts_tess),
with [provenance and limitations](https://github.com/yuema137/siderius-exp/blob/c30ebb8a9f54dca8de89d016f9ebf313aa489daf/tasks/phyts_tess/PROVENANCE.md).
The task predicts near-core rotation from the PhyTS TESS regression subset;
it is not the variability-classification task. The upstream
[dataset](https://huggingface.co/datasets/PhyTS-team/PhyTS-bench) is fetched at
revision `9f203f4c338645a1e4b2c9dc7d6f820269ca5114` with the official checksum pins.
No raw light curves are committed to this kit. Preview scaling is for the plot
only and does not replace the official task's preprocessing.
