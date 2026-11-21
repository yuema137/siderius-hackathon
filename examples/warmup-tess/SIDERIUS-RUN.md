# CPU TESS warm-up declaration

Preparation is the default. `preview.py` reads pinned public training metadata
and validation light curves, computes the training-mean baseline, and writes an
HTML preview and measured JSON. It does not call a model provider.

`orchestrate.py` without `--live` only prepares and validates the native
interpretation request. `--live` explicitly enables one Interpretation call
followed by one Proposal call, using the typed native handoff. This is a bounded
teaching sequence, not an autonomous search campaign. The final caller completed a real Luna run in 3 requests and approximately
37.3 seconds, with 12,734 total tokens. Native artifacts were checked for correct
baseline evidence, scalar-per-curve target, and no unmeasured-success claim.
Nine offline tests also passed. Earlier failures and remaining content caveats
are documented in CONTENT-REVIEW.md and VALIDATION.json. No model was trained.

Framework: SIDERIUS v1.1.0, using that checkout's own frozen `.venv/bin/python`.
Task source: official siderius-exp commit
`c30ebb8a9f54dca8de89d016f9ebf313aa489daf`, read-only task definitions.
Data: preview.py's two checksum-pinned files; no held-out test population.
Metric: declared TESS R², higher is better; MAE/RMSE are contextual.
Baseline: measured mean of training targets, evaluated on 442 validation curves.
No native training or Health-check success is claimed for that baseline.

Enabled live capabilities: native interpretation and proposal only.
Internal routing: OpenAI gpt-6-luna, low reasoning; only OPENAI_API_KEY from
launch environment. No credential values belong in any project file.
External coding-agent model: selected/billed separately by the participant.
Forbidden in this warm-up: implementation, training, tuning, executing proposed
code, scientific web retrieval, extra data access, evaluator/test data access.

Limits shared across the two calls: 240-second wall timer, 4 bridge requests,
8192 output tokens/request, 100000 prompt characters/request, no transport
retries. A provider failure/timeout or schema error is an incomplete run, not
permission to extend these limits. These limits do not guarantee a currency
price. API rates and external-agent charges are separate.

Output: the CLI's fresh output directory, including started/failure states,
native requests/returns and native bridge usage receipts. Preserve failed runs.
Native capability calls only reason over the supplied summary and task; they do
not materialize data or execute training, so this preparation route does not
require bubblewrap/CUDA. The full native training route does require the
framework's documented Linux/CUDA execution environment and is out of scope.
