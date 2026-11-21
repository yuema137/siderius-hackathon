# Warm-up qualification notes

The warm-up is a measured CPU preview followed by native Interpretation and
Proposal. It does not implement or train the proposed model. A returned JSON
object alone is not a scientific success criterion.

## Defects found by actual live review

1. The initial medium-reasoning, 4096-token configuration returned empty visible
   intermediate reasoning. The final proposal then claimed no interpretation
   was provided, although its typed input retained the findings. The caller
   now uses Luna low reasoning with an 8192-token ceiling, refuses empty text,
   and carries measured preview evidence into the native legacy commit directly.
2. The second run correctly cited R² −0.00928 on 442 validation curves and
   distinguished a proposed improvement from a measured result. Full inspection
   found that the v1.1.0 native legacy commit prompt hardcodes “the head emits
   one continuous value per time step,” even for the TESS `[B, 1]` output.
   This TESS-only caller corrects that exact prompt literal to one near-core
   rotation scalar per light curve. It does not patch the framework or alter
   returned model artifacts after generation.

The nine offline tests include native schema/handoff validation, prompt capture
showing baseline evidence and scalar-target semantics, empty-reasoning refusal,
shared request/time/prompt limits, and cache/missing-key guards.

See `VALIDATION.json` for the final live qualification status. Model-generated
explanations are hypotheses; baseline numbers alone are measured here. Formal
training, Health checks, scientific gains, and deployment quality are not
certified by this warm-up.

## Final live review

The final run made 3 requests in approximately 37.3 seconds. All three receipts
report `ok`, totaling 9,109 prompt and 3,625 completion tokens. The native agents
returned a `multiscale_residual_curve_regressor` proposal. Review confirmed the
constant mean (1.0897 cycles/day), 442 validation curves and R² about −0.0093;
the output is one scalar per curve, and suggested gains remain hypotheses.

The proposed average-plus-maximum versus average-only pooling comparison is
concrete, but it has not been run. Generated parameter count and screening
schedule are unverified suggestions. `falsifiable_prediction` is null in this
legacy path. One S4D-background statement is attributed to “the interpretation”
instead of the supplied task context; do not mistake it for preview evidence.
These limits are acceptable for a critically reviewed proposal exercise, not
for claiming a scientifically validated model. No further API calls are needed.
