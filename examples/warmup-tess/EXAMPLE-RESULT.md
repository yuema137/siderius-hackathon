# What a completed warm-up tells you

**Measured:** predicting the training-target mean (1.08968 cycles/day) for all
442 validation light curves gives R² −0.009278. This ignores each curve and is
therefore a useful, deliberately weak reference.

**Proposed, not measured:** the real Luna run suggested a compact multiscale
convolutional model that reads a light curve and returns one rotation-frequency
scalar. It suggested comparing average-plus-maximum temporal pooling with
average-only pooling under matched conditions.

**Next scientific question:** does using the curve improve validation R² over
the constant reference, and does that pooling change help under the same split
and training budget? Neither answer is established by this warm-up. Run those
experiments later using the official GPU workflow.

The native proposal includes generated hyperparameters and a CUDA training
configuration. Those are candidate settings for later experimentation; the
warm-up does not execute them on your CPU. Treat background-paper comparisons,
parameter estimates and suggested schedules as claims to inspect, not facts
measured by this preview.

Qualification: real native Interpretation → Proposal completed in 3 requests,
approximately 37.3 seconds, 12,734 total tokens. See VALIDATION.json for limits
and CONTENT-REVIEW.md for defects caught and remaining caveats.
