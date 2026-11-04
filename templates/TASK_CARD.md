# Task card: <task title>

<!--
Copy this file to <your package folder>/TASK_CARD.md and replace every comment
with your text. The title heading is free text: the task's registered title and
its id (t1-NNN) come from the submission portal, where you create the task
before uploading. Keep the nine "##" headings exactly as written: the checker looks for
them and refuses an empty section. Comments like this one are ignored, so a
section that still holds only its comment counts as empty. You may add "###"
subheadings inside a section.
-->

## Scientific question

<!--
The question in two to five sentences, for a scientist outside your field.
What is measured, what should be predicted, and why the answer matters.
-->

## Task design

<!--
What one sample is; the model input (shape, units, dtype) and output (shape,
meaning); the splits (which rows, files, runs, detectors or time ranges go to
training, validation and any held-out test) and why the split matches the
scientific use. Name the task.yaml settings that implement this.
-->

## Data

<!--
Required facts, one per line:
- DOI: 10.5281/zenodo.<record id>   (must match data_source.json)
- Size: total bytes on Zenodo and number of files; rows/events/images per split
- License: the Zenodo license id (for example cc-by-4.0)
- Access: public, or restricted and how organizers get access (share link)
- Processing: what fetch_data.py does to the Zenodo files and what it writes
Also: where the data comes from (instrument, simulation, publication) and any
known quality issues.
-->

## Evaluation

<!--
Primary metric (required): its id in task.yaml, direction (higher/lower is
better), formula in words, and why it answers the scientific question.
Optional and encouraged: secondary metrics, a combined formula, and Health
checks (which invalid outputs each one catches, and why that failure matters
scientifically). Say what a trivial predictor (constant, majority class,
mean) scores, so readers can judge the baseline.
-->

## Baseline

<!--
Method: framework (PyTorch, TensorFlow/Keras, scikit-learn, XGBoost,
LightGBM, CatBoost or other), model, input features, training recipe and
hyperparameters; the model file format (final_model.<format>, and
init_model.<format> for neural networks) and any extra packages in
baseline/requirements.txt. Then the score on the evaluation split for every
declared metric (copy from baseline/baseline_result.json), the seed, the
hardware and wall time to train and to evaluate, and whether it needs a GPU.
Mention the bonus SIDERIUS model plugin if you provide one (PyTorch only).
-->

## Novelty

<!--
What is new compared with existing public benchmarks and published results on
this or similar data. Cite the closest prior work.
-->

## Difficulty

<!--
Why the task is hard and where the headroom is: the gap between the baseline
and the best plausible score, noise level or irreducible error, data size,
class imbalance, distribution shift between splits.
-->

## Known shortcuts and leakage risks

<!--
Ways a model could score well without solving the science: duplicated or
near-duplicated samples across splits, identifiers or file order that encode
the label, simulation artifacts, normalization computed on validation data.
For each, say how the task design prevents it or what Health check would
catch it. "None known" is acceptable only after you have looked.
-->

## Contact

<!--
Name, affiliation and e-mail of the task author(s), and who answers questions
during Track 2.
-->
