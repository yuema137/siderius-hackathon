# Local walkthrough

**Start with [quickstart_classification](quickstart_classification/).** It shows
what a task package looks like: a brief, data preparation, evaluation, and a
small PyTorch baseline. The synthetic data is generated locally: four CSV files,
256 rows total. No data download or Zenodo account is needed for this walkthrough.

This is a packaging example, not the scale or scientific depth expected of a
real challenge. **Do not upload this example as your submission.**

## Give this to your agent

```text
Walk me through examples/quickstart_classification in this kit.
Use SIDERIUS v1.1.0 and follow AGENTS.md, “Local walkthrough and advanced variants”.
Generate the tiny quickstart dataset locally outside the package folder.
Verify its checksums, prepare the train/validation files, and evaluate the saved
baseline. Explain how the question, data, metrics, and baseline fit together.
Ask before retraining or installing dependencies.

This is a local walkthrough: its Zenodo record is intentionally a placeholder.
Show the real checker results, including the expected Zenodo failure.
Do not bypass that check, invent a DOI, or claim the package is ready to submit.
```

## What you should learn

- How the task defines inputs, outputs, splits, and evaluation.
- How a saved baseline produces results you can reproduce.
- What to change when you build your own scientific challenge.

For a real submission, publish your data on Zenodo, replace the placeholder
record and DOI, and pass the complete checker. The local walkthrough does not
relax those requirements.

Already comfortable with the format? Optional [advanced variants](advanced/)
show scikit-learn and Keras baselines. Start with one framework, not all three.
