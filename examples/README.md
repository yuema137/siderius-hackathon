# Local walkthrough

**Start with [quickstart_classification](quickstart_classification/).** It shows
what a task package looks like: a brief, data preparation, evaluation, and a
small PyTorch baseline. The synthetic data is four small CSV files, 256 rows in
total, published on Zenodo ([record 23988652](https://zenodo.org/records/23988652)).
The checker downloads and verifies them; no Zenodo account is needed.

This is a packaging example, not the scale or scientific depth expected of a
real challenge. **Do not upload this example as your submission.**

## Give this to your agent

```text
Walk me through examples/quickstart_classification in this kit.
Use SIDERIUS v1.1.0 and follow AGENTS.md, “Local walkthrough and advanced variants”.
Let the kit's checker download the tiny quickstart dataset from Zenodo outside
the package folder. Verify its checksums, prepare the train/validation files,
and evaluate the saved baseline. Explain how the question, data, metrics, and baseline fit together.
Ask before retraining or installing dependencies.

Show the real checker results. Do not claim this example is my submission.
```

## What you should learn

- How the task defines inputs, outputs, splits, and evaluation.
- How a saved baseline produces results you can reproduce.
- What to change when you build your own scientific challenge.

For your own submission, publish your data as your own Zenodo record, put its
record id and DOI in `data_source.json` and `TASK_CARD.md`, and pass the
complete checker.

Already comfortable with the format? Optional [advanced variants](advanced/)
show scikit-learn and Keras baselines. Start with one framework, not all three.
