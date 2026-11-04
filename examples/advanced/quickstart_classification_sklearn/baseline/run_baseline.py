"""Standalone scikit-learn baseline for the quickstart classification task: a decision tree.

Runs in the isolated environment the checker builds from ``requirements.txt``
(SIDERIUS plus pinned scikit-learn and skops), because the task's own plugins
(data access and metrics) import the SIDERIUS framework::

    run_baseline.py train    --data_dir D --seed S --output_dir O
    run_baseline.py evaluate --data_dir D --model final_model.skops --output_dir O

``D`` is the folder ``fetch_data.py --output_dir`` wrote. ``train`` writes
``O/final_model.skops`` and ``O/metrics.json``; ``evaluate`` writes
``O/metrics.json`` (and ``O/predictions.json``). A decision tree has no
initial model: the seed and the hyperparameters below reproduce it.

The model is saved with skops, which loads without unpickling arbitrary
objects; ``evaluate`` refuses a file containing types skops does not trust.
``metrics.json`` holds the values computed by the task's declared metric
plugins (``accuracy`` and ``balanced_accuracy``) via SIDERIUS's composer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "task" / "task.yaml"

#: Everything needed to retrain, together with the seed (also in baseline_result.json).
HYPERPARAMETERS = {"criterion": "gini", "max_depth": 5, "min_samples_leaf": 2}


def compose_task():
    from workflows.task_composition import compose_run_task_bindings

    return compose_run_task_bindings(str(MANIFEST))


def full_scope(data_path, *, training: bool):
    from execute_tools.task_data_path import ScopeBuildRequest

    request = ScopeBuildRequest(round_kind="formal", selection_strategy="snapshot", portion=1.0)
    return data_path.build_training_scope(request) if training else data_path.build_eval_scope(request)


def as_arrays(dataset) -> tuple[np.ndarray, np.ndarray]:
    features = np.stack([dataset[i][0].numpy() for i in range(len(dataset))])
    labels = np.asarray([int(dataset[i][1]) for i in range(len(dataset))])
    return features, labels


def evaluate_model(model, composition, data_dir: Path, output_dir: Path) -> dict[str, float]:
    """Predict validation.csv and score with the task's declared metrics."""
    from execute_tools.task_data_path import EvalMaterializationParams, bind_task_data_path

    data_path = composition.task_data_path
    scope = full_scope(data_path, training=False)
    features, _ = as_arrays(data_path.validation_dataset(scope, EvalMaterializationParams(data_dir=str(data_dir))))
    # labels_by_sample_id lists the selected rows in dataset order.
    sample_ids = list(data_path.labels_by_sample_id(scope, str(data_dir)))
    predictions = dict(zip(sample_ids, (int(p) for p in model.predict(features)), strict=True))
    output_dir.mkdir(parents=True, exist_ok=True)
    deliverable = output_dir / "predictions.json"
    deliverable.write_text(json.dumps(predictions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics: dict[str, float] = {}
    with bind_task_data_path(data_path):
        for metric in [composition.metric, *composition.secondary_metrics]:
            outcome = metric.evaluate(
                {0: str(deliverable)}, evaluation_payload=predictions, task_scope=scope, data_dir=str(data_dir)
            )
            scalar = getattr(outcome, "scalar", None)
            if scalar is None:
                raise RuntimeError(f"metric {metric.spec.id} could not score: {outcome}")
            metrics[metric.spec.id] = float(scalar)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def command_train(args: argparse.Namespace) -> int:
    import skops.io
    from execute_tools.task_data_path import EpochSamplingParams
    from sklearn.tree import DecisionTreeClassifier

    if args.init_model is not None:
        raise SystemExit("a decision tree has no initial model; omit --init_model")
    composition = compose_task()
    data_path = composition.task_data_path
    features, labels = as_arrays(
        data_path.training_dataset(
            full_scope(data_path, training=True), EpochSamplingParams(data_dir=str(args.data_dir))
        )
    )
    started = time.perf_counter()
    model = DecisionTreeClassifier(random_state=args.seed, **HYPERPARAMETERS).fit(features, labels)
    train_seconds = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    skops.io.dump(model, args.output_dir / "final_model.skops")
    metrics = evaluate_model(model, composition, args.data_dir, args.output_dir)
    print(json.dumps({"metrics": metrics, "train_seconds": round(train_seconds, 3)}, sort_keys=True))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    import skops.io

    untrusted = skops.io.get_untrusted_types(file=args.model)
    if untrusted:
        raise SystemExit(f"{args.model} contains types skops does not trust: {untrusted}")
    model = skops.io.load(args.model, trusted=[])
    started = time.perf_counter()
    metrics = evaluate_model(model, compose_task(), args.data_dir, args.output_dir)
    print(json.dumps({"metrics": metrics, "evaluate_seconds": round(time.perf_counter() - started, 3)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description="quickstart classification decision-tree baseline")
    commands = cli.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fit the tree; write final_model.skops + metrics")
    train.add_argument("--data_dir", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--output_dir", type=Path, required=True)
    train.add_argument("--init_model", type=Path, help="not used: a tree has no initial model")
    evaluate = commands.add_parser("evaluate", help="score a saved model on the evaluation split")
    evaluate.add_argument("--data_dir", type=Path, required=True)
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--output_dir", type=Path, required=True)
    args = cli.parse_args(argv)
    return {"train": command_train, "evaluate": command_evaluate}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
