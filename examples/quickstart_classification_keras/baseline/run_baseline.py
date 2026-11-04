"""Standalone Keras baseline for the quickstart classification task: a tiny MLP on CPU.

Runs in the isolated environment the checker builds from ``requirements.txt``
(SIDERIUS plus pinned Keras 3). Keras uses the PyTorch backend, which SIDERIUS
already installs, so no TensorFlow download is needed::

    run_baseline.py init     --seed S --output init_model.keras
    run_baseline.py train    --data_dir D --seed S --output_dir O --init_model init_model.keras
    run_baseline.py evaluate --data_dir D --model final_model.keras --output_dir O

``D`` is the folder ``fetch_data.py --output_dir`` wrote. ``train`` writes
``O/final_model.keras`` and ``O/metrics.json``; ``evaluate`` writes
``O/metrics.json`` (and ``O/predictions.json``). ``init`` is this example's
helper for creating ``init_model.keras``; the kit does not call it.

Models are loaded with ``keras.saving.load_model`` in its default safe mode,
which refuses serialized Python lambdas. ``metrics.json`` holds the values
computed by the task's declared metric plugins via SIDERIUS's composer.
Determinism: CPU, one thread, deterministic algorithms, ``keras.utils.set_random_seed``.
"""

from __future__ import annotations

import os

os.environ["KERAS_BACKEND"] = "torch"  # must be set before keras is imported

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "task" / "task.yaml"

#: Everything needed to retrain, together with the seed (also in baseline_result.json).
HYPERPARAMETERS = {"hidden_units": 16, "learning_rate": 0.01, "batch_size": 32, "epochs": 150}


def setup(seed: int) -> None:
    import keras
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    keras.utils.set_random_seed(seed)


def build_model():
    import keras

    return keras.Sequential(
        [
            keras.Input(shape=(4,)),
            keras.layers.Dense(HYPERPARAMETERS["hidden_units"], activation="relu"),
            keras.layers.Dense(2),
        ]
    )


def compose_task():
    from workflows.task_composition import compose_run_task_bindings

    return compose_run_task_bindings(str(MANIFEST))


def full_scope(data_path, *, training: bool):
    from execute_tools.task_data_path import ScopeBuildRequest

    request = ScopeBuildRequest(round_kind="formal", selection_strategy="snapshot", portion=1.0)
    return data_path.build_training_scope(request) if training else data_path.build_eval_scope(request)


def as_arrays(dataset) -> tuple[np.ndarray, np.ndarray]:
    features = np.stack([dataset[i][0].numpy() for i in range(len(dataset))]).astype("float32")
    labels = np.asarray([int(dataset[i][1]) for i in range(len(dataset))])
    return features, labels


def evaluate_model(model, composition, data_dir: Path, output_dir: Path) -> dict[str, float]:
    """Predict validation.csv and score with the task's declared metrics."""
    from execute_tools.task_data_path import EvalMaterializationParams, bind_task_data_path

    data_path = composition.task_data_path
    scope = full_scope(data_path, training=False)
    features, _ = as_arrays(data_path.validation_dataset(scope, EvalMaterializationParams(data_dir=str(data_dir))))
    sample_ids = list(data_path.labels_by_sample_id(scope, str(data_dir)))
    logits = np.asarray(model.predict(features, verbose=0))
    predictions = dict(zip(sample_ids, (int(p) for p in logits.argmax(axis=1)), strict=True))
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


def command_init(args: argparse.Namespace) -> int:
    setup(args.seed)
    model = build_model()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"wrote {args.output}")
    return 0


def command_train(args: argparse.Namespace) -> int:
    import keras
    from execute_tools.task_data_path import EpochSamplingParams

    if args.init_model is None:
        raise SystemExit("this Keras baseline needs --init_model (baseline/init_model.keras)")
    setup(args.seed)
    composition = compose_task()
    data_path = composition.task_data_path
    features, labels = as_arrays(
        data_path.training_dataset(
            full_scope(data_path, training=True), EpochSamplingParams(data_dir=str(args.data_dir))
        )
    )
    model = keras.saving.load_model(args.init_model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=HYPERPARAMETERS["learning_rate"]),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    started = time.perf_counter()
    model.fit(
        features,
        labels,
        batch_size=HYPERPARAMETERS["batch_size"],
        epochs=HYPERPARAMETERS["epochs"],
        shuffle=True,
        verbose=0,
    )
    train_seconds = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "final_model.keras")
    metrics = evaluate_model(model, composition, args.data_dir, args.output_dir)
    print(json.dumps({"metrics": metrics, "train_seconds": round(train_seconds, 3)}, sort_keys=True))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    import keras

    setup(0)
    model = keras.saving.load_model(args.model)
    started = time.perf_counter()
    metrics = evaluate_model(model, compose_task(), args.data_dir, args.output_dir)
    print(json.dumps({"metrics": metrics, "evaluate_seconds": round(time.perf_counter() - started, 3)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description="quickstart classification Keras baseline")
    commands = cli.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="write the seeded initial model (example helper)")
    init.add_argument("--seed", type=int, required=True)
    init.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train", help="train from the initial model; write final_model.keras + metrics")
    train.add_argument("--data_dir", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--output_dir", type=Path, required=True)
    train.add_argument("--init_model", type=Path)
    evaluate = commands.add_parser("evaluate", help="score a saved model on the evaluation split")
    evaluate.add_argument("--data_dir", type=Path, required=True)
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--output_dir", type=Path, required=True)
    args = cli.parse_args(argv)
    handler = {"init": command_init, "train": command_train, "evaluate": command_evaluate}[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
