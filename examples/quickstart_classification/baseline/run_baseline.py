"""Standalone baseline for quickstart_classification: a small MLP trained on CPU.

Run with the SIDERIUS checkout's interpreter, because the task's own plugins
(data access and metrics) import the SIDERIUS framework::

    PY=/path/to/SIDERIUS/.venv/bin/python
    $PY run_baseline.py init     --seed S --output W.pt
    $PY run_baseline.py train    --data_dir D --init_weights W.pt --seed S --output_dir O
    $PY run_baseline.py evaluate --data_dir D --weights F.pt --output_dir O

``D`` is the folder ``fetch_data.py --output_dir`` wrote. ``train`` writes
``O/final_weights.pt`` and ``O/metrics.json``; ``evaluate`` writes
``O/metrics.json`` (and ``O/predictions.json``). ``init`` is this example's
helper for creating ``init_weights.pt``; the kit does not call it.

How the task is used. ``task/task.yaml`` is composed by SIDERIUS's own composer,
so training and evaluation read the rows through the task's data path
(``train.csv`` for training, ``validation.csv`` for evaluation) and
``metrics.json`` holds the values computed by the task's declared metric
plugins (``accuracy`` and ``balanced_accuracy``). The model is the task's model
plugin ``quickstart_classification_mlp`` with its default configuration.

Determinism. CPU only, one thread, deterministic algorithms; the seed fixes the
order of training batches. Training from the same initial weights and seed
gives the same final weights on the same software stack.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "task" / "task.yaml"
MODEL_PLUGIN = HERE.parent / "task" / "plugins" / "quickstart_classification_mlp.py"

EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-4


def setup_torch() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def build_model() -> torch.nn.Module:
    spec = importlib.util.spec_from_file_location("quickstart_classification_mlp_baseline", MODEL_PLUGIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PLUGIN_MODEL_CLASS(module.PLUGIN_CONFIG_CLASS())


def compose_task():
    from workflows.task_composition import compose_run_task_bindings

    return compose_run_task_bindings(str(MANIFEST))


def full_scope(data_path, *, training: bool):
    from execute_tools.task_data_path import ScopeBuildRequest

    request = ScopeBuildRequest(round_kind="formal", selection_strategy="snapshot", portion=1.0)
    return data_path.build_training_scope(request) if training else data_path.build_eval_scope(request)


def load_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def evaluate_model(model: torch.nn.Module, composition, data_dir: Path, output_dir: Path) -> dict[str, float]:
    """Predict validation.csv and score with the task's declared metrics."""
    from execute_tools.task_data_path import EvalMaterializationParams, bind_task_data_path

    data_path = composition.task_data_path
    scope = full_scope(data_path, training=False)
    dataset = data_path.validation_dataset(scope, EvalMaterializationParams(data_dir=str(data_dir)))
    # labels_by_sample_id lists the selected rows in dataset order.
    sample_ids = list(data_path.labels_by_sample_id(scope, str(data_dir)))
    features = torch.stack([dataset[i][0] for i in range(len(dataset))])
    model.eval()
    with torch.no_grad():
        predicted = model(features).argmax(dim=1).tolist()
    if len(predicted) != len(sample_ids):
        raise RuntimeError("prediction count differs from the validation rows")
    predictions = dict(zip(sample_ids, (int(p) for p in predicted), strict=True))
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
    setup_torch()
    torch.manual_seed(args.seed)
    model = build_model()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"wrote {args.output}")
    return 0


def command_train(args: argparse.Namespace) -> int:
    from execute_tools.task_data_path import EpochSamplingParams

    setup_torch()
    composition = compose_task()
    data_path = composition.task_data_path
    dataset = data_path.training_dataset(
        full_scope(data_path, training=True), EpochSamplingParams(data_dir=str(args.data_dir))
    )
    features = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.stack([dataset[i][1] for i in range(len(dataset))])
    model = build_model()
    model.load_state_dict(load_state(args.init_weights))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(args.seed)
    started = time.perf_counter()
    model.train()
    for epoch in range(EPOCHS):
        order = torch.randperm(len(features), generator=generator)
        total = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features[batch]), labels[batch])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        if epoch % 25 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d} train loss {total / len(features):.4f}", flush=True)
    train_seconds = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "final_weights.pt")
    started = time.perf_counter()
    metrics = evaluate_model(model, composition, args.data_dir, args.output_dir)
    print(
        json.dumps(
            {
                "metrics": metrics,
                "train_seconds": round(train_seconds, 3),
                "evaluate_seconds": round(time.perf_counter() - started, 3),
            },
            sort_keys=True,
        )
    )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    setup_torch()
    composition = compose_task()
    model = build_model()
    model.load_state_dict(load_state(args.weights))
    started = time.perf_counter()
    metrics = evaluate_model(model, composition, args.data_dir, args.output_dir)
    print(json.dumps({"metrics": metrics, "evaluate_seconds": round(time.perf_counter() - started, 3)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description="quickstart_classification baseline")
    commands = cli.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="write seeded initial weights (example helper)")
    init.add_argument("--seed", type=int, required=True)
    init.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train", help="train from initial weights; write final weights + metrics")
    train.add_argument("--data_dir", type=Path, required=True)
    train.add_argument("--init_weights", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--output_dir", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="score weights on the evaluation split")
    evaluate.add_argument("--data_dir", type=Path, required=True)
    evaluate.add_argument("--weights", type=Path, required=True)
    evaluate.add_argument("--output_dir", type=Path, required=True)
    args = cli.parse_args(argv)
    handler = {"init": command_init, "train": command_train, "evaluate": command_evaluate}[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
