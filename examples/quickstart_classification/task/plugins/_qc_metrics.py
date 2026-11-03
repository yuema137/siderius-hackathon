"""Scores: accuracy (primary) and balanced accuracy (secondary) on validation.csv.

Declared by ``declared/metric_accuracy.json`` and
``declared/metric_balanced_accuracy.json``; bound by ``task.yaml``. These
classes own only the arithmetic. The labels are read from the validation CSV
through the bound task data path (``labels_by_sample_id``), so the metric and the
datasets parse the file with one set of rules and select the same rows. A
validation row with no prediction counts as wrong; predictions for other IDs
are ignored. ``baseline/run_baseline.py`` calls these same classes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from execute_tools.evaluation_metric import EvaluationMetric
from execute_tools.task_data_path import require_bound_task_data_path


def _truth(task_scope: Any, data_dir: str | None) -> dict[str, int]:
    if data_dir is None:
        raise ValueError("these metrics read labels from the data folder; data_dir is None")
    labels_by_sample_id = getattr(require_bound_task_data_path(), "labels_by_sample_id", None)
    if not callable(labels_by_sample_id):
        raise TypeError("the bound task data path does not provide labels_by_sample_id")
    truth = labels_by_sample_id(task_scope, data_dir)
    if not isinstance(truth, dict) or not truth:
        raise ValueError("labels_by_sample_id must return a non-empty {sample_id: class} mapping")
    return truth


class AccuracyMetric(EvaluationMetric):
    """Fraction of selected validation rows whose predicted class is correct."""

    IMPLEMENTS: ClassVar[tuple[str, ...]] = ("accuracy",)

    def _compute(
        self,
        deliverables: Mapping[int, str],
        /,
        *,
        evaluation_payload: Mapping[str, int],
        task_scope: Any,
        data_dir: str | None = None,
    ) -> tuple[float, list[float | None] | None, tuple[str, ...]]:
        truth = _truth(task_scope, data_dir)
        correct = sum(evaluation_payload.get(sample_id) == label for sample_id, label in truth.items())
        return correct / len(truth), None, ()


class BalancedAccuracyMetric(EvaluationMetric):
    """Mean over classes of per-class recall on the selected validation rows."""

    IMPLEMENTS: ClassVar[tuple[str, ...]] = ("balanced_accuracy",)

    def _compute(
        self,
        deliverables: Mapping[int, str],
        /,
        *,
        evaluation_payload: Mapping[str, int],
        task_scope: Any,
        data_dir: str | None = None,
    ) -> tuple[float, list[float | None] | None, tuple[str, ...]]:
        truth = _truth(task_scope, data_dir)
        recalls = []
        for label in sorted(set(truth.values())):
            members = [sample_id for sample_id, value in truth.items() if value == label]
            correct = sum(evaluation_payload.get(sample_id) == label for sample_id in members)
            recalls.append(correct / len(members))
        return sum(recalls) / len(recalls), None, ()
