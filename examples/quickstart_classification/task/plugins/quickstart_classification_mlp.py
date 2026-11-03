"""Baseline model as a SIDERIUS model plugin (the Track 1 bonus).

Linear(4 -> hidden) -> ReLU -> Linear(hidden -> 2), the SIDERIUS quickstart
reference MLP. ``baseline/run_baseline.py`` builds this same class, so
``baseline/final_weights.pt`` loads into the model SIDERIUS runs.

Model plugin contract: the module defines ``PLUGIN_MODEL_TYPE``,
``PLUGIN_OUTPUT_TYPE``, ``PLUGIN_CONFIG_CLASS`` and ``PLUGIN_MODEL_CLASS``. The
forward pass maps ``[B, 4] float32`` to ``[B, 2]`` logits, matching
``forward_contract.model_io`` in ``declared/task_config.yaml``.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel, Field
from torch import nn

FEATURE_COUNT = 4
CLASS_COUNT = 2

PLUGIN_MODEL_TYPE = "quickstart_classification_mlp"
PLUGIN_OUTPUT_TYPE = "classifier"


class QuickstartClassificationMlpConfig(BaseModel):
    """Settings the planner may choose; unknown planner fields are ignored, not refused."""

    model_type: str = Field(default="quickstart_classification_mlp")
    #: Read by the training engine at start-up; a tabular input has no segments.
    segmentation_size: int = Field(default=FEATURE_COUNT, ge=1)
    batch_size: int = Field(default=32, ge=1)
    hidden_dim: int = Field(default=16, ge=2, le=1024)


PLUGIN_CONFIG_CLASS = QuickstartClassificationMlpConfig


class QuickstartClassificationMlp(nn.Module):
    def __init__(self, config: QuickstartClassificationMlpConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_COUNT, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, CLASS_COUNT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits: cross-entropy and the deliverable codec (argmax) expect them."""
        return self.net(x)


PLUGIN_MODEL_CLASS = QuickstartClassificationMlp
