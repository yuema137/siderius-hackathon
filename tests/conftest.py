from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

EXAMPLE = REPO / "examples" / "quickstart_classification"


@pytest.fixture
def submission(tmp_path: Path) -> Path:
    """A private copy of the worked example, named by its task id."""
    target = tmp_path / "quickstart_classification"
    shutil.copytree(EXAMPLE, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return target
