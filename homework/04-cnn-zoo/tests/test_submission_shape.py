"""Public sanity checks for hw 04-cnn-zoo.

Запускается студентом локально: `pytest homework/04-cnn-zoo/tests/`.
Грейдер запускает то же самое перед leaderboard-eval. Если эти тесты
красные, грейдер не считает метрику.

Тесты НЕ должны иметь доступа к скрытому test set — только к публичным
индексам и форматам.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


HW_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = HW_ROOT / "submission"
PUBLIC = HW_ROOT / "data" / "public"


def test_submission_dir_exists():
    assert SUBMISSION.exists(), "submission/ folder is missing"


def test_predictions_file_exists():
    assert (SUBMISSION / "predictions.npy").exists(), \
        "submission/predictions.npy is missing"


def test_predictions_shape_and_dtype():
    preds = np.load(SUBMISSION / "predictions.npy")
    assert preds.dtype == np.int64, f"dtype must be int64, got {preds.dtype}"
    assert preds.ndim == 1, f"shape must be (N,), got {preds.shape}"

    test_index_path = PUBLIC / "test_index.json"
    if not test_index_path.exists():
        pytest.skip("test_index.json not in repo (public data not unpacked)")
    expected_n = len(json.loads(test_index_path.read_text()))
    assert preds.shape[0] == expected_n, \
        f"expected {expected_n} predictions, got {preds.shape[0]}"


def test_predictions_in_class_range():
    preds = np.load(SUBMISSION / "predictions.npy")
    assert preds.min() >= 0, "negative class id"
    assert preds.max() < 40, f"class id ≥ 40 (TinyImageNet subset has 40)"


def test_model_md_present():
    md = SUBMISSION / "MODEL.md"
    assert md.exists(), "submission/MODEL.md is missing — judge needs it"
    assert len(md.read_text()) > 200, "MODEL.md is too short to evaluate"


def test_notebook_present():
    nb = SUBMISSION / "notebook.py"
    assert nb.exists(), "submission/notebook.py is missing"
    text = nb.read_text()
    assert "=== SUBMISSION (start) ===" in text, \
        "submission marker missing in notebook.py"
    assert "=== SUBMISSION (end) ===" in text, \
        "submission marker missing in notebook.py"
