"""Эвристика против тренировки на test set.

Полную защиту обеспечивает только то, что test labels вообще не лежат в
репозитории — они в data/hidden/, gitignored. Этот тест ловит более грубое:
- захардкоженные predictions без модели,
- попытки подгрузить hidden-каталог из ноутбука.
"""

from __future__ import annotations

from pathlib import Path

HW_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = HW_ROOT / "submission"


def test_notebook_does_not_open_hidden():
    nb = SUBMISSION / "notebook.py"
    if not nb.exists():
        return
    text = nb.read_text()
    forbidden_substrings = [
        "data/hidden",
        "test.npz",  # имя файла скрытого теста
        "labels.npy",
    ]
    for sub in forbidden_substrings:
        assert sub not in text, \
            f"notebook.py refers to '{sub}', which suggests test leakage"


def test_predictions_not_constant():
    import numpy as np
    p = SUBMISSION / "predictions.npy"
    if not p.exists():
        return
    preds = np.load(p)
    if preds.size < 10:
        return
    unique = np.unique(preds)
    assert len(unique) > 1, \
        "all predictions identical — looks like a stub, not a model"
