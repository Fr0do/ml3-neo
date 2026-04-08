"""Мини-тесты для verify_submission.

Без реального marimo-ноутбука: создаём python-скрипт, который пишет
predictions.npy из переменной ML3_HIDDEN_TEST. Execute-verifier пройдётся по
fallback-пути (python script, без marimo).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml3_grade.execute import verify_submission, ExecutionReport


@pytest.fixture
def project(tmp_path) -> Path:
    # Минимальный «root курса» с нужными путями.
    (tmp_path / "_quarto.yml").write_text("project: {}\n")
    (tmp_path / "grader" / "state").mkdir(parents=True)
    return tmp_path


def make_submission(root: Path, notebook_body: str, preds: np.ndarray) -> Path:
    sub = root / "submission_dir"
    (sub / "submission").mkdir(parents=True, exist_ok=True)
    (sub / "submission" / "notebook.py").write_text(notebook_body)
    np.save(sub / "submission" / "predictions.npy", preds)
    return sub


def make_hidden(root: Path, labels: np.ndarray) -> Path:
    path = root / "hidden.npz"
    np.savez_compressed(
        path,
        images=np.zeros((len(labels), 3, 4, 4), dtype=np.uint8),
        labels=labels,
    )
    return path


def test_reproducible_predictions(project):
    labels = np.array([0, 1, 0, 1, 2, 2, 1, 0], dtype=np.int64)
    hidden = make_hidden(project, labels)

    # "Модель", которая возвращает правильные ответы для всех.
    notebook_body = """
import os, numpy as np
test = np.load(os.environ['ML3_HIDDEN_TEST'])
preds = test['labels'].copy()
np.save(os.environ['ML3_OUTPUT_PREDICTIONS'], preds)
"""
    sub = make_submission(project, notebook_body, labels)
    report = verify_submission(project, "04-cnn-zoo", sub, hidden, timeout=30)
    assert report.ok
    assert report.agreement_rate == 1.0


def test_mismatch_detected(project):
    labels = np.array([0, 1, 0, 1, 2, 2, 1, 0], dtype=np.int64)
    hidden = make_hidden(project, labels)

    # Студент приносит predictions от другой модели; исполняемый код
    # возвращает что-то иное → mismatch.
    stale = np.zeros_like(labels)
    notebook_body = """
import os, numpy as np
test = np.load(os.environ['ML3_HIDDEN_TEST'])
preds = test['labels'].copy()  # "рабочая модель" с идеальной точностью
np.save(os.environ['ML3_OUTPUT_PREDICTIONS'], preds)
"""
    sub = make_submission(project, notebook_body, stale)
    report = verify_submission(project, "04-cnn-zoo", sub, hidden, timeout=30)
    assert not report.ok
    assert report.agreement_rate < 0.98


def test_notebook_crash_reported(project):
    labels = np.array([0, 1, 0], dtype=np.int64)
    hidden = make_hidden(project, labels)

    sub = make_submission(
        project,
        "raise RuntimeError('boom')\n",
        labels,
    )
    report = verify_submission(project, "04-cnn-zoo", sub, hidden, timeout=30)
    assert not report.ok
    assert "exited" in report.reason or "crash" in report.reason.lower()


def test_missing_output(project):
    labels = np.array([0, 1, 0], dtype=np.int64)
    hidden = make_hidden(project, labels)
    sub = make_submission(project, "pass\n", labels)
    report = verify_submission(project, "04-cnn-zoo", sub, hidden, timeout=30)
    assert not report.ok
    assert "did not write" in report.reason
