"""Public sanity checks for hw 01-backprop.

Запускается студентом локально: `pytest homework/01-backprop/tests/`.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

HW_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = HW_ROOT / "submission"

def test_submission_dir_exists():
    assert SUBMISSION.exists(), "submission/ folder is missing"

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

def test_results_json_valid():
    res = SUBMISSION / "results.json"
    if res.exists():
        try:
            json.loads(res.read_text())
        except json.JSONDecodeError:
            pytest.fail("submission/results.json is not a valid JSON file")
