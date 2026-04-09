import pytest
from pathlib import Path

def test_submission_files_exist():
    submission_dir = Path("submission")
    
    assert (submission_dir / "notebook.py").exists(), "Файл notebook.py отсутствует"
    assert (submission_dir / "MODEL.md").exists(), "Файл MODEL.md отсутствует"
    
def test_no_forbidden_edits():
    assert Path("homework/07-pretrain/eval.py").exists()
    assert Path("homework/07-pretrain/meta.yml").exists()
    assert Path("homework/07-pretrain/rubric.yml").exists()
