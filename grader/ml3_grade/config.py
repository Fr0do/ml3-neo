"""Утилиты загрузки конфигов и поиска корня курса."""

from __future__ import annotations

from pathlib import Path

import yaml


def find_course_root(start: Path | None = None) -> Path:
    """Поднимаемся вверх, пока не найдём _quarto.yml."""
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "_quarto.yml").exists():
            return cand
    raise SystemExit(
        "ml3-grade: не нашёл _quarto.yml в текущем дереве. "
        "Запускай из корня курса."
    )


def load_meta(root: Path, hw_id: str) -> dict:
    meta_path = root / "homework" / hw_id / "meta.yml"
    if not meta_path.exists():
        raise SystemExit(f"meta.yml не найден: {meta_path}")
    return yaml.safe_load(meta_path.read_text())


def load_rubric(root: Path, hw_id: str) -> dict:
    meta = load_meta(root, hw_id)
    rubric_name = meta.get("judge", {}).get("rubric", "rubric.yml")
    rubric_path = root / "homework" / hw_id / rubric_name
    if not rubric_path.exists():
        raise SystemExit(f"rubric.yml не найден: {rubric_path}")
    return yaml.safe_load(rubric_path.read_text())
