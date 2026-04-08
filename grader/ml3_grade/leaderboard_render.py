"""Рендерит state/leaderboards.json в /leaderboards.qmd сайта.

Сам leaderboards.qmd в корне курса делает это инлайн через {python} ячейку
во время Quarto render — этот модуль нужен только для CLI команды
`ml3 render-leaderboards`, которую запускают вне Quarto (например, в
pre-render hook'е).
"""

from __future__ import annotations

import json
from pathlib import Path

STATE_PATH = "grader/state/leaderboards.json"
OUT_PATH = "leaderboards.qmd"


def render_state(root: Path) -> Path:
    state_path = root / STATE_PATH
    out_path = root / OUT_PATH

    if not state_path.exists():
        out_path.write_text(
            "---\ntitle: Leaderboards\n---\n\n*Пока нет сабмитов.*\n"
        )
        return out_path

    state = json.loads(state_path.read_text())
    lines = ["---", "title: Leaderboards", "---", ""]
    for hw_id, board in sorted(state.items()):
        lines.append(f"## {hw_id}\n")
        lines.append(
            f"**Метрика:** `{board['metric']}` · **Цель:** {board['goal']}\n"
        )
        lines.append("| Rank | Student | Score | Submitted |")
        lines.append("|---:|:---|---:|:---|")
        rows = sorted(
            board["submissions"],
            key=lambda r: r["score"],
            reverse=board["goal"] == "max",
        )
        for i, row in enumerate(rows, 1):
            lines.append(
                f"| {i} | {row['student']} | {row['score']:.4f} | {row['ts']} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path
