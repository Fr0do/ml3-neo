"""Leaderboard runner.

Запускает homework/<hw>/eval.py на скрытом тесте, получает JSON, сохраняет
запись в grader/state/leaderboards.json. Никакой LLM не задействован.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path

STATE_FILE = "grader/state/leaderboards.json"


def run_leaderboard(
    root: Path,
    hw_id: str,
    meta: dict,
    submission: Path,
    student: str,
    *,
    hidden_override: Path | None = None,
) -> dict:
    """Запускает eval.py на скрытом тесте.

    Если передан `hidden_override` — используется он (персональный инстанс
    студента). Иначе — путь из meta.yml. При per-student instances глобальный
    hidden_data может отсутствовать, и это нормально.
    """
    eval_py = root / "homework" / hw_id / "eval.py"
    if not eval_py.exists():
        return {"ok": False, "error": f"missing {eval_py}"}

    if hidden_override is not None:
        hidden = hidden_override
    else:
        hidden_rel = meta.get("leaderboard", {}).get("hidden_data")
        if not hidden_rel:
            return {"ok": False, "error": "meta.yml: leaderboard.hidden_data not set"}
        hidden = root / hidden_rel

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        out_path = Path(tmp.name)

    proc = subprocess.run(
        [
            sys.executable,
            str(eval_py),
            "--submission", str(submission),
            "--hidden", str(hidden),
            "--out", str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": "eval.py failed",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    result = json.loads(out_path.read_text())
    out_path.unlink(missing_ok=True)

    if not result.get("ok"):
        return result

    _append_to_state(root, hw_id, meta, student, result)
    return result


def _append_to_state(
    root: Path, hw_id: str, meta: dict, student: str, result: dict
) -> None:
    state_path = root / STATE_FILE
    state_path.parent.mkdir(parents=True, exist_ok=True)

    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())

    board = state.setdefault(
        hw_id,
        {
            "metric": meta["leaderboard"]["metric"],
            "goal": meta["leaderboard"]["goal"],
            "submissions": [],
        },
    )

    board["submissions"] = [
        s for s in board["submissions"] if s["student"] != student
    ]
    board["submissions"].append(
        {
            "student": student,
            "score": result["score"],
            "ts": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    )

    state_path.write_text(json.dumps(state, indent=2))
