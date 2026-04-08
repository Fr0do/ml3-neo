"""Шаблон eval.py для нового ДЗ.

Грейдер запускает:
    python homework/<id>/eval.py \
        --submission <path> --hidden <path> --out <json>

Модуль НЕ должен исполнять студенческий код. Только сравнивать
заранее посчитанный сабмит с hidden labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(submission: Path, hidden: Path) -> dict:
    pred_path = submission / "submission" / "predictions.npy"
    if not pred_path.exists():
        return {"ok": False, "error": f"missing {pred_path}",
                "metric": "TODO", "score": 0.0}

    if not hidden.exists():
        return {"ok": False, "error": f"hidden test not mounted: {hidden}",
                "metric": "TODO", "score": 0.0}

    preds = np.load(pred_path)
    test = np.load(hidden)
    labels = test["labels"]

    # TODO: реализовать метрику ДЗ.
    score = 0.0

    return {"ok": True, "metric": "TODO", "score": float(score),
            "n": int(labels.shape[0])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--hidden", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    result = evaluate(args.submission, args.hidden)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
