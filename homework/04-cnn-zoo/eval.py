"""Leaderboard evaluator for homework 04-cnn-zoo.

Запускается грейдером:
    python homework/04-cnn-zoo/eval.py \
        --submission path/to/submission \
        --hidden data/hidden/04-cnn-zoo/test.npz \
        --out result.json

Скрытый тест — npz с массивами `images` (N, 3, 64, 64) uint8 и
`labels` (N,) int64. eval.py НЕ должен видеть `labels` до момента сравнения
и НЕ должен исполнять студенческий код напрямую — он только сравнивает
predictions.npy с labels.

Это намеренно. Грейдер для judge-режима отдельно исполняет ноутбук,
leaderboard-режим же максимально дешёв и не требует GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(submission: Path, hidden: Path) -> dict:
    pred_path = submission / "submission" / "predictions.npy"
    if not pred_path.exists():
        return {
            "ok": False,
            "error": f"missing {pred_path}",
            "metric": "accuracy",
            "score": 0.0,
        }

    preds = np.load(pred_path)
    if not hidden.exists():
        return {
            "ok": False,
            "error": f"hidden test not mounted: {hidden}",
            "metric": "accuracy",
            "score": 0.0,
        }
    test = np.load(hidden)
    labels = test["labels"]

    if preds.shape != labels.shape:
        return {
            "ok": False,
            "error": f"shape mismatch: preds {preds.shape} vs labels {labels.shape}",
            "metric": "accuracy",
            "score": 0.0,
        }

    accuracy = float((preds == labels).mean())
    return {
        "ok": True,
        "metric": "accuracy",
        "score": accuracy,
        "n": int(labels.shape[0]),
    }


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
