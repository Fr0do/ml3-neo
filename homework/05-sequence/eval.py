"""Leaderboard evaluator for homework 05-sequence.

Запускается грейдером:
    python homework/05-sequence/eval.py \
        --submission path/to/submission \
        --hidden data/hidden/05-sequence/test.npz \
        --out result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def evaluate(submission: Path, hidden: Path) -> dict:
    pred_path = submission / "submission" / "predictions.npy"
    if not pred_path.exists():
        return {
            "ok": False,
            "error": f"missing {pred_path}",
            "metric": "macro_f1",
            "score": 0.0,
        }

    preds = np.load(pred_path)
    if not hidden.exists():
        return {
            "ok": False,
            "error": f"hidden test not mounted: {hidden}",
            "metric": "macro_f1",
            "score": 0.0,
        }
    test = np.load(hidden)
    labels = test["labels"]

    if preds.shape != labels.shape:
        return {
            "ok": False,
            "error": f"shape mismatch: preds {preds.shape} vs labels {labels.shape}",
            "metric": "macro_f1",
            "score": 0.0,
        }

    macro_f1 = float(f1_score(labels, preds, average="macro"))
    return {
        "ok": True,
        "metric": "macro_f1",
        "score": macro_f1,
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
