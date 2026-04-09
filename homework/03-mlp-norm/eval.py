import json
import sys
from pathlib import Path

import numpy as np


def main():
    truth_path = Path("data/hidden/test.npz")
    pred_path = Path("submission/predictions.npy")

    if not pred_path.exists():
        print(json.dumps({"error": "predictions.npy not found"}))
        sys.exit(1)
        
    if not truth_path.exists():
        # Fallback for local testing if hidden test is missing
        truth_labels = np.zeros(100) # Dummy
    else:
        truth = np.load(truth_path)
        truth_labels = truth["labels"]

    preds = np.load(pred_path)

    if len(preds) != len(truth_labels):
        print(json.dumps({"error": f"Length mismatch: {len(preds)} != {len(truth_labels)}"}))
        sys.exit(1)

    acc = (preds == truth_labels).mean()

    result = {
        "accuracy": float(acc)
    }
    
    # Write to standard output
    print(json.dumps(result))


if __name__ == "__main__":
    main()
