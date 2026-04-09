import argparse
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, type=str)
    parser.add_argument("--hidden", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    args = parser.parse_args()

    try:
        preds = np.load(args.submission)
        labels_data = np.load(args.hidden)
        # Предполагаем, что истинные лейблы хранятся под ключом 'labels'
        labels = labels_data['labels']
        
        if preds.shape != labels.shape:
            raise ValueError(f"Shape mismatch: {preds.shape} vs {labels.shape}")
            
        accuracy = float((preds == labels).mean())
        score = accuracy
        ok = True
        n_samples = len(labels)
    except Exception as e:
        accuracy = 0.0
        score = 0.0
        ok = False
        n_samples = 0
        print(f"Error during evaluation: {e}")

    result = {
        "ok": ok,
        "score": score,
        "metric": "accuracy",
        "n_samples": n_samples
    }

    with open(args.out, "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
