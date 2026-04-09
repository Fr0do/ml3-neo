from pathlib import Path

import numpy as np


def test_submission_shape():
    pred_path = Path("submission/predictions.npy")
    if not pred_path.exists():
        return  # test will just pass if file is missing before submission

    preds = np.load(pred_path)
    assert preds.ndim == 1, "Predictions must be 1D array"
    assert np.issubdtype(preds.dtype, np.integer), "Predictions must be integer (int64)"
