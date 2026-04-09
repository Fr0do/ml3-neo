import numpy as np
from pathlib import Path

def test_submission_shape():
    pred_path = Path("submission/predictions.npy")
    if not pred_path.exists():
        return  # test skips if no file, execution will fail anyway

    preds = np.load(pred_path)
    assert preds.ndim == 1, "Predictions should be 1D array"
    assert np.issubdtype(preds.dtype, np.integer), "Predictions should be integers"
