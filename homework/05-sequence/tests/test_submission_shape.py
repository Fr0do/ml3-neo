"""Tests for submission format."""

import numpy as np
from pathlib import Path
import json

def test_submission_shape():
    pred_path = Path("homework/05-sequence/submission/predictions.npy")
    if not pred_path.exists():
        return  # test skips if no file, useful for local dev before submitting
        
    preds = np.load(pred_path)
    
    # Check shape constraints
    assert len(preds.shape) == 1, f"Predictions must be 1D, got {preds.shape}"
    assert preds.dtype in [np.int32, np.int64], f"Predictions must be integers, got {preds.dtype}"
    
    # Values should be non-negative class indices
    assert np.all(preds >= 0), "Class indices must be non-negative"
