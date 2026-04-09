import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate(y_true_path, y_pred_path):
    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)
    return roc_auc_score(y_true, y_pred)
