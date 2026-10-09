"""
common/metrics.py
Prognostics performance metrics: RMSE and NASA Asymmetric Scoring Function.
"""

import numpy as np


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def compute_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA Asymmetric Scoring Function (Saxena et al., 2008).
    d_i = y_pred_i - y_true_i
    Penalizes late predictions (d_i >= 0) more severely than early predictions (d_i < 0).
    """
    d = np.asarray(y_pred) - np.asarray(y_true)
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))