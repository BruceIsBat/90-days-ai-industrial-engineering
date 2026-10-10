"""
common/metrics.py
Prognostics performance metrics: RMSE and NASA Asymmetric Scoring Function.
Strict input validation prevents silent broadcasting, shape mismatch, and NaN poisoning.
"""

from typing import Union
import numpy as np


def _validate_inputs(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> tuple[np.ndarray, np.ndarray]:
    """Validate shape, type, emptiness, and numerical sanity."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if yt.size == 0 or yp.size == 0:
        raise ValueError("Input arrays must not be empty.")

    # Flatten 1D/2D single-column arrays to avoid silent (n, 1) vs (n,) broadcasting
    if yt.ndim > 2 or yp.ndim > 2:
        raise ValueError(f"Inputs must be 1-D or 2-D single column. Got shapes: {yt.shape}, {yp.shape}")

    yt = yt.ravel()
    yp = yp.ravel()

    if yt.shape != yp.shape:
        raise ValueError(f"Array length mismatch: y_true has {len(yt)} items, y_pred has {len(yp)} items.")

    if not np.all(np.isfinite(yt)) or not np.all(np.isfinite(yp)):
        raise ValueError("Inputs contain NaN or Infinite values.")

    return yt, yp


def compute_rmse(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """Compute Root Mean Squared Error with shape guards."""
    yt, yp = _validate_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def compute_nasa_score(y_true: Union[np.ndarray, list], y_pred: Union[np.ndarray, list]) -> float:
    """
    NASA Asymmetric Scoring Function (Saxena et al., 2008 / PHM08).
    
    d_i = y_pred_i - y_true_i
    Early prediction (d_i < 0):   s_i = exp(-d_i / 13) - 1
    Late prediction  (d_i >= 0):  s_i = exp(d_i / 10) - 1
    """
    yt, yp = _validate_inputs(y_true, y_pred)
    d = yp - yt
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))