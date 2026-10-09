"""
tests/test_metrics.py
Unit tests for prognostics evaluation metrics: RMSE and NASA Score.
"""

import numpy as np
import pytest
from common.metrics import compute_rmse, compute_nasa_score


def test_rmse_known_values():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 20.0, 26.0])
    # Differences: [2, 0, -4] -> Squared: [4, 0, 16] -> Mean: 20/3 -> sqrt(20/3) ≈ 2.581988897
    expected = np.sqrt(20.0 / 3.0)
    assert pytest.approx(compute_rmse(y_true, y_pred), 1e-6) == expected


def test_nasa_asymmetric_scoring():
    y_true = np.array([100.0, 100.0])
    # Early prediction by 10 cycles (d = -10): exp(-(-10)/13) - 1 = exp(10/13) - 1 ≈ 1.158
    # Late prediction by 10 cycles (d = +10):  exp(10/10) - 1 = exp(1) - 1 ≈ 1.718
    # Total penalty must penalize late prediction more than early prediction
    y_early = np.array([90.0])
    y_late = np.array([110.0])
    s_early = compute_nasa_score(np.array([100.0]), y_early)
    s_late = compute_nasa_score(np.array([100.0]), y_late)

    assert s_late > s_early
    assert pytest.approx(s_early, 1e-3) == (np.exp(10.0 / 13.0) - 1.0)
    assert pytest.approx(s_late, 1e-3) == (np.exp(1.0) - 1.0)
