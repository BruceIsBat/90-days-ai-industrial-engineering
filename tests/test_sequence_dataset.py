"""
tests/test_sequence_dataset.py
Verifies sliding window isolation, normalization isolation, and padding in SequenceDataset.
"""

import numpy as np
import polars as pl
import pytest
import torch

from src.data.sequence_dataset import (
    ScalerState,
    CMAPSSSequenceDataset,
    build_trajectory_windows,
    prepare_sequence_datasets,
)
from src.data.features import INFORMATIVE_SENSORS


@pytest.fixture
def dummy_train_df():
    # 2 engines, Engine 1 runs 40 cycles, Engine 2 runs 35 cycles
    rows = []
    for unit, length in [(1, 40), (2, 35)]:
        for c in range(1, length + 1):
            row = {"unit_number": unit, "time_cycles": c, "rul": max(0.0, 125.0 - c)}
            for s in INFORMATIVE_SENSORS:
                row[s] = float(c * 2.0)
            rows.append(row)
    return pl.DataFrame(rows)


def test_window_trajectory_isolation(dummy_train_df):
    feature_cols = [c for c in INFORMATIVE_SENSORS if c in dummy_train_df.columns]
    scaler = ScalerState(feature_cols=feature_cols).fit(dummy_train_df)
    
    window_size = 30
    windows, targets, masks = build_trajectory_windows(
        dummy_train_df, scaler, window_size=window_size, is_test=False
    )

    # Engine 1 (40 cycles) -> 40 - 30 + 1 = 11 windows
    # Engine 2 (35 cycles) -> 35 - 30 + 1 = 6 windows
    # Total expected windows = 17
    assert len(windows) == 17
    assert windows.shape == (17, 30, len(feature_cols))
    assert masks.all()


def test_short_test_sequence_padding():
    # Test engine with only 15 cycles (shorter than window size of 30)
    rows = []
    for c in range(1, 16):
        row = {"unit_number": 1, "time_cycles": c}
        for s in INFORMATIVE_SENSORS:
            row[s] = 100.0 + c
        rows.append(row)
    test_df = pl.DataFrame(rows)

    feature_cols = [c for c in INFORMATIVE_SENSORS if c in test_df.columns]
    scaler = ScalerState(feature_cols=feature_cols).fit(test_df)

    test_rul = np.array([50.0])
    windows, targets, masks = build_trajectory_windows(
        test_df, scaler, window_size=30, is_test=True, test_rul=test_rul
    )

    assert len(windows) == 1
    assert windows.shape == (1, 30, len(feature_cols))
    
    # First 15 steps must be zero-padded, last 15 steps valid
    assert not masks[0, :15].any()
    assert masks[0, 15:].all()
    assert np.all(windows[0, :15, :] == 0.0)
    assert targets[0] == 50.0


def test_scaler_leakage_isolation(dummy_train_df):
    train_units = np.array([1])
    val_units = np.array([2])
    test_rul = np.array([25.0])
    
    # Engine 1 has max feature value 80.0, Engine 2 goes up to 70.0
    train_ds, val_ds, test_ds, scaler = prepare_sequence_datasets(
        train_df=dummy_train_df,
        test_df=dummy_train_df.filter(pl.col("unit_number") == 1),
        raw_test_rul=test_rul,
        train_units=train_units,
        val_units=val_units,
        window_size=30,
    )

    # Scaler max must match Engine 1 exactly, untouched by Engine 2
    expected_max = 40 * 2.0
    assert np.allclose(scaler.maxs, expected_max)