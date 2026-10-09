"""
tests/test_tabular_baseline.py
Verification tests for tabular feature extraction and pipeline structures.
"""

import polars as pl
import numpy as np
import pytest

from src.data.features import add_rolling_features, INFORMATIVE_SENSORS


def test_add_rolling_features_partition_isolation():
    # Construct synthetic multi-engine time-series
    data = {
        "unit_number": [1, 1, 1, 2, 2, 2],
        "time_cycles": [1, 2, 3, 1, 2, 3],
    }
    for s in INFORMATIVE_SENSORS:
        data[s] = [10.0, 20.0, 30.0, 100.0, 200.0, 300.0]

    df = pl.DataFrame(data)
    df_feat = add_rolling_features(df, window_sizes=[2], sensor_subset=["sensor_2"])

    assert "sensor_2_roll_mean_2" in df_feat.columns
    assert "sensor_2_roll_std_2" in df_feat.columns

    # Check that unit 2 cycle 1 rolling mean does NOT incorporate unit 1 values
    u2_c1_mean = df_feat.filter(
        (pl.col("unit_number") == 2) & (pl.col("time_cycles") == 1)
    )["sensor_2_roll_mean_2"][0]
    assert u2_c1_mean == 100.0
