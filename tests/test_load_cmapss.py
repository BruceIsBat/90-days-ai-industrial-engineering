"""
tests/test_load_cmapss.py
Verification tests for C-MAPSS raw ingestion and label structures.
"""

from pathlib import Path
import numpy as np
import polars as pl
import pytest

from src.data.cmapss_loader import load_cmapss_subset

DATA_DIR = Path("data/raw/cmapss")


@pytest.mark.skipif(not (DATA_DIR / "train_FD001.txt").exists(), reason="C-MAPSS data not downloaded.")
def test_cmapss_fd001_shapes_and_rul():
    train_df, test_df, true_test_rul = load_cmapss_subset(DATA_DIR, subset="FD001", max_rul=125.0)

    # 100 units in FD001
    assert train_df["unit_number"].n_unique() == 100
    assert test_df["unit_number"].n_unique() == 100
    assert len(true_test_rul) == 100

    # Ensure piecewise ceiling is obeyed
    assert train_df["rul"].max() <= 125.0
    assert train_df["rul"].min() == 0.0

    # No extra ghost columns
    assert "column_27" not in train_df.columns
    assert "column_28" not in train_df.columns

    # Final cycle of every training engine has RUL == 0
    final_cycles_rul = (
        train_df.group_by("unit_number")
        .agg(pl.col("rul").last())
        .select("rul")
        .to_numpy()
        .flatten()
    )
    np.testing.assert_array_equal(final_cycles_rul, np.zeros(100))
