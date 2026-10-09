"""
src/data/cmapss_loader.py
Ingestion and parsing for NASA C-MAPSS turbofan degradation datasets.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import polars as pl

from common.splits import compute_rul_piecewise_linear

# Standard 26-column schema defined by NASA Ames PHM benchmark
RAW_COLUMNS = [
    "unit_number",
    "time_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]

SCHEMA_OVERRIDES = {
    "unit_number": pl.Int64,
    "time_cycles": pl.Int64,
    **{f"op_setting_{i}": pl.Float64 for i in range(1, 4)},
    **{f"sensor_{i}": pl.Float64 for i in range(1, 22)},
}


def load_cmapss_subset(
    data_dir: Path,
    subset: str = "FD001",
    max_rul: float = 125.0,
) -> Tuple[pl.DataFrame, pl.DataFrame, np.ndarray]:
    """
    Load train, test, and true test RUL for a specific C-MAPSS subset.

    Returns:
        train_df: Run-to-failure trajectories with piecewise linear 'rul' column.
        test_df: Censored operational trajectories.
        true_test_rul: 1D array of ground truth RUL values for test engines at cutoff.
    """
    train_path = data_dir / f"train_{subset}.txt"
    test_path = data_dir / f"test_{subset}.txt"
    rul_path = data_dir / f"RUL_{subset}.txt"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train dataset: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test dataset: {test_path}")
    if not rul_path.exists():
        raise FileNotFoundError(f"Missing RUL reference: {rul_path}")

    # Read space-delimited text files into Polars
    train_df = pl.read_csv(
        train_path,
        separator=" ",
        has_header=False,
        new_columns=RAW_COLUMNS,
        schema_overrides=SCHEMA_OVERRIDES,
        truncate_ragged_lines=True,
    )

    test_df = pl.read_csv(
        test_path,
        separator=" ",
        has_header=False,
        new_columns=RAW_COLUMNS,
        schema_overrides=SCHEMA_OVERRIDES,
        truncate_ragged_lines=True,
    )

    true_test_rul = np.loadtxt(rul_path, dtype=np.int64)
# Read text files; ignore extra trailing whitespace columns
    train_raw = pl.read_csv(
        train_path,
        separator=" ",
        has_header=False,
        truncate_ragged_lines=True,
    )
    # Retain strictly the 26 true columns (col 0 to 25)
    train_df = train_raw.select(train_raw.columns[:26])
    train_df.columns = RAW_COLUMNS
    train_df = train_df.cast(SCHEMA_OVERRIDES)

    test_raw = pl.read_csv(
        test_path,
        separator=" ",
        has_header=False,
        truncate_ragged_lines=True,
    )
    test_df = test_raw.select(test_raw.columns[:26])
    test_df.columns = RAW_COLUMNS
    test_df = test_df.cast(SCHEMA_OVERRIDES)
    # Compute RUL for run-to-failure train trajectories
    # Compute per engine unit to avoid cross-engine corruption
    rul_series_list = []
    for unit_id in train_df["unit_number"].unique().sort():
        unit_cycles = train_df.filter(pl.col("unit_number") == unit_id)["time_cycles"].to_numpy()
        unit_rul = compute_rul_piecewise_linear(unit_cycles, max_rul=max_rul, true_rul_at_end=None)
        rul_series_list.extend(unit_rul)

    train_df = train_df.sort(["unit_number", "time_cycles"]).with_columns(
        pl.Series("rul", rul_series_list, dtype=pl.Float64)
    )

    return train_df, test_df, true_test_rul


if __name__ == "__main__":
    # Point directly to the verified location
    data_directory = Path("data/raw/cmapss")
    train, test, true_rul = load_cmapss_subset(data_directory, subset="FD001")
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("True RUL first 5:", true_rul[:5])
    print(train.head(3))