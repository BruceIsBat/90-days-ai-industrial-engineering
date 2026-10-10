"""
src/data/cmapss_loader.py
Ingestion, parsing, and data validation for NASA C-MAPSS turbofan degradation datasets.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import polars as pl

from common.splits import compute_rul_piecewise_linear

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


def _read_cmapss_file(file_path: Path) -> pl.DataFrame:
    """
    Parse space-delimited text files into Polars.
    NASA files contain trailing whitespaces producing ghost columns.
    Slices the first 26 columns and casts schema types.
    """
    raw_df = pl.read_csv(
        file_path,
        separator=" ",
        has_header=False,
        truncate_ragged_lines=True,
    )
    if len(raw_df.columns) < 26:
        raise ValueError(
            f"File '{file_path}' has {len(raw_df.columns)} columns, expected at least 26."
        )

    df = raw_df.select(raw_df.columns[:26])
    df.columns = RAW_COLUMNS
    return df.cast(SCHEMA_OVERRIDES)


def _validate_trajectory_integrity(df: pl.DataFrame, context_name: str) -> pl.DataFrame:
    """
    Validate data cleanliness, duplicate keys, contiguous cycle indexing, and sorting.
    """
    # 1. Null and finite checks across all 26 columns
    for col in df.columns:
        if df[col].is_null().any():
            raise ValueError(f"Null values detected in {context_name} column '{col}'.")
        if df[col].dtype == pl.Float64 and not np.all(np.isfinite(df[col].to_numpy())):
            raise ValueError(f"Non-finite values detected in {context_name} column '{col}'.")

    # 2. Check for duplicate (unit_number, time_cycles)
    duplicate_count = df.select(["unit_number", "time_cycles"]).is_duplicated().sum()
    if duplicate_count > 0:
        raise ValueError(f"Duplicate (unit_number, time_cycles) pairs detected in {context_name}.")

    # 3. Sort strictly by trajectory and cycle
    df_sorted = df.sort(["unit_number", "time_cycles"])

    # 4. Check that cycles per engine are strictly contiguous starting at cycle 1
    # For every unit: cycle_count == max_cycle and min_cycle == 1
    cycle_stats = (
        df_sorted.group_by("unit_number")
        .agg([
            pl.col("time_cycles").min().alias("min_cycle"),
            pl.col("time_cycles").max().alias("max_cycle"),
            pl.col("time_cycles").count().alias("n_cycles"),
        ])
    )
    invalid_starts = cycle_stats.filter(pl.col("min_cycle") != 1)
    if len(invalid_starts) > 0:
        bad_units = invalid_starts["unit_number"].to_list()
        raise ValueError(f"Engines in {context_name} do not start at cycle 1: {bad_units}")

    gapped = cycle_stats.filter(pl.col("n_cycles") != pl.col("max_cycle"))
    if len(gapped) > 0:
        bad_units = gapped["unit_number"].to_list()
        raise ValueError(f"Gapped or missing cycles detected in {context_name} for units: {bad_units}")

    return df_sorted


def load_cmapss_subset(
    data_dir: Path,
    subset: str = "FD001",
    max_rul: float = 125.0,
) -> Tuple[pl.DataFrame, pl.DataFrame, np.ndarray]:
    """
    Load train, test, and true test RUL for a specific C-MAPSS subset.

    Returns:
        train_df: Run-to-failure trajectories with piecewise linear 'rul' column.
        test_df: Censored operational trajectories sorted by (unit, cycle).
        true_test_rul: 1D array of ground truth RUL values for test engines at cutoff.
    """
    train_path = data_dir / f"train_{subset}.txt"
    test_path = data_dir / f"test_{subset}.txt"
    rul_path = data_dir / f"RUL_{subset}.txt"

    for path, desc in [(train_path, "train"), (test_path, "test"), (rul_path, "RUL reference")]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {desc} dataset: {path}")

    train_df = _read_cmapss_file(train_path)
    test_df = _read_cmapss_file(test_path)
    true_test_rul = np.loadtxt(rul_path, dtype=np.int64)

    # Validate structural trajectory integrity
    train_df = _validate_trajectory_integrity(train_df, context_name="train")
    test_df = _validate_trajectory_integrity(test_df, context_name="test")

    # Validate test unit sequence is strictly 1..N
    unique_test_units = test_df["unit_number"].unique().sort().to_numpy()
    n_test_units = len(unique_test_units)
    expected_unit_ids = np.arange(1, n_test_units + 1)
    if not np.array_equal(unique_test_units, expected_unit_ids):
        raise ValueError(
            f"Test engine IDs are not contiguous 1..N. Expected 1..{n_test_units}, got {unique_test_units.tolist()}"
        )

    if len(true_test_rul) != n_test_units:
        raise ValueError(
            f"Shape mismatch: {n_test_units} test units found, but {len(true_test_rul)} true RUL entries."
        )

    # Compute piecewise linear RUL via Polars expression
    # RUL = min(max_rul, max_cycle - cycle)
    train_df = train_df.with_columns(
        (pl.col("time_cycles").max().over("unit_number") - pl.col("time_cycles"))
        .cast(pl.Float64)
        .clip(upper_bound=max_rul)
        .alias("rul")
    )

    return train_df, test_df, true_test_rul


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    data_directory = repo_root / "data" / "raw" / "cmapss"
    train, test, true_rul = load_cmapss_subset(data_directory, subset="FD001")
    print("Ingestion verified:")
    print(f"  Train: {train.shape[0]} rows, {train['unit_number'].n_unique()} units")
    print(f"  Test:  {test.shape[0]} rows, {test['unit_number'].n_unique()} units")