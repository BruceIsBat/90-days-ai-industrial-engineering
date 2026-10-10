"""
tests/test_load_cmapss.py
Verification tests for C-MAPSS raw ingestion, trajectory integrity, and label alignment.
"""

from pathlib import Path
import numpy as np
import polars as pl
import pytest

from common.splits import compute_rul_piecewise_linear
from src.data.cmapss_loader import load_cmapss_subset, _read_cmapss_file

REAL_DATA_DIR = Path("data/raw/cmapss")


def _write_mock_cmapss_files(
    tmp_path: Path,
    train_lines: list[str],
    test_lines: list[str],
    rul_lines: list[str],
):
    """Helper to write synthetic C-MAPSS text files with trailing spaces."""
    subset = "FD001"
    train_file = tmp_path / f"train_{subset}.txt"
    test_file = tmp_path / f"test_{subset}.txt"
    rul_file = tmp_path / f"RUL_{subset}.txt"

    train_file.write_text("\n".join(train_lines) + "\n")
    test_file.write_text("\n".join(test_lines) + "\n")
    rul_file.write_text("\n".join(rul_lines) + "\n")
    return tmp_path


def _make_dummy_cycle_line(unit: int, cycle: int, sensor_val: float = 0.5) -> str:
    # 26 numbers + 2 trailing spaces (NASA Ames format)
    cols = [str(unit), str(cycle), "0.0", "0.0", "100.0"] + [f"{sensor_val:.2f}"] * 21
    return " ".join(cols) + "  "


def test_synthetic_good_file_and_hand_calculated_rul(tmp_path: Path):
    # Engine 1 runs 3 cycles (1, 2, 3) -> raw RUL: [2, 1, 0]
    # Engine 2 runs 2 cycles (1, 2)    -> raw RUL: [1, 0]
    train_lines = [
        _make_dummy_cycle_line(1, 1),
        _make_dummy_cycle_line(1, 2),
        _make_dummy_cycle_line(1, 3),
        _make_dummy_cycle_line(2, 1),
        _make_dummy_cycle_line(2, 2),
    ]
    test_lines = [
        _make_dummy_cycle_line(1, 1),
        _make_dummy_cycle_line(2, 1),
    ]
    rul_lines = ["25", "18"]

    _write_mock_cmapss_files(tmp_path, train_lines, test_lines, rul_lines)
    train_df, test_df, true_rul = load_cmapss_subset(tmp_path, subset="FD001", max_rul=125.0)

    assert len(train_df.columns) == 27
    assert len(test_df.columns) == 26
    assert train_df["rul"].to_list() == [2.0, 1.0, 0.0, 1.0, 0.0]
    np.testing.assert_array_equal(true_rul, np.array([25, 18]))


def test_max_rul_cap_applied_directly_in_loader(tmp_path: Path):
    # Engine 1: raw RUL is [2, 1, 0]. With max_rul=1.0, capped RUL must be [1, 1, 0]
    # Engine 2: raw RUL is [1, 0].    With max_rul=1.0, capped RUL must be [1, 0]
    train_lines = [
        _make_dummy_cycle_line(1, 1),
        _make_dummy_cycle_line(1, 2),
        _make_dummy_cycle_line(1, 3),
        _make_dummy_cycle_line(2, 1),
        _make_dummy_cycle_line(2, 2),
    ]
    test_lines = [
        _make_dummy_cycle_line(1, 1),
        _make_dummy_cycle_line(2, 1),
    ]
    _write_mock_cmapss_files(tmp_path, train_lines, test_lines, ["25", "18"])

    train_df, _, _ = load_cmapss_subset(tmp_path, subset="FD001", max_rul=1.0)
    assert train_df["rul"].to_list() == [1.0, 1.0, 0.0, 1.0, 0.0]


@pytest.mark.parametrize("bad_max_rul,expected_err", [
    (-1.0, ValueError),
    (0.0, ValueError),
    (np.nan, ValueError),
    (np.inf, ValueError),
    (True, TypeError),
    ("125", TypeError),
])
def test_loader_rejects_invalid_max_rul(tmp_path: Path, bad_max_rul, expected_err):
    train_lines = [_make_dummy_cycle_line(1, 1)]
    test_lines = [_make_dummy_cycle_line(1, 1)]
    _write_mock_cmapss_files(tmp_path, train_lines, test_lines, ["10"])

    with pytest.raises(expected_err):
        load_cmapss_subset(tmp_path, subset="FD001", max_rul=bad_max_rul)


def test_loader_sorts_shuffled_rows(tmp_path: Path):
    # Pass cycles in reverse/scrambled order: (2, 2), (1, 3), (1, 1), (2, 1), (1, 2)
    train_lines = [
        _make_dummy_cycle_line(2, 2),
        _make_dummy_cycle_line(1, 3),
        _make_dummy_cycle_line(1, 1),
        _make_dummy_cycle_line(2, 1),
        _make_dummy_cycle_line(1, 2),
    ]
    test_lines = [_make_dummy_cycle_line(1, 1)]
    _write_mock_cmapss_files(tmp_path, train_lines, test_lines, ["10"])

    train_df, _, _ = load_cmapss_subset(tmp_path, subset="FD001", max_rul=125.0)

    # Order must be restored to (1, 1), (1, 2), (1, 3), (2, 1), (2, 2)
    assert train_df["unit_number"].to_list() == [1, 1, 1, 2, 2]
    assert train_df["time_cycles"].to_list() == [1, 2, 3, 1, 2]
    assert train_df["rul"].to_list() == [2.0, 1.0, 0.0, 1.0, 0.0]


def test_loader_raises_on_train_cycle_gap(tmp_path: Path):
    train_lines = [
        _make_dummy_cycle_line(1, 1),
        _make_dummy_cycle_line(1, 3),  # Missing cycle 2
    ]
    test_lines = [_make_dummy_cycle_line(1, 1)]
    _write_mock_cmapss_files(tmp_path, train_lines, test_lines, ["10"])

    with pytest.raises(ValueError, match="Gapped or missing cycles detected in train"):
        load_cmapss_subset(tmp_path, subset="FD001")


def test_loader_raises_on_test_cycle_gap(tmp_path: Path):
    # Train is completely clean
    train_lines = [_make_dummy_cycle_line(1, 1), _make_dummy_cycle_line(1, 2)]
    # Test unit 1 has a cycle gap: [1, 3]
    test_lines = [_make_dummy_cycle_line(1, 1), _make_dummy_cycle_line(1, 3)]
    _write_mock_cmapss_files(tmp_path, train_lines, test_lines, ["10"])

    with pytest.raises(ValueError, match="Gapped or missing cycles detected in test"):
        load_cmapss_subset(tmp_path, subset="FD001")


def test_loader_raises_on_null_value_from_ragged_short_row(tmp_path: Path):
    # Row with only 10 columns; truncate_ragged_lines fills remaining 16 with nulls
    short_line = "1 1 0.0 0.0 100.0 0.5 0.5 0.5 0.5 0.5"
    train_lines = [short_line]
    test_lines = [_make_dummy_cycle_line(1, 1)]
    _write_mock_cmapss_files(tmp_path, train_lines, test_lines, ["10"])

    with pytest.raises(ValueError, match="Null values detected in train column"):
        load_cmapss_subset(tmp_path, subset="FD001")


def test_loader_raises_on_non_finite_nan_sensor_value(tmp_path: Path):
    line1 = _make_dummy_cycle_line(1, 1)
    tokens = line1.strip().split()
    tokens[5] = "NaN"
    line2_nan = " ".join(tokens) + "  "

    train_lines = [line1, line2_nan]
    test_lines = [_make_dummy_cycle_line(1, 1)]
    _write_mock_cmapss_files(tmp_path, train_lines, test_lines, ["10"])

    with pytest.raises(ValueError, match="Non-finite values detected in train column"):
        load_cmapss_subset(tmp_path, subset="FD001")


def test_loader_raises_on_duplicate_cycles(tmp_path: Path):
    train_lines = [
        _make_dummy_cycle_line(1, 1),
        _make_dummy_cycle_line(1, 1),
    ]
    test_lines = [_make_dummy_cycle_line(1, 1)]
    _write_mock_cmapss_files(tmp_path, train_lines, test_lines, ["10"])

    with pytest.raises(ValueError, match="Duplicate"):
        load_cmapss_subset(tmp_path, subset="FD001")


def test_loader_raises_on_non_contiguous_test_engine_ids(tmp_path: Path):
    train_lines = [_make_dummy_cycle_line(1, 1)]
    test_lines = [
        _make_dummy_cycle_line(1, 1),
        _make_dummy_cycle_line(3, 1),  # Missing engine 2
    ]
    _write_mock_cmapss_files(tmp_path, train_lines, test_lines, ["10", "20"])

    with pytest.raises(ValueError, match="Test engine IDs are not contiguous 1..N"):
        load_cmapss_subset(tmp_path, subset="FD001")


def test_loader_raises_on_fewer_than_26_columns(tmp_path: Path):
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("1 1 0.5 0.5 0.5\n")
    with pytest.raises(ValueError, match="expected at least 26"):
        _read_cmapss_file(bad_file)


def test_loader_raises_on_rul_length_mismatch(tmp_path: Path):
    train_lines = [_make_dummy_cycle_line(1, 1)]
    test_lines = [_make_dummy_cycle_line(1, 1), _make_dummy_cycle_line(2, 1)]
    _write_mock_cmapss_files(tmp_path, train_lines, test_lines, ["10"])

    with pytest.raises(ValueError, match="Shape mismatch"):
        load_cmapss_subset(tmp_path, subset="FD001")


@pytest.mark.skipif(not (REAL_DATA_DIR / "train_FD001.txt").exists(), reason="Real data not present.")
def test_real_fd001_exact_counts_and_reference_rul_parity():
    train_df, test_df, true_test_rul = load_cmapss_subset(REAL_DATA_DIR, subset="FD001", max_rul=125.0)

    # Documented benchmarks
    assert train_df["unit_number"].n_unique() == 100
    assert test_df["unit_number"].n_unique() == 100
    assert len(train_df) == 20631
    assert len(test_df) == 13096
    assert len(true_test_rul) == 100

    # Verify loader expression RUL matches compute_rul_piecewise_linear engine-by-engine
    for unit_id in range(1, 101):
        unit_df = train_df.filter(pl.col("unit_number") == unit_id)
        cycles = unit_df["time_cycles"].to_numpy()
        expected_rul = compute_rul_piecewise_linear(cycles, max_rul=125.0)
        np.testing.assert_allclose(unit_df["rul"].to_numpy(), expected_rul)