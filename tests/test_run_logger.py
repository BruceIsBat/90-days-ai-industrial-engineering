"""
tests/test_run_logger.py
Verification suite for the experiment run ledger.
"""

import csv
import json
import subprocess
from pathlib import Path
import numpy as np
import pytest

import common.run_logger as run_logger
from common.run_logger import (
    log_experiment,
    CSV_HEADERS,
    get_git_commit_hash,
)


def _valid_payload() -> dict:
    """Helper returning a standard, valid payload."""
    return {
        "project": "01-predictive-maintenance",
        "phase": "Phase 1",
        "model_name": "baseline_linear",
        "dataset": "C-MAPSS_FD001",
        "split": "val",
        "seed": 42,
        "metric_name": "RMSE",
        "metric_value": 18.4231,
        "config": {"lr": 0.001, "batch_size": 32},
        "notes": "initial valid run",
        "commit_hash": "abc1234",
    }


# Test 1: First call creates header + row; second call appends without duplicate header
def test_append_lifecycle_and_dictreader(tmp_path: Path):
    dest = tmp_path / "results.csv"
    p1 = _valid_payload()
    p1["filepath"] = dest

    log_experiment(**p1)

    p2 = _valid_payload()
    p2["filepath"] = dest
    p2["seed"] = 43
    p2["metric_value"] = 17.5
    log_experiment(**p2)

    with open(dest, mode="r", newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 2
    assert reader[0]["seed"] == "42"
    assert reader[1]["seed"] == "43"

    with open(dest, mode="r", newline="", encoding="utf-8") as f:
        raw_rows = list(csv.reader(f))
    assert len(raw_rows) == 3
    assert raw_rows[0] == CSV_HEADERS


# Test 2: Existing empty file gets header; mismatched header raises and leaves file unchanged
def test_empty_file_and_mismatched_header(tmp_path: Path):
    # Case A: Empty 0-byte file
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()
    assert empty_file.stat().st_size == 0

    p = _valid_payload()
    p["filepath"] = empty_file
    log_experiment(**p)

    with open(empty_file, mode="r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2
    assert rows[0] == CSV_HEADERS

    # Case B: Mismatched header raises and file content is left intact
    corrupt_file = tmp_path / "corrupt.csv"
    initial_content = "wrong,header,row\n1,2,3\n"
    corrupt_file.write_text(initial_content, encoding="utf-8")

    p["filepath"] = corrupt_file
    with pytest.raises(ValueError, match="Header mismatch"):
        log_experiment(**p)

    assert corrupt_file.read_text(encoding="utf-8") == initial_content


# Test 3: Precision round-trips exactly
def test_metric_precision_roundtrips(tmp_path: Path):
    dest = tmp_path / "precision.csv"
    exact_val = 0.1234567890123456
    p = _valid_payload()
    p["filepath"] = dest
    p["metric_value"] = exact_val

    log_experiment(**p)

    with open(dest, mode="r", newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert float(reader[0]["metric_value"]) == exact_val


# Test 4: Notes with commas, quotes, and newlines round-trip exactly
def test_notes_with_special_characters_roundtrip(tmp_path: Path):
    dest = tmp_path / "notes.csv"
    special_notes = 'Line 1,\nLine 2 with "quotes", commas, and tabs:\t[ok]'
    p = _valid_payload()
    p["filepath"] = dest
    p["notes"] = special_notes

    log_experiment(**p)

    with open(dest, mode="r", newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert reader[0]["notes"] == special_notes


# Test 5: Parametrize bad inputs - each raises and leaves no file behind
@pytest.mark.parametrize(
    "bad_key,bad_value,expected_error",
    [
        ("metric_value", float("nan"), ValueError),
        ("metric_value", float("inf"), ValueError),
        ("split", "invalid_split", ValueError),
        ("phase", "Phase 99", ValueError),
        ("seed", -1, ValueError),
        ("seed", 2**32, ValueError),
        ("metric_value", True, TypeError),  # bool is not a valid metric
    ],
)
def test_bad_inputs_raise_and_create_no_file(tmp_path: Path, bad_key, bad_value, expected_error):
    dest = tmp_path / f"should_not_exist_{bad_key}.csv"
    p = _valid_payload()
    p["filepath"] = dest
    p[bad_key] = bad_value

    with pytest.raises(expected_error):
        log_experiment(**p)

    assert not dest.exists(), f"File {dest} should not have been created on failed validation!"


# Test 6 & 7: The EXPECTED FAILING CASES against current unhardened code
def test_numpy_float_metric_accepted(tmp_path: Path):
    """NumPy scalars (e.g. np.float32) must be accepted as valid metric values."""
    dest = tmp_path / "numpy_metric.csv"
    p = _valid_payload()
    p["filepath"] = dest
    p["metric_value"] = np.float32(1.5)

    log_experiment(**p)

    with open(dest, mode="r", newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert float(reader[0]["metric_value"]) == 1.5


def test_phase_4_and_5_accepted(tmp_path: Path):
    """The 90-day plan spans 5 phases; Phase 4 and 5 must be accepted."""
    dest = tmp_path / "phase4.csv"
    p = _valid_payload()
    p["filepath"] = dest
    p["phase"] = "Phase 4"

    log_experiment(**p)

    with open(dest, mode="r", newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert reader[0]["phase"] == "Phase 4"