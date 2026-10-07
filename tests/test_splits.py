"""
tests/test_splits.py
Unit tests for trajectory-based splitting and piecewise-linear RUL target generation.
"""

import numpy as np
import pytest

from common.splits import (
    split_trajectories_by_unit,
    get_split_masks,
    compute_rul_piecewise_linear,
)


# ===========================================================================
# Tests for split_trajectories_by_unit
# ===========================================================================

def test_split_disjointness_and_coverage():
    """1. Train and validation unit sets are disjoint, and their union equals all unique units."""
    units = [1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10]
    train, val = split_trajectories_by_unit(units, val_ratio=0.2, seed=42)

    assert set(train).isdisjoint(set(val))
    assert set(train).union(set(val)) == set(np.unique(units))
    assert len(train) > 0
    assert len(val) > 0


def test_split_determinism_and_shuffle_invariance():
    """2. Same seed gives same split, different seed differs, shuffled inputs give identical output."""
    units = list(range(1, 101))
    t1, v1 = split_trajectories_by_unit(units, val_ratio=0.2, seed=42)
    t2, v2 = split_trajectories_by_unit(units, val_ratio=0.2, seed=42)
    t3, v3 = split_trajectories_by_unit(units, val_ratio=0.2, seed=99)

    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(v1, v2)
    assert not np.array_equal(t1, t3)

    shuffled_units = np.random.default_rng(7).permutation(units)
    t_shuf, v_shuf = split_trajectories_by_unit(shuffled_units, val_ratio=0.2, seed=42)
    np.testing.assert_array_equal(t1, t_shuf)
    np.testing.assert_array_equal(v1, v_shuf)


def test_split_partition_sizes():
    """3. 100 units at 0.2 gives 20 val units. 2 units gives 1 and 1."""
    units_100 = list(range(100))
    _, v100 = split_trajectories_by_unit(units_100, val_ratio=0.2, seed=42)
    assert len(v100) == 20

    units_2 = ["A", "B"]
    t2, v2 = split_trajectories_by_unit(units_2, val_ratio=0.5, seed=42)
    assert len(t2) == 1 and len(v2) == 1


@pytest.mark.parametrize(
    "bad_ratio,err",
    [
        (0.0, ValueError),
        (1.0, ValueError),
        (-0.2, ValueError),
        (1.2, ValueError),
        (True, TypeError),
        ("0.2", TypeError),
    ],
)
def test_split_bad_ratio_raises(bad_ratio, err):
    """4a. val_ratio boundaries and invalid types raise."""
    with pytest.raises(err):
        split_trajectories_by_unit([1, 2, 3], val_ratio=bad_ratio)


def test_split_insufficient_units_raises():
    """4b. Fewer than 2 unique units must raise ValueError."""
    with pytest.raises(ValueError, match="Need at least 2 unique units"):
        split_trajectories_by_unit([1, 1, 1], val_ratio=0.2)


def test_window_level_leakage_prevented():
    """
    5. Window-level check: build simulated windows, map through the split,
    and assert that no unit_id appears on both sides.
    """
    # 20 engines, 50 cycles each -> 1000 observations
    engine_ids = np.repeat(np.arange(1, 21), repeats=50)
    train_units, val_units = split_trajectories_by_unit(engine_ids, val_ratio=0.2, seed=42)

    train_mask, val_mask = get_split_masks(engine_ids, train_units, val_units)

    # Train and val rows must cover all data and share zero engine IDs
    assert np.all(train_mask | val_mask)
    assert not np.any(train_mask & val_mask)

    train_engines = np.unique(engine_ids[train_mask])
    val_engines = np.unique(engine_ids[val_mask])
    assert set(train_engines).isdisjoint(set(val_engines))


# ===========================================================================
# Tests for compute_rul_piecewise_linear
# ===========================================================================

def test_rul_hand_calculated_values():
    """1. cycles=[1,2,3,4,5], max_rul=3 gives [3, 3, 2, 1, 0]."""
    cycles = [1, 2, 3, 4, 5]
    rul = compute_rul_piecewise_linear(cycles, max_rul=3.0)
    expected = np.array([3.0, 3.0, 2.0, 1.0, 0.0])
    np.testing.assert_allclose(rul, expected)


def test_rul_cap_larger_than_run_and_single_cycle():
    """2. A cap larger than run length leaves raw values. Single cycle gives [0.0]."""
    cycles = [1, 2, 3]
    rul = compute_rul_piecewise_linear(cycles, max_rul=100.0)
    np.testing.assert_allclose(rul, [2.0, 1.0, 0.0])

    single = compute_rul_piecewise_linear([1], max_rul=50.0)
    np.testing.assert_allclose(single, [0.0])


def test_rul_unsorted_cycles_consistent():
    """3. Unsorted cycles yield consistent RUL values per cycle."""
    cycles = [4, 1, 3, 2]
    rul = compute_rul_piecewise_linear(cycles, max_rul=2.0)
    #max_c = 4 -> raw: [0, 3, 1, 2] -> capped at 2.0: [0, 2, 1, 2]
    np.testing.assert_allclose(rul, [0.0, 2.0, 1.0, 2.0])


def test_rul_censored_test_engine_support():
    """Censored engines: true_rul_at_end adds residual life to all cycles."""
    cycles = [1, 2, 3]  # Truncated at cycle 3, with 10 cycles remaining
    rul = compute_rul_piecewise_linear(cycles, max_rul=125.0, true_rul_at_end=10)
    # Cycle 3 -> 10; Cycle 2 -> 11; Cycle 1 -> 12
    np.testing.assert_allclose(rul, [12.0, 11.0, 10.0])


@pytest.mark.parametrize(
    "bad_cap,err",
    [
        (0.0, ValueError),
        (-10.0, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("125", TypeError),
    ],
)
def test_rul_invalid_max_rul_raises(bad_cap, err):
    """4a. Invalid max_rul values raise."""
    with pytest.raises(err):
        compute_rul_piecewise_linear([1, 2, 3], max_rul=bad_cap)


def test_rul_invalid_cycles_raise():
    """4b. Empty, 2D, duplicate, float, and non-positive cycles all raise."""
    with pytest.raises(ValueError, match="non-empty 1D array"):
        compute_rul_piecewise_linear([])

    with pytest.raises(ValueError, match="non-empty 1D array"):
        compute_rul_piecewise_linear([[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="Duplicate cycle"):
        compute_rul_piecewise_linear([1, 2, 2, 3])

    with pytest.raises(TypeError, match="integer cycle"):
        compute_rul_piecewise_linear([1.0, 2.5, 3.0])

    with pytest.raises(ValueError, match="strictly positive"):
        compute_rul_piecewise_linear([0, 1, 2])