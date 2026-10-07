"""
common/splits.py
Leakage-free dataset splitting strategies for industrial run-to-failure telemetry.
"""

from __future__ import annotations

import numbers
from typing import Optional, Sequence, Tuple, Union
import numpy as np

from common.seeds import validate_seed


def split_trajectories_by_unit(
    unit_ids: Union[Sequence[Union[int, str]], np.ndarray],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split time-series data at the physical unit/entity boundary to prevent trajectory leakage.

    Guarantees:
    - Every unique unit_id appears in either train OR val, NEVER both.
    - Set intersection of train_units and val_units is strictly empty.
    - Split proportions match val_ratio as closely as discrete unit counts allow.
    - Both partitions are guaranteed at least one unit.
    - Output arrays are sorted.

    Args:
        unit_ids: Sequence or array of unit identifiers present in the dataset.
        val_ratio: Fraction of distinct units assigned to validation (0.0 < val_ratio < 1.0).
        seed: Valid PRNG seed for deterministic entity shuffling.

    Returns:
        Tuple of (train_units, val_units) as sorted NumPy arrays of unique IDs.
    """
    if isinstance(val_ratio, bool) or not isinstance(val_ratio, numbers.Real):
        raise TypeError(f"val_ratio must be a real float, got {type(val_ratio).__name__}.")
    
    val_ratio_f = float(val_ratio)
    if not (0.0 < val_ratio_f < 1.0):
        raise ValueError(f"val_ratio must be strictly between 0 and 1, got {val_ratio_f}.")

    seed_clean = validate_seed(seed)
    unique_units = np.unique(unit_ids)
    n_units = len(unique_units)

    if n_units < 2:
        raise ValueError(f"Need at least 2 unique units to split, found {n_units}.")

    n_val = int(np.round(n_units * val_ratio_f))
    # Guarantee at least 1 unit in each partition
    n_val = max(1, min(n_val, n_units - 1))

    rng = np.random.default_rng(seed_clean)
    shuffled_units = unique_units.copy()
    rng.shuffle(shuffled_units)

    val_units = np.sort(shuffled_units[:n_val])
    train_units = np.sort(shuffled_units[n_val:])

    return train_units, val_units


def get_split_masks(
    all_units: Union[Sequence[Union[int, str]], np.ndarray],
    train_units: Sequence[Union[int, str]],
    val_units: Sequence[Union[int, str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map partitioned unit sets back to boolean row masks across an entire dataset.
    Asserts zero intersection between the train and validation masks.
    """
    arr = np.asarray(all_units)
    train_mask = np.isin(arr, train_units)
    val_mask = np.isin(arr, val_units)

    if np.any(train_mask & val_mask):
        raise ValueError("Data leakage detected: unit overlap between train and val masks.")

    return train_mask, val_mask


def compute_rul_piecewise_linear(
    cycles: Union[Sequence[int], np.ndarray],
    max_rul: float = 125.0,
    true_rul_at_end: Optional[int] = None,
) -> np.ndarray:
    """
    Compute Remaining Useful Life (RUL) with a piecewise linear ceiling (early-life saturation).

    Empirical modeling heuristic (Heimes, 2008):
    In early operational cycles, a machine does not exhibit measurable degradation;
    its RUL is capped at `max_rul` until degradation onset, after which it drops linearly to 0.

    CRITICAL BOUNDARY REQUIREMENT:
      - For RUN-TO-FAILURE training engines: Leave `true_rul_at_end=None` (or 0). The final 
        observed cycle is treated as the failure point (RUL = 0).
      - For CENSORED test engines (e.g. C-MAPSS test sets): You MUST supply `true_rul_at_end`
        from the reference ground-truth file. Calling this without `true_rul_at_end` on 
        truncated test engines assigns RUL = 0 at the cutoff cycle, which will silently 
        corrupt evaluation.

    Args:
        cycles: 1D sequence of operational cycle counts for a single unit.
        max_rul: Upper saturation ceiling for RUL (finite positive scalar).
        true_rul_at_end: Ground-truth RUL remaining at the final cycle (for censored test runs).

    Returns:
        1D float64 array of RUL values matching cycles.
    """
    # 1. Validate max_rul
    if isinstance(max_rul, bool) or not isinstance(max_rul, numbers.Real):
        raise TypeError(f"max_rul must be a positive real scalar, got: {type(max_rul).__name__}")
    
    max_rul_f = float(max_rul)
    if np.isnan(max_rul_f) or np.isinf(max_rul_f) or max_rul_f <= 0.0:
        raise ValueError(f"max_rul must be a finite, strictly positive number, got: {max_rul}")

    # 2. Validate cycles shape and content
    cycles_arr = np.asarray(cycles)
    if cycles_arr.ndim != 1 or cycles_arr.size == 0:
        raise ValueError("cycles must be a non-empty 1D array.")

    # Check for non-integers without silent float truncation
    if not np.issubdtype(cycles_arr.dtype, np.integer):
        if not np.all(np.equal(np.mod(cycles_arr, 1), 0)):
            raise TypeError("cycles must contain integer cycle counters.")
        cycles_arr = cycles_arr.astype(np.int64)

    if np.any(cycles_arr <= 0):
        raise ValueError("Operating cycles must be strictly positive (1-indexed).")

    if len(cycles_arr) != len(np.unique(cycles_arr)):
        raise ValueError("Duplicate cycle values detected within single unit trajectory.")

    # 3. Validate true_rul_at_end
    base_residual = 0
    if true_rul_at_end is not None:
        if isinstance(true_rul_at_end, bool) or not isinstance(true_rul_at_end, numbers.Integral):
            raise TypeError(f"true_rul_at_end must be an integer, got {type(true_rul_at_end).__name__}")
        if true_rul_at_end < 0:
            raise ValueError(f"true_rul_at_end cannot be negative, got {true_rul_at_end}")
        base_residual = int(true_rul_at_end)

    max_c = np.max(cycles_arr)
    raw_rul = (max_c - cycles_arr).astype(np.float64) + base_residual
    return np.minimum(raw_rul, max_rul_f)