"""
experiments/demo_leakage.py
Empirical Demonstration: Naive random row splitting vs. Trajectory unit splitting.
"""

import numpy as np
from common.splits import split_trajectories_by_unit

def run_leakage_demo():
    print("=== Temporal Data Leakage Demonstration ===")
    num_units = 10
    cycles_per_unit = 100
    unit_ids = np.repeat(np.arange(1, num_units + 1), cycles_per_unit)

    # Split A: Naive Random Row Split (Flawed)
    rng = np.random.default_rng(42)
    indices = np.arange(len(unit_ids))
    rng.shuffle(indices)
    split_idx = int(0.8 * len(unit_ids))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    leaked_units = np.intersect1d(unit_ids[train_idx], unit_ids[val_idx])
    print(f"\n[Split A: Naive Row Split]")
    print(f"Total Unique Units: {num_units}")
    print(f"Leaked Units Appearing in BOTH Train & Val: {len(leaked_units)} / {num_units}")
    print("Flaw: Future temporal states of validation units leak into training windows.")

    # Split B: Trajectory Boundary Split (Rigorous)
    train_units, val_units = split_trajectories_by_unit(unit_ids, val_ratio=0.2, seed=42)
    intersection = np.intersect1d(train_units, val_units)
    print(f"\n[Split B: Trajectory Boundary Split]")
    print(f"Train Units: {train_units.tolist()}")
    print(f"Val Units:   {val_units.tolist()}")
    print(f"Overlapping Unit Count: {len(intersection)}")
    assert len(intersection) == 0, "Leakage occurred in Split B!"
    print("Verdict: Strict physical separation; zero temporal contamination.\n")

if __name__ == "__main__":
    run_leakage_demo()
