"""
experiments/demo_leakage.py
Empirical Demonstration: Naive random row splitting vs. Trajectory unit splitting.
Trains LightGBM on both splits to measure empirical score corruption.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "01-predictive-maintenance"))

import numpy as np
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

from src.data.cmapss_loader import load_cmapss_subset
from src.data.features import add_rolling_features
from src.models.tabular_baseline import extract_features_and_target
from common.splits import split_trajectories_by_unit


def run_leakage_experiment():
    print("=== Empirical Temporal Data Leakage Demonstration ===\n")
    data_dir = REPO_ROOT / "data/raw/cmapss"

    # Load FD001 training subset with piecewise capped RUL at 125 cycles
    train_df, _, _ = load_cmapss_subset(subset="FD001", data_dir=data_dir, max_rul=125.0)

    # Generate rolling window features using the exact baseline pipeline
    df_feat = add_rolling_features(train_df)
    X, y, _ = extract_features_and_target(df_feat, is_train=True)
    units = df_feat["unit_number"].to_numpy()

    # ---------------------------------------------------------
    # Split A: Naive Random Row Split (LEAKED)
    # ---------------------------------------------------------
    rng = np.random.default_rng(42)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split_idx = int(0.8 * len(X))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    model_a = lgb.LGBMRegressor(random_state=42, verbose=-1, n_estimators=100)
    model_a.fit(X[train_idx], y[train_idx])
    preds_a = model_a.predict(X[val_idx])
    rmse_a = root_mean_squared_error(y[val_idx], preds_a)

    leaked_units = np.intersect1d(units[train_idx], units[val_idx])

    # ---------------------------------------------------------
    # Split B: Trajectory Boundary Split (RIGOROUS)
    # ---------------------------------------------------------
    train_units, val_units = split_trajectories_by_unit(units, val_ratio=0.2, seed=42)
    train_mask = np.isin(units, train_units)
    val_mask = np.isin(units, val_units)

    model_b = lgb.LGBMRegressor(random_state=42, verbose=-1, n_estimators=100)
    model_b.fit(X[train_mask], y[train_mask])
    preds_b = model_b.predict(X[val_mask])
    rmse_b = root_mean_squared_error(y[val_mask], preds_b)

    print(f"[Split A - Naive Random Row Split]")
    print(f"  Overlapping Units: {len(leaked_units)} / {len(np.unique(units))}")
    print(f"  Validation RMSE:   {rmse_a:.4f} cycles\n")

    print(f"[Split B - Trajectory Grouped Split]")
    print(f"  Overlapping Units: 0 / {len(np.unique(units))}")
    print(f"  Validation RMSE:   {rmse_b:.4f} cycles\n")

    diff = rmse_b - rmse_a
    print(f"Leakage Distortion: Split A masks {diff:.4f} cycles of true error.")


if __name__ == "__main__":
    run_leakage_experiment()