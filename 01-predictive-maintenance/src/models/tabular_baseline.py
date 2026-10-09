"""
src/models/tabular_baseline.py
Tuned LightGBM regression baseline for C-MAPSS RUL estimation.
"""

import sys
from pathlib import Path

# Ensure 01-predictive-maintenance and repo root are discoverable
project_root = Path(__file__).resolve().parents[2]
repo_root = Path(__file__).resolve().parents[3]
for p in (str(project_root), str(repo_root)):
    if p not in sys.path:
        sys.path.insert(0, p)


import numpy as np
import polars as pl
import lightgbm as lgb
import optuna

from common.splits import split_trajectories_by_unit
from common.metrics import compute_rmse, compute_nasa_score
from common.run_logger import log_experiment
from common.seeds import set_seed
from src.data.cmapss_loader import load_cmapss_subset
from src.data.features import add_rolling_features

optuna.logging.set_verbosity(optuna.logging.WARNING)


def extract_features_and_target(df: pl.DataFrame, is_train: bool = True):
    feature_cols = [c for c in df.columns if c not in ("unit_number", "time_cycles", "rul")]
    X = df.select(feature_cols).to_numpy()
    y = df.select("rul").to_numpy().flatten() if is_train else None
    return X, y, feature_cols


def run_baseline(
    data_dir: Path,
    subset: str = "FD001",
    seed: int = 42,
    n_trials: int = 50,
):
    set_seed(seed)

    # 1. Load data
    train_df, test_df, true_test_rul = load_cmapss_subset(data_dir, subset=subset)

    # 2. Trajectory Split (80 Train Units / 20 Val Units)
    unique_units = train_df["unit_number"].unique().sort().to_numpy()
    train_units, val_units = split_trajectories_by_unit(unique_units, val_ratio=0.2, seed=seed)

    # 3. Add rolling features
    train_full_feat = add_rolling_features(train_df)
    test_feat = add_rolling_features(test_df)

    # Slice train and validation sets
    train_split_df = train_full_feat.filter(pl.col("unit_number").is_in(train_units))
    val_split_df = train_full_feat.filter(pl.col("unit_number").is_in(val_units))

    X_train, y_train, feature_names = extract_features_and_target(train_split_df, is_train=True)
    X_val, y_val, _ = extract_features_and_target(val_split_df, is_train=True)

    # Test set: evaluated strictly at the terminal cycle of each engine
    test_last_df = test_feat.group_by("unit_number").last().sort("unit_number")
    X_test, _, _ = extract_features_and_target(test_last_df, is_train=False)

    # 4. Optuna Study (Strict budget: n_trials)
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": seed,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

        model = lgb.LGBMRegressor(**params, n_estimators=300)
        model.fit(
            X_train, y_train,
            eval_X=X_val,
            eval_y=y_val,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        preds_val = model.predict(X_val)
        return compute_rmse(y_val, preds_val)

    

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params.update({"objective": "regression", "metric": "rmse", "verbosity": -1, "random_state": seed})

    # 5. Fit best model on train and evaluate on Test
    final_model = lgb.LGBMRegressor(**best_params, n_estimators=500)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    test_preds = final_model.predict(X_test)
    test_rmse = compute_rmse(true_test_rul, test_preds)
    test_nasa_score = compute_nasa_score(true_test_rul, test_preds)

    print(f"\n--- [FD001 LightGBM Baseline Evaluation] ---")
    print(f"Test RMSE:       {test_rmse:.4f}")
    print(f"NASA Score:      {test_nasa_score:.2f}")
    print(f"Best Params:     {study.best_params}")

    # 6. Log to results.csv ledger
    log_experiment(
        project="01-predictive-maintenance",
        phase="Phase 1",
        model_name="lightgbm_baseline",
        dataset=f"C-MAPSS_{subset}",
        split="test",
        seed=seed,
        metric_name="RMSE",
        metric_value=test_rmse,
        device="cpu",
        config=study.best_params,
        notes=f"NASA_Score={test_nasa_score:.2f}; 50_trials_optuna_early_stopping",
    )


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    default_dir = repo_root / "data" / "raw" / "cmapss"
    run_baseline(data_dir=default_dir, subset="FD001", seed=42, n_trials=50)
