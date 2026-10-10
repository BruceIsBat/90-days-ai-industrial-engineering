"""
src/models/tabular_baseline.py
Audited LightGBM regression baseline for NASA C-MAPSS RUL estimation.
Complies with evaluation_protocol.md (Draft v1):
- Multi-seed execution (seeds 42-46) with fixed split seed (42)
- Capped primary target (RUL <= 125) with uncapped sensitivity reporting
- Individual queryable ledger rows for RMSE, RMSE_uncapped, and NASA_Score
- Prediction array export (.npy) for paired bootstrap significance tests
- Uses non-deprecated eval_X and eval_y fit parameters
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb
import optuna

# Ensure repo root and project root are on sys.path
project_root = Path(__file__).resolve().parents[2]
repo_root = Path(__file__).resolve().parents[3]
for p in (str(project_root), str(repo_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common.splits import split_trajectories_by_unit
from common.metrics import compute_rmse, compute_nasa_score
from common.run_logger import log_experiment
from common.seeds import set_seed
from src.data.cmapss_loader import load_cmapss_subset
from src.data.features import add_rolling_features

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

PRED_DIR = project_root / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)


def extract_features_and_target(df: pl.DataFrame, is_train: bool = True):
    feature_cols = [c for c in df.columns if c not in ("unit_number", "time_cycles", "rul")]
    X = df.select(feature_cols).to_numpy()
    y = df.select("rul").to_numpy().flatten() if is_train else None
    return X, y, feature_cols


def prepare_tabular_splits(data_dir: Path, subset: str = "FD001", split_seed: int = 42):
    train_df, test_df, raw_test_rul = load_cmapss_subset(data_dir, subset=subset, max_rul=125.0)

    # Trajectory-level split: 80% train engines / 20% validation engines
    unique_units = train_df["unit_number"].unique().sort().to_numpy()
    train_units, val_units = split_trajectories_by_unit(unique_units, val_ratio=0.2, seed=split_seed)

    # Feature extraction isolated by trajectory
    train_feat = add_rolling_features(train_df)
    test_feat = add_rolling_features(test_df)

    train_split_df = train_feat.filter(pl.col("unit_number").is_in(train_units))
    val_split_df = train_feat.filter(pl.col("unit_number").is_in(val_units))

    X_train, y_train, feature_names = extract_features_and_target(train_split_df, is_train=True)
    X_val, y_val, _ = extract_features_and_target(val_split_df, is_train=True)

    # Test features taken strictly from the terminal cycle of each test engine
    test_terminal_df = test_feat.group_by("unit_number").last().sort("unit_number")
    X_test, _, _ = extract_features_and_target(test_terminal_df, is_train=False)

    # Ground truth targets: primary capped target (<=125.0) and uncapped raw target
    y_test_capped = np.minimum(raw_test_rul, 125.0).astype(np.float64)
    y_test_uncapped = raw_test_rul.astype(np.float64)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test_capped, y_test_uncapped), feature_names


def run_audited_baseline(
    data_dir: Path,
    subset: str = "FD001",
    split_seed: int = 42,
    seeds: list[int] = [42, 43, 44, 45, 46],
    n_trials: int = 50,
):
    (X_train, y_train), (X_val, y_val), (X_test, y_test_capped, y_test_uncapped), feature_names = (
        prepare_tabular_splits(data_dir, subset=subset, split_seed=split_seed)
    )

    all_seed_preds = []

    print(f"\n=== Running Multi-Seed LightGBM Benchmark (Seeds: {seeds}) ===")

    for seed in seeds:
        set_seed(seed)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "random_state": seed,
                "n_estimators": 500,
                "n_jobs": 1,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_X=X_val,
                eval_y=y_val,
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
            )
            val_preds = model.predict(X_val)
            return compute_rmse(y_val, val_preds)

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_params.update({
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": seed,
            "n_estimators": 500,
            "n_jobs": 1,
        })

        final_model = lgb.LGBMRegressor(**best_params)
        final_model.fit(
            X_train, y_train,
            eval_X=X_val,
            eval_y=y_val,
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        raw_test_preds = final_model.predict(X_test)
        # Enforce physical domain boundary [0, 125]
        preds_clipped = np.clip(raw_test_preds, 0.0, 125.0)
        all_seed_preds.append(preds_clipped)

        # Compute evaluated metrics
        test_rmse_capped = compute_rmse(y_test_capped, preds_clipped)
        test_rmse_uncapped = compute_rmse(y_test_uncapped, raw_test_preds)
        test_nasa_score = compute_nasa_score(y_test_capped, preds_clipped)

        # Persist test predictions for paired bootstrap significance tests
        np.save(PRED_DIR / f"lightgbm_{subset.lower()}_seed_{seed}.npy", preds_clipped)

        config_payload = {
            "best_params": study.best_params,
            "split_seed": split_seed,
            "n_features": len(feature_names),
        }

        # 1. Log Primary Metric: Capped RMSE
        log_experiment(
            project="01-predictive-maintenance",
            phase="Phase 1",
            model_name="lightgbm_baseline",
            dataset=f"C-MAPSS_{subset}",
            split="test",
            seed=seed,
            metric_name="RMSE",
            metric_value=test_rmse_capped,
            device="cpu",
            config=config_payload,
            notes=f"Primary capped metric (target<=125); 50_trials_optuna",
        )

        # 2. Log Sensitivity Metric: Uncapped RMSE
        log_experiment(
            project="01-predictive-maintenance",
            phase="Phase 1",
            model_name="lightgbm_baseline",
            dataset=f"C-MAPSS_{subset}",
            split="test",
            seed=seed,
            metric_name="RMSE_uncapped",
            metric_value=test_rmse_uncapped,
            device="cpu",
            config=config_payload,
            notes=f"Sensitivity uncapped metric against raw RUL; 50_trials_optuna",
        )

        # 3. Log Secondary Domain Metric: NASA Asymmetric Score
        log_experiment(
            project="01-predictive-maintenance",
            phase="Phase 1",
            model_name="lightgbm_baseline",
            dataset=f"C-MAPSS_{subset}",
            split="test",
            seed=seed,
            metric_name="NASA_Score",
            metric_value=test_nasa_score,
            device="cpu",
            config=config_payload,
            notes=f"NASA PHM08 asymmetric score on capped target",
        )

        print(
            f" [Seed {seed}] "
            f"Capped RMSE: {test_rmse_capped:.4f} | "
            f"Uncapped RMSE: {test_rmse_uncapped:.4f} | "
            f"NASA Score: {test_nasa_score:.2f}"
        )

    # 4. Save Ensemble Predictions & Output Ensemble Score
    ensemble_preds = np.mean(all_seed_preds, axis=0)
    np.save(PRED_DIR / f"lightgbm_{subset.lower()}_ensemble.npy", ensemble_preds)

    ens_capped = compute_rmse(y_test_capped, ensemble_preds)
    ens_uncapped = compute_rmse(y_test_uncapped, ensemble_preds)
    ens_nasa = compute_nasa_score(y_test_capped, ensemble_preds)

    print(f"\n--- [{subset} LightGBM 5-Seed Ensemble Evaluation] ---")
    print(f"Capped RMSE (Primary):       {ens_capped:.4f}")
    print(f"Uncapped RMSE (Sensitivity): {ens_uncapped:.4f}")
    print(f"NASA Score:                  {ens_nasa:.2f}")


if __name__ == "__main__":
    data_directory = repo_root / "data" / "raw" / "cmapss"
    run_audited_baseline(
        data_dir=data_directory,
        subset="FD001",
        split_seed=42,
        seeds=[42, 43, 44, 45, 46],
        n_trials=50,
    )