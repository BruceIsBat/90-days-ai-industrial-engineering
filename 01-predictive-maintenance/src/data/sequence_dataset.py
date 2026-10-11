"""
01-predictive-maintenance/src/data/sequence_dataset.py
PyTorch Dataset and sequence builder for C-MAPSS FD001.
Implements sliding windows with trajectory boundary isolation,
train-only normalization scaling, and left-padding for short sequences.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from src.data.features import INFORMATIVE_SENSORS


class ScalerState:
    """Stores feature-wise min and max fitted strictly on training engines."""
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.mins: Optional[np.ndarray] = None
        self.maxs: Optional[np.ndarray] = None

    def fit(self, df: pl.DataFrame) -> "ScalerState":
        features = df.select(self.feature_cols).to_numpy()
        self.mins = np.min(features, axis=0)
        self.maxs = np.max(features, axis=0)
        # Avoid division by zero for any edge-case constant feature
        range_span = self.maxs - self.mins
        range_span[range_span == 0.0] = 1.0
        self._range_span = range_span
        return self

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        if self.mins is None or self.maxs is None:
            raise RuntimeError("ScalerState must be fitted before transforming.")
        features = df.select(self.feature_cols).to_numpy()
        return (features - self.mins) / self._range_span


class CMAPSSSequenceDataset(Dataset):
    """
    PyTorch Dataset providing sliding sequence windows for turbofan RUL prediction.
    Outputs:
        x: FloatTensor of shape (window_size, num_features)
        y: FloatTensor scalar (RUL target) or None for unlabelled test
        mask: BoolTensor of shape (window_size,) indicating valid (non-padded) cycles
    """
    def __init__(
        self,
        windows: np.ndarray,
        targets: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
    ):
        self.windows = torch.as_tensor(windows, dtype=torch.float32)
        self.targets = (
            torch.as_tensor(targets, dtype=torch.float32)
            if targets is not None
            else None
        )
        self.masks = (
            torch.as_tensor(masks, dtype=torch.bool)
            if masks is not None
            else torch.ones((len(windows), windows.shape[1]), dtype=torch.bool)
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "x": self.windows[idx],
            "mask": self.masks[idx],
        }
        if self.targets is not None:
            sample["y"] = self.targets[idx]
        return sample


def build_trajectory_windows(
    df: pl.DataFrame,
    scaler: ScalerState,
    window_size: int = 30,
    is_test: bool = False,
    test_rul: Optional[np.ndarray] = None,
    max_rul: float = 125.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs sliding windows while strictly preventing cross-trajectory bleeding.
    If is_test is True:
        Extracts only the terminal window ending at cycle T_j per engine.
        Applies left-padding if T_j < window_size.
    """
    feature_cols = scaler.feature_cols
    windows_list: List[np.ndarray] = []
    targets_list: List[float] = []
    masks_list: List[np.ndarray] = []

    unique_units = df["unit_number"].unique().sort().to_list()

    for idx, unit_id in enumerate(unique_units):
        unit_df = df.filter(pl.col("unit_number") == unit_id).sort("time_cycles")
        n_cycles = len(unit_df)
        unit_features = scaler.transform(unit_df)

        if not is_test:
            # Training/Validation: use capped piecewise RUL from dataframe
            unit_targets = unit_df["rul"].to_numpy().astype(np.float64)

            # Slide window from window_size up to total lifetime
            for end_idx in range(window_size, n_cycles + 1):
                start_idx = end_idx - window_size
                window = unit_features[start_idx:end_idx]
                target = unit_targets[end_idx - 1]
                mask = np.ones(window_size, dtype=bool)

                windows_list.append(window)
                targets_list.append(target)
                masks_list.append(mask)

        else:
            # Test: terminal window evaluation only
            if test_rul is None:
                raise ValueError("test_rul must be provided when is_test=True")

            capped_target = min(float(test_rul[idx]), max_rul)

            if n_cycles >= window_size:
                # Standard terminal slice
                window = unit_features[-window_size:]
                mask = np.ones(window_size, dtype=bool)
            else:
                # Left-padding for engines with lifetime shorter than window_size
                pad_len = window_size - n_cycles
                padding = np.zeros((pad_len, unit_features.shape[1]), dtype=np.float64)
                window = np.vstack([padding, unit_features])
                mask = np.concatenate([np.zeros(pad_len, dtype=bool), np.ones(n_cycles, dtype=bool)])

            windows_list.append(window)
            targets_list.append(capped_target)
            masks_list.append(mask)

    return (
        np.array(windows_list, dtype=np.float32),
        np.array(targets_list, dtype=np.float32),
        np.array(masks_list, dtype=bool),
    )


def prepare_sequence_datasets(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    raw_test_rul: np.ndarray,
    train_units: np.ndarray,
    val_units: np.ndarray,
    window_size: int = 30,
    max_rul: float = 125.0,
) -> Tuple[CMAPSSSequenceDataset, CMAPSSSequenceDataset, CMAPSSSequenceDataset, ScalerState]:
    """
    Prepares train, validation, and test PyTorch datasets with strict leakage isolation.
    """
    feature_cols = [c for c in INFORMATIVE_SENSORS if c in train_df.columns]

    # Split train dataframe into trajectory-isolated folds
    train_fold_df = train_df.filter(pl.col("unit_number").is_in(train_units))
    val_fold_df = train_df.filter(pl.col("unit_number").is_in(val_units))

    # Fit scaler ONLY on the training fold engines
    scaler = ScalerState(feature_cols=feature_cols).fit(train_fold_df)

    # 1. Build training windows
    x_train, y_train, m_train = build_trajectory_windows(
        train_fold_df, scaler, window_size=window_size, is_test=False, max_rul=max_rul
    )
    train_ds = CMAPSSSequenceDataset(x_train, y_train, m_train)

    # 2. Build validation windows
    x_val, y_val, m_val = build_trajectory_windows(
        val_fold_df, scaler, window_size=window_size, is_test=False, max_rul=max_rul
    )
    val_ds = CMAPSSSequenceDataset(x_val, y_val, m_val)

    # 3. Build terminal test windows
    x_test, y_test, m_test = build_trajectory_windows(
        test_df, scaler, window_size=window_size, is_test=True, test_rul=raw_test_rul, max_rul=max_rul
    )
    test_ds = CMAPSSSequenceDataset(x_test, y_test, m_test)

    return train_ds, val_ds, test_ds, scaler