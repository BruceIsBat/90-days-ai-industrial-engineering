"""
src/data/features.py
Feature engineering: rolling statistics partitioned by trajectory.
"""

from typing import Sequence
import polars as pl

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
SETTINGS_COLS = [f"op_setting_{i}" for i in range(1, 4)]

# In FD001, these sensors are constant/flatline and provide zero variance
INFORMATIVE_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]


def add_rolling_features(
    df: pl.DataFrame,
    window_sizes: Sequence[int] = (5, 10, 20),
    sensor_subset: Sequence[str] = INFORMATIVE_SENSORS,
) -> pl.DataFrame:
    """
    Compute rolling mean and rolling std for sensors within each unit trajectory.
    Guarantees no cross-engine leakage by partitioning `.over("unit_number")`.
    """
    expressions = []
    for w in window_sizes:
        for col in sensor_subset:
            expressions.extend([
                pl.col(col)
                .rolling_mean(window_size=w, min_samples=1)
                .over("unit_number")
                .alias(f"{col}_roll_mean_{w}"),
                pl.col(col)
                .rolling_std(window_size=w, min_samples=1)
                .over("unit_number")
                .fill_null(0.0)
                .alias(f"{col}_roll_std_{w}"),
            ])

    return df.with_columns(expressions)
