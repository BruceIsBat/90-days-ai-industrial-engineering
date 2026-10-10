"""
01-predictive-maintenance/experiments/eda_fd001.py
Day 8 Task: Exploratory Data Analysis for NASA C-MAPSS FD001.
Outputs:
1. Lifetime histogram & sensor trajectory plots -> 01-predictive-maintenance/figures/fd001_eda_summary.png
2. Sensor variance verification table        -> 01-predictive-maintenance/figures/sensor_variance_table.md
"""

from pathlib import Path
import sys
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

# Ensure repo and project roots are discoverable
project_root = Path(__file__).resolve().parents[1]
repo_root = Path(__file__).resolve().parents[2]
for p in (str(project_root), str(repo_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data.cmapss_loader import load_cmapss_subset
from src.data.features import INFORMATIVE_SENSORS

FIGURES_DIR = project_root / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_eda():
    data_dir = repo_root / "data" / "raw" / "cmapss"
    train_df, test_df, _ = load_cmapss_subset(data_dir, subset="FD001", max_rul=125.0)

    # 1. Lifetime distribution
    lifetime_df = (
        train_df.group_by("unit_number")
        .agg(pl.col("time_cycles").max().alias("max_cycle"))
        .sort("unit_number")
    )
    lifetimes = lifetime_df["max_cycle"].to_numpy()

    # 2. Sensor variance check across all 21 sensors
    all_sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    variance_records = []
    
    for s in all_sensor_cols:
        std_val = float(train_df[s].std())
        is_flat = std_val < 1e-4
        in_informative = s in INFORMATIVE_SENSORS
        variance_records.append({
            "sensor": s,
            "std": std_val,
            "is_flat": is_flat,
            "in_hardcoded_list": in_informative,
            "consistent": (not is_flat) == in_informative,
        })

    # Save verification table
    table_lines = [
        "# NASA C-MAPSS FD001 Sensor Variance Analysis",
        "",
        "| Sensor | Standard Deviation | Is Flat (Zero Variance) | In INFORMATIVE_SENSORS | Verification |",
        "|---|---|---|---|---|",
    ]
    for r in variance_records:
        status = "MATCH" if r["consistent"] else "MISMATCH"
        table_lines.append(
            f"| `{r['sensor']}` | {r['std']:.6f} | `{r['is_flat']}` | `{r['in_hardcoded_list']}` | **{status}** |"
        )

    table_path = FIGURES_DIR / "sensor_variance_table.md"
    table_path.write_text("\n".join(table_lines) + "\n")
    print(f"Variance table written to {table_path}")

    # 3. Generate summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NASA C-MAPSS FD001 Exploratory Data Analysis", fontsize=16)

    # Panel A: Run-to-failure lifetime histogram
    axes[0, 0].hist(lifetimes, bins=20, color="#1f77b4", edgecolor="black", alpha=0.8)
    axes[0, 0].axvline(np.median(lifetimes), color="red", linestyle="--", label=f"Median ({np.median(lifetimes):.0f})")
    axes[0, 0].set_title("Engine Lifetimes Distribution (N=100)")
    axes[0, 0].set_xlabel("Cycles to Failure")
    axes[0, 0].set_ylabel("Engine Count")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel B: Flat sensor demonstration (Sensor 1 - Fan Inlet Temp)
    for unit_id in [1, 2, 3]:
        sub = train_df.filter(pl.col("unit_number") == unit_id)
        axes[0, 1].plot(sub["time_cycles"], sub["sensor_1"], label=f"Engine {unit_id}")
    axes[0, 1].set_title("Flat Sensor Example: Sensor 1 (Std = 0.000)")
    axes[0, 1].set_xlabel("Operating Cycles")
    axes[0, 1].set_ylabel("Measurement")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # Panel C: Degrading sensor trajectory (Sensor 2 - LPC Outlet Temp)
    for unit_id in [1, 2, 3]:
        sub = train_df.filter(pl.col("unit_number") == unit_id)
        axes[1, 0].plot(sub["time_cycles"], sub["sensor_2"], label=f"Engine {unit_id}")
    axes[1, 0].set_title("Degrading Sensor: Sensor 2 (LPC Outlet Temp)")
    axes[1, 0].set_xlabel("Operating Cycles")
    axes[1, 0].set_ylabel("Measurement")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel D: Degrading sensor trajectory (Sensor 4 - LPT Outlet Temp)
    for unit_id in [1, 2, 3]:
        sub = train_df.filter(pl.col("unit_number") == unit_id)
        axes[1, 1].plot(sub["time_cycles"], sub["sensor_4"], label=f"Engine {unit_id}")
    axes[1, 1].set_title("Degrading Sensor: Sensor 4 (LPT Outlet Temp)")
    axes[1, 1].set_xlabel("Operating Cycles")
    axes[1, 1].set_ylabel("Measurement")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = FIGURES_DIR / "fd001_eda_summary.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Summary figure written to {plot_path}")


if __name__ == "__main__":
    run_eda()