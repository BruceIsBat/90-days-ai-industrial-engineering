"""
common/run_logger.py
Centralized experiment ledger for appending verified benchmark runs to results.csv.
"""

import csv
import json
import math
import numbers
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from common.seeds import _validate_seed

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_FILE = REPO_ROOT / "results.csv"

CSV_HEADERS = [
    "timestamp",
    "project",
    "phase",
    "model_name",
    "dataset",
    "split",
    "seed",
    "metric_name",
    "metric_value",
    "config_json",
    "commit_hash",
    "strict_determinism",
    "torch_version",
    "numpy_version",
    "device",
    "notes",
]

VALID_SPLITS = {"train", "val", "test"}
VALID_PHASES = {f"Phase {i}" for i in range(1, 6)}


def get_git_commit_hash() -> str:
    """
    Safely retrieve the git commit hash.
    Appends '-dirty' if uncommitted changes or untracked files are present in the working tree.
    """
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        if not commit:
            return "unknown"

        
        # Check for uncommitted changes in tracked files, excluding results.csv
        status = (
            subprocess.check_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                    "--",
                    ".",
                    ":(exclude)results.csv",
                ],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        if status:
            return f"{commit}-dirty"
        return commit
    except Exception:
        return "unknown"


def _get_environment_metadata(strict_determinism: Optional[bool] = None) -> Dict[str, str]:
    """Capture environment and framework versions for the reproducibility contract."""
    np_ver = np.__version__
    if TORCH_AVAILABLE:
        torch_ver = torch.__version__
        device = "cuda" if torch.cuda.is_available() else "cpu"
        det_mode = (
            str(strict_determinism)
            if strict_determinism is not None
            else str(torch.are_deterministic_algorithms_enabled())
        )
    else:
        torch_ver = "none"
        device = "cpu"
        det_mode = "false"

    return {
        "torch_version": torch_ver,
        "numpy_version": np_ver,
        "device": device,
        "strict_determinism": det_mode,
    }


def log_experiment(
    project: str,
    phase: str,
    model_name: str,
    dataset: str,
    split: str,
    seed: int,
    metric_name: str,
    metric_value: float,
    config: Optional[Dict[str, Any]] = None,
    notes: str = "",
    commit_hash: Optional[str] = None,
    filepath: Optional[Union[Path, str]] = None,
    strict_determinism: Optional[bool] = None,
) -> None:
    """
    Append a verified experiment run to results.csv with full precision and audit metadata.
    """
    # 1. Input string validations
    if not project or not project.strip():
        raise ValueError("project must not be empty.")
    if not model_name or not model_name.strip():
        raise ValueError("model_name must not be empty.")
    if not dataset or not dataset.strip():
        raise ValueError("dataset must not be empty.")
    if not metric_name or not metric_name.strip():
        raise ValueError("metric_name must not be empty.")

    # 2. Strict phase and split validation
    if phase not in VALID_PHASES:
        raise ValueError(f"Invalid phase '{phase}'. Must be one of {sorted(VALID_PHASES)}.")
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split '{split}'. Must be one of {sorted(VALID_SPLITS)}.")

    # 3. Seed validation via common/seeds.py contract
    seed = _validate_seed(seed)

    # 4. Strict numerical metric validation (reject bools, accept numbers.Real, reject NaN/inf)
    if isinstance(metric_value, bool):
        raise TypeError(f"metric_value cannot be a boolean, got: {metric_value}")
    if not isinstance(metric_value, numbers.Real):
        raise TypeError(
            f"metric_value must be a real scalar number, got {type(metric_value).__name__}."
        )

    val_float = float(metric_value)
    if math.isnan(val_float) or math.isinf(val_float):
        raise ValueError(f"metric_value cannot be NaN or Inf, got {metric_value}.")


    # 5. Serialization of config and metadata
    config_dict = config if config is not None else {}
    config_json = json.dumps(config_dict, sort_keys=True)
    env_meta = _get_environment_metadata(strict_determinism)
    resolved_commit = commit_hash if commit_hash is not None else get_git_commit_hash()
    timestamp = datetime.now(timezone.utc).isoformat()

    # 6. Target path resolution
    target_path = Path(filepath) if filepath is not None else DEFAULT_RESULTS_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 7. Header verification and writing
    if target_path.exists() and target_path.stat().st_size > 0:
        with open(target_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
            if existing_header != CSV_HEADERS:
                raise ValueError(
                    f"Header mismatch in {target_path}.\n"
                    f"Expected: {CSV_HEADERS}\n"
                    f"Found:    {existing_header}"
                )
        write_header = False
    else:
        write_header = True

    # 8. Row construction with full precision (repr)
    row = [
        timestamp,
        project,
        phase,
        model_name,
        dataset,
        split,
        str(seed),
        metric_name,
        repr(val_float),
        config_json,
        resolved_commit,
        env_meta["strict_determinism"],
        env_meta["torch_version"],
        env_meta["numpy_version"],
        env_meta["device"],
        notes,
    ]

    with open(target_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADERS)
        writer.writerow(row)