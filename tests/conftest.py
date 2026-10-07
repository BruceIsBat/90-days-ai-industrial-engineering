"""
tests/conftest.py
Global pytest fixtures and safety nets.
"""

from pathlib import Path
import random
import numpy as np
import pytest

import common.seeds as seeds_module
import common.run_logger as run_logger
from common.seeds import (
    set_seed,
    get_numpy_rng,
    get_torch_generator,
    get_worker_init_fn,
    TORCH_AVAILABLE,
)

if TORCH_AVAILABLE:
    import torch


@pytest.fixture(autouse=True)
def restore_torch_determinism_state():
    """Snapshot and restore PyTorch deterministic mode to prevent state leakage."""
    if TORCH_AVAILABLE:
        prev_state = torch.are_deterministic_algorithms_enabled()
        yield
        torch.use_deterministic_algorithms(prev_state)
    else:
        yield


@pytest.fixture(autouse=True)
def isolate_default_results_csv(tmp_path: Path, monkeypatch):
    """
    Safety net: redirect DEFAULT_RESULTS_FILE to a temporary directory for all tests
    so a test without an explicit filepath never corrupts the real results.csv.
    """
    safe_target = tmp_path / "sandbox_results.csv"
    monkeypatch.setattr(run_logger, "DEFAULT_RESULTS_FILE", safe_target)


@pytest.fixture(autouse=True)
def reset_determinism_state():
    """
    Capture both determinism state and warn_only mode prior to test execution
    and restore both precisely to prevent cross-test leakage.
    """
    prev_enabled = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    yield

    # Restore the full original 2-tuple state
    torch.use_deterministic_algorithms(prev_enabled, warn_only=prev_warn_only)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()