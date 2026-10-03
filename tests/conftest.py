import random
import numpy as np
import pytest

import common.seeds as seeds_module
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