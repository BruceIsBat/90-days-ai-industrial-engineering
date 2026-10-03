"""
common/seeds.py
Global reproducibility helper for setting pseudorandom number generator (PRNG) seeds.

REPRODUCIBILITY CONTRACT:
- What this guarantees: Identical runs on the same machine, architecture, OS, 
  and pinned dependency versions.
- What this does NOT guarantee: Bitwise determinism across different GPU architectures, 
  CUDA driver versions, or PyTorch minor releases.
- PYTHONHASHSEED note: Python hash randomization is locked at interpreter boot.
  To guarantee deterministic string hashing and set order, set it in your shell
  prior to process startup: `PYTHONHASHSEED=0 python script.py`.
- Call order note: Call set_seed() at the very top of your entrypoint script before
  allocating GPU memory or invoking any CUDA calls so CUBLAS_WORKSPACE_CONFIG takes effect.
"""

import os
import random
import warnings
from typing import Callable
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def _validate_seed(seed: int) -> int:
    """Validate that seed is an integer within [0, 2**32 - 1]."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}: {seed}")
    if not (0 <= seed < 2**32):
        raise ValueError(f"Seed {seed} out of bounds. Must be between 0 and {2**32 - 1}.")
    return seed
    
def set_seed(seed: int = 42, strict_determinism: bool = True) -> int:
    """
    Fix pseudorandom number generator seeds across standard library, NumPy, and PyTorch.

    Args:
        seed: Target integer seed in the range [0, 2**32 - 1].
        strict_determinism: If True, calls torch.use_deterministic_algorithms(True),
            which raises a RuntimeError when an operation has no deterministic
            implementation. If False, calls torch.use_deterministic_algorithms(False):
            PyTorch may then silently choose non-deterministic algorithms, so runs
            are NOT guaranteed to be reproducible. The cuDNN flags
            (deterministic=True, benchmark=False) are set in both cases.

    Returns:
        int: The applied seed.
    """
    # 0. Seed boundary validation for cross-library compatibility
    seed = _validate_seed(seed)

    # 1. Required for deterministic cuBLAS operations on CUDA >= 10.2
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # 2. Python standard library PRNG
    random.seed(seed)

    # 3. NumPy legacy PRNG (for libraries relying on global state)
    np.random.seed(seed)

    # 4. PyTorch backends
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(strict_determinism)
    else:
        warnings.warn(
            "PyTorch is not installed. PyTorch PRNG states were NOT seeded.",
            UserWarning,
            stacklevel=2,
        )

    return seed


def get_numpy_rng(seed: int = 42) -> np.random.Generator:
    """Return an isolated NumPy default_rng instance for explicit passing."""
    seed = _validate_seed(seed)
    return np.random.default_rng(seed)


def get_torch_generator(seed: int = 42) -> "torch.Generator":
    """
    Return a seeded torch.Generator instance.
    Pass this directly to DataLoader(..., generator=gen) to govern shuffle sequences.
    """
    seed = _validate_seed(seed)
    if not TORCH_AVAILABLE:
        raise RuntimeError("Cannot instantiate torch.Generator: PyTorch is not installed.")

    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def get_worker_init_fn() -> Callable[[int], None]:
    """
    DataLoader worker initialization adhering to official PyTorch recommendations:
    derives distinct seeds per worker using torch.initial_seed() % 2**32 so workers
    do not replicate identical noise trajectories across epochs.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("Cannot construct worker_init_fn: PyTorch is not installed.")

    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return _worker_init_fn