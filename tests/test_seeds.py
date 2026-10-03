"""
tests/test_seeds.py
Verification suite for PRNG determinism and contract adherence.
"""

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


# 1. Seed boundary & type validation tests
def test_seed_bounds_and_type_validation():
    with pytest.raises(ValueError, match="out of bounds"):
        set_seed(-1)
    with pytest.raises(ValueError, match="out of bounds"):
        set_seed(2**32)
    with pytest.raises(TypeError, match="Seed must be an integer"):
        set_seed(3.5)
    with pytest.raises(TypeError, match="Seed must be an integer"):
        set_seed(True)
    with pytest.raises(ValueError, match="out of bounds"):
        get_numpy_rng(-1)

    # Valid boundary checks (must succeed without raising)
    rng_min = get_numpy_rng(0)
    assert isinstance(rng_min, np.random.Generator)

    rng_max = get_numpy_rng(2**32 - 1)
    assert isinstance(rng_max, np.random.Generator)


# 2. Python standard library random test
def test_python_random_reproducibility():
    set_seed(42)
    s1 = [random.random() for _ in range(5)]

    set_seed(42)
    s2 = [random.random() for _ in range(5)]

    set_seed(99)
    s3 = [random.random() for _ in range(5)]

    assert s1 == s2
    assert s1 != s3


# 3. NumPy legacy PRNG test
def test_numpy_legacy_seed_reproducibility():
    set_seed(42)
    sample_1 = np.random.rand(5)

    set_seed(42)
    sample_2 = np.random.rand(5)

    set_seed(99)
    sample_diff = np.random.rand(5)

    np.testing.assert_array_equal(sample_1, sample_2)
    assert not np.array_equal(sample_1, sample_diff)


# 4. NumPy default_rng generator test
def test_numpy_generator_reproducibility():
    rng1 = get_numpy_rng(42)
    arr1 = rng1.standard_normal(5)

    rng2 = get_numpy_rng(42)
    arr2 = rng2.standard_normal(5)

    rng3 = get_numpy_rng(99)
    arr3 = rng3.standard_normal(5)

    np.testing.assert_array_equal(arr1, arr2)
    assert not np.array_equal(arr1, arr3)


# 5. PyTorch core seeding tests
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_torch_seed_reproducibility():
    set_seed(42)
    t1 = torch.randn(4, 4)

    set_seed(42)
    t2 = torch.randn(4, 4)

    set_seed(99)
    t_diff = torch.randn(4, 4)

    assert torch.equal(t1, t2)
    assert not torch.equal(t1, t_diff)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_torch_generator_reproducibility():
    g1 = get_torch_generator(42)
    t1 = torch.randn(5, generator=g1)

    g2 = get_torch_generator(42)
    t2 = torch.randn(5, generator=g2)

    assert torch.equal(t1, t2)


# 6. Strict determinism check
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_torch_strict_determinism_flag():
    set_seed(42, strict_determinism=True)
    assert torch.are_deterministic_algorithms_enabled() is True

    set_seed(42, strict_determinism=False)
    assert torch.are_deterministic_algorithms_enabled() is False


# 7. Worker init guard
def test_worker_init_fn_guard(monkeypatch):
    monkeypatch.setattr(seeds_module, "TORCH_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="PyTorch is not installed"):
        seeds_module.get_worker_init_fn()


# 8. Warning when PyTorch is not available
def test_torch_missing_warning(monkeypatch):
    monkeypatch.setattr(seeds_module, "TORCH_AVAILABLE", False)
    with pytest.warns(UserWarning, match="PyTorch is not installed"):
        seeds_module.set_seed(42)
