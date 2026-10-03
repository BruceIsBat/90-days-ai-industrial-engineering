@pytest.fixture(autouse=True)
def restore_torch_determinism_state():
    """Snapshot and restore PyTorch deterministic mode to prevent state leakage."""
    if TORCH_AVAILABLE:
        prev_state = torch.are_deterministic_algorithms_enabled()
        yield
        torch.use_deterministic_algorithms(prev_state)
    else:
        yield