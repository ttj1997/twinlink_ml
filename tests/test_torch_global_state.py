"""Proves that the autouse fixture ``_restore_torch_global_state`` contains the leak.

Torch dtype leak from ``test_greybox_numerics.py``, fixture in conftest.py. Order
is part of the test here, not just chance: pytest collects test functions
within a file in definition order (no ordering plugin needed). The second
function below therefore runs guaranteed directly after the first -- exactly the order
under which the leak was originally reproducible (one test sets globally, the next
sees the state that was not undone).
"""

from __future__ import annotations

import importlib.util

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_torch_default_dtype_leak_reproduced() -> None:
    """Mimics the leak behaviour: sets the default dtype, does NOT reset it.

    Deliberately no try/finally here -- exactly its absence is the reproduced
    error from the two affected gradient tests. Whether the fixture nevertheless
    contains it is checked by the next function in this file.
    """
    import torch

    torch.set_default_dtype(torch.float64)
    assert torch.get_default_dtype() == torch.float64  # leak actually active


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch nicht installiert")
def test_torch_default_dtype_restored_after_leaking_test() -> None:
    """Must stand immediately AFTER ``test_torch_default_dtype_leak_reproduced`` (see above).

    Runs guaranteed directly after the leaking function above (definition order).
    If this test fails, either the autouse fixture in conftest.py is broken
    or the two functions were accidentally reordered.
    """
    import torch

    assert torch.get_default_dtype() == torch.float32
