"""Array backend abstraction for the grey-box process model.

Why this abstraction exists
---------------------------
The process model has two uses with different requirements:

1. **Data generation** for the architecture comparison (AP 1): needs only fast,
   batched evaluation. NumPy suffices.
2. **RL environment / gradient check** (from month 5-6): benefits from
   differentiability (analytical gradients, verification of the monotonicity constraints
   against the ground truth, differentiable MPC).

Instead of writing the model twice, every operation is encapsulated behind a narrow protocol.
The physics modules know only this protocol (dependency injection), not NumPy or PyTorch.
The untestable surface thus stays confined to the ~20 lines of the respective backend.

Convention: arrays have the shape ``(batch, ...)``. A single run is ``batch=1``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# Runtime type of an array: np.ndarray or torch.Tensor.
# Deliberately `Any`: a real protocol for all arithmetic dunders would bring no
# insight here, only noise. The contracts are in the docstrings, the units in the
# variable names.
Array = Any

FloatArray = NDArray[np.float64]


@runtime_checkable
class ArrayBackend(Protocol):
    """Minimal set of elementwise operations that the process model needs."""

    name: str

    @staticmethod
    def asarray(value: Any) -> Array: ...

    @staticmethod
    def exp(x: Array) -> Array: ...

    @staticmethod
    def log(x: Array) -> Array: ...

    @staticmethod
    def sqrt(x: Array) -> Array: ...

    @staticmethod
    def clip(x: Array, lower: Array, upper: Array) -> Array: ...

    @staticmethod
    def maximum(x: Array, y: Array) -> Array: ...

    @staticmethod
    def minimum(x: Array, y: Array) -> Array: ...

    @staticmethod
    def where(condition: Array, x: Array, y: Array) -> Array: ...

    @staticmethod
    def zeros_like(x: Array) -> Array: ...

    @staticmethod
    def stack(arrays: list[Array], axis: int) -> Array: ...

    @staticmethod
    def to_numpy(x: Array) -> FloatArray: ...


class NumpyBackend:
    """Primary, fully tested backend."""

    name = "numpy"

    @staticmethod
    def asarray(value: Any) -> Array:
        return np.asarray(value, dtype=np.float64)

    @staticmethod
    def exp(x: Array) -> Array:
        return np.exp(x)

    @staticmethod
    def log(x: Array) -> Array:
        return np.log(x)

    @staticmethod
    def sqrt(x: Array) -> Array:
        return np.sqrt(x)

    @staticmethod
    def clip(x: Array, lower: Array, upper: Array) -> Array:
        return np.clip(x, lower, upper)

    @staticmethod
    def maximum(x: Array, y: Array) -> Array:
        return np.maximum(x, y)

    @staticmethod
    def minimum(x: Array, y: Array) -> Array:
        return np.minimum(x, y)

    @staticmethod
    def where(condition: Array, x: Array, y: Array) -> Array:
        return np.where(condition, x, y)

    @staticmethod
    def zeros_like(x: Array) -> Array:
        return np.zeros_like(x)

    @staticmethod
    def stack(arrays: list[Array], axis: int) -> Array:
        return np.stack(arrays, axis=axis)

    @staticmethod
    def to_numpy(x: Array) -> FloatArray:
        return np.asarray(x, dtype=np.float64)


class TorchBackend:
    """Backend for differentiable rollouts and GPU batches.

    Caution: not executable in the development environment without PyTorch and therefore
    **not covered by the test suite**. Before first use, run
    ``pytest tests/ -k torch`` with PyTorch installed; the
    physics tests are backend-parametrized and then cover this path as well.
    """

    name = "torch"

    @staticmethod
    def _torch() -> Any:
        import torch

        return torch

    @staticmethod
    def asarray(value: Any) -> Array:
        torch = TorchBackend._torch()
        if isinstance(value, torch.Tensor):
            return value.to(dtype=torch.float64)
        return torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=torch.float64)

    @staticmethod
    def exp(x: Array) -> Array:
        return TorchBackend._torch().exp(x)

    @staticmethod
    def log(x: Array) -> Array:
        return TorchBackend._torch().log(x)

    @staticmethod
    def sqrt(x: Array) -> Array:
        return TorchBackend._torch().sqrt(x)

    @staticmethod
    def clip(x: Array, lower: Array, upper: Array) -> Array:
        torch = TorchBackend._torch()
        return torch.clamp(x, min=lower, max=upper)

    @staticmethod
    def maximum(x: Array, y: Array) -> Array:
        torch = TorchBackend._torch()
        return torch.maximum(x, torch.as_tensor(y, dtype=x.dtype))

    @staticmethod
    def minimum(x: Array, y: Array) -> Array:
        torch = TorchBackend._torch()
        return torch.minimum(x, torch.as_tensor(y, dtype=x.dtype))

    @staticmethod
    def where(condition: Array, x: Array, y: Array) -> Array:
        return TorchBackend._torch().where(condition, x, y)

    @staticmethod
    def zeros_like(x: Array) -> Array:
        return TorchBackend._torch().zeros_like(x)

    @staticmethod
    def stack(arrays: list[Array], axis: int) -> Array:
        return TorchBackend._torch().stack(arrays, dim=axis)

    @staticmethod
    def to_numpy(x: Array) -> FloatArray:
        return np.asarray(x.detach().cpu().numpy(), dtype=np.float64)


def get_backend(name: str) -> ArrayBackend:
    """Create the backend so that configs need to contain only a string."""
    backends: dict[str, ArrayBackend] = {
        "numpy": NumpyBackend(),
        "torch": TorchBackend(),
    }
    if name not in backends:
        raise ValueError(f"Unbekanntes Backend {name!r}, erlaubt: {sorted(backends)}")
    return backends[name]
