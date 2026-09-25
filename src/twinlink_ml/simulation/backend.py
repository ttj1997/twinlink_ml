"""Array-Backend-Abstraktion für das Grey-Box-Prozessmodell.

Warum diese Abstraktion existiert
---------------------------------
Das Prozessmodell hat zwei Verwendungen mit unterschiedlichen Anforderungen:

1. **Datengenerierung** für den Architekturvergleich (AP 1): braucht nur schnelle,
   batchweise Auswertung. NumPy genügt.
2. **RL-Environment / Gradientenprüfung** (ab Monat 5-6): profitiert von
   Differenzierbarkeit (analytische Gradienten, Verifikation der Monotonie-Constraints
   gegen die Ground Truth, differenzierbare MPC).

Statt das Modell zweimal zu schreiben, wird jede Operation über ein schmales Protokoll
gekapselt. Die Physik-Module kennen nur dieses Protokoll (Dependency Injection), nicht
NumPy oder PyTorch. Die untestbare Fläche bleibt damit auf die ~20 Zeilen des jeweiligen
Backends beschränkt.

Konvention: Arrays haben die Form ``(batch, ...)``. Ein einzelner Lauf ist ``batch=1``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# Laufzeit-Typ eines Feldes: np.ndarray oder torch.Tensor.
# Absichtlich `Any`: ein echtes Protokoll für alle Arithmetik-Dunder brächte hier
# keinen Erkenntnisgewinn, sondern nur Rauschen. Die Verträge stehen in den
# Docstrings, die Einheiten in den Variablennamen.
Array = Any

FloatArray = NDArray[np.float64]


@runtime_checkable
class ArrayBackend(Protocol):
    """Minimale Menge elementweiser Operationen, die das Prozessmodell benötigt."""

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
    """Primäres, vollständig getestetes Backend."""

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
    """Backend für differenzierbare Rollouts und GPU-Batches.

    Achtung: in der Entwicklungsumgebung ohne PyTorch nicht ausführbar und daher
    **nicht durch die Testsuite gedeckt**. Vor der ersten Verwendung
    ``pytest tests/ -k torch`` mit installiertem PyTorch laufen lassen; die
    Physiktests sind backend-parametrisiert und decken dann auch diesen Pfad ab.
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
    """Erzeuge das Backend, damit Configs nur einen String enthalten müssen."""
    backends: dict[str, ArrayBackend] = {
        "numpy": NumpyBackend(),
        "torch": TorchBackend(),
    }
    if name not in backends:
        raise ValueError(f"Unbekanntes Backend {name!r}, erlaubt: {sorted(backends)}")
    return backends[name]
