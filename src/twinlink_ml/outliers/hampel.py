r"""Causal Hampel filter: an online-implementable outlier filter for one measured channel.

At step :math:`k` the window holds the last ``window`` RAW values including the current one,
:math:`x[k-w+1], \dots, x[k]` (fewer at the start of a run). With the window median :math:`m`
and :math:`\mathrm{MAD} = \operatorname{median}|x - m|`, the value is replaced by :math:`m` if

.. math:: |x[k] - m| > c \cdot 1.4826 \cdot \mathrm{MAD}.

1.4826 makes the MAD a consistent estimate of the standard deviation for Gaussian data. The window
uses raw values, not already filtered ones (the standard, non-recursive Hampel filter). The
output at step ``k`` depends on ``x[:k + 1]`` only -- unlike the centred Savitzky-Golay smoothing
of the predecessor project (docs/04), this filter can run on line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..simulation.backend import FloatArray

__all__ = ["MAD_SCALE", "CausalHampelFilter", "FilterResult"]

MAD_SCALE = 1.4826


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Filtered signal and the mask of replaced samples."""

    values: FloatArray
    replaced: np.ndarray

    @property
    def replaced_fraction(self) -> float:
        """Share of replaced samples."""
        return float(self.replaced.mean()) if self.replaced.size else 0.0


@dataclass(frozen=True, slots=True)
class CausalHampelFilter:
    """Causal Hampel filter on signals of shape ``(runs, time)``."""

    window: int = 8
    threshold: float = 3.0

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.threshold <= 0.0:
            raise ValueError("threshold must be > 0")

    def apply(self, signal: FloatArray) -> FilterResult:
        """Filter every run; ``signal`` has the shape ``(runs, time)``."""
        values = np.asarray(signal, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("signal must have the shape (runs, time)")
        out = values.copy()
        replaced = np.zeros(values.shape, dtype=bool)
        for k in range(values.shape[1]):
            window = values[:, max(0, k - self.window + 1) : k + 1]
            median = np.median(window, axis=1)
            mad = np.median(np.abs(window - median[:, None]), axis=1)
            outlier = np.abs(values[:, k] - median) > self.threshold * MAD_SCALE * mad
            out[outlier, k] = median[outlier]
            replaced[:, k] = outlier
        return FilterResult(values=out, replaced=replaced)

    def describe(self) -> dict[str, float | int | str]:
        """For the run log."""
        return {"kind": "hampel_causal", "window": self.window, "threshold": self.threshold}
