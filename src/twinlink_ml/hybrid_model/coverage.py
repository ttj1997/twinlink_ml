r"""Classification (b): whether an operating point occurred in a training run.

The excitation of the Befund 94 pipeline covers every channel over its full range already at
``n_train`` 16 (inlet temperature 60-120 degC, gas velocity 0.8-2.5 m/s, ambient 5-35 degC,
0.25-0.85), so a per-channel range says nothing. An operating point ``z = (T_in, u_gas, T_amb,
phi)`` counts as *inside* the training range of a model iff some sample of its training runs lies
within ``threshold`` of ``z`` in every one of the four channels, each normalised by its design range
(Chebyshev distance). Spray rate, nozzle pressure and solids are not part of the distance (the
steady problem holds them fixed). The operating point of a case is the optimum of the oracle (the
truth, margin 0): model- and margin-independent, so the classification is the same for every model
of a (draw, seed, ``n_train``) cell (paired).

Threshold 0.2 (fixed before the run, decision taken without asking): chosen on the input data
only -- master seeds 1 and 2, the 25 Befund-52 contexts at the nominal oracle points -- so that at
``n_train`` 4 about a third and at ``n_train`` 32 most of the points are inside (0.1 would leave
about ten cases at ``n_train`` 4 for the 95 % quantile of the margin).
"""

from __future__ import annotations

import numpy as np

__all__ = ["chebyshev_distance"]


def chebyshev_distance(
    points: np.ndarray, samples: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Distance of every point ``(p, d)`` to its nearest sample ``(s, d)``, range-normalised."""
    span = np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64)
    p = (np.asarray(points, dtype=np.float64) - lower) / span
    s = (np.asarray(samples, dtype=np.float64) - lower) / span
    out = np.empty(p.shape[0])
    for i, row in enumerate(p):
        out[i] = float(np.min(np.max(np.abs(s - row), axis=1)))
    return out
