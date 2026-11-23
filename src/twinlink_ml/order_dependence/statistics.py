r"""Descriptive statistics of the per-query deviations (``queries.summarise`` columns).

All shares are over the queries for which the passes compared returned a point. A query is
*worse warm* at a threshold :math:`\tau` if the warm pass needs more power than the multi-start,
:math:`(P_w - P_m) / P_m > \tau`, and *better warm* if :math:`(P_w - P_m) / P_m < -\tau` (the
multi-start missed a better optimum the warm chain happened to reach).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .settings import OrderDependenceSettings

__all__ = ["pool_units", "query_statistics"]

_COLUMNS = (
    "rel_cold",
    "rel_reference",
    "d_inlet_k",
    "d_gas_m_per_s",
    "d_inlet_orders_k",
    "d_gas_orders_m_per_s",
    "distinct_bits",
    "reference_optima",
    "reference_failed",
    "reference_spread",
)


def _array(values: Sequence[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=np.float64)


def pool_units(units: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Concatenate the per-query columns of several units (units with an error are skipped)."""
    good = [u for u in units if "error" not in u]
    if not good:
        raise ValueError("no unit without an error")
    n_orders = len(good[0]["orders"])
    out = {name: np.concatenate([_array(u[name]) for u in good]) for name in _COLUMNS}
    out["rel_warm"] = np.stack(
        [np.concatenate([_array(u["rel_warm"][k]) for u in good]) for k in range(n_orders)]
    )
    out["natural_reproduced"] = np.array([bool(u["natural_reproduced"]) for u in good])
    return out


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def query_statistics(
    pooled: Mapping[str, np.ndarray], settings: OrderDependenceSettings
) -> dict[str, Any]:
    """Shares and sizes of the deviations for one pool of queries, see the module docstring."""
    warm = pooled["rel_warm"]
    natural = warm[0]
    n = int(natural.size)
    warm_ok = np.all(np.isfinite(warm), axis=0)
    with np.errstate(invalid="ignore"):
        spread = np.nanmax(warm, axis=0) - np.nanmin(warm, axis=0)
        worst_order = np.nanmax(warm, axis=0)
    natural_ok = np.isfinite(natural)
    cold, reference = pooled["rel_cold"], pooled["rel_reference"]
    cold_ok, reference_ok = np.isfinite(cold), np.isfinite(reference)
    out: dict[str, Any] = {
        "n_queries": n,
        "n_units": int(pooled["natural_reproduced"].size),
        "natural_pass_reproduced_in_all_units": bool(np.all(pooled["natural_reproduced"])),
        "failed": {
            "warm_natural": int(np.sum(~natural_ok)),
            "warm_any_order": int(np.sum(~warm_ok)),
            "cold": int(np.sum(~cold_ok)),
            "reference_all_starts": int(np.sum(~reference_ok)),
        },
        "share_bit_distinct_between_orders": float(np.mean(pooled["distinct_bits"] > 1)),
        "by_threshold": {},
    }
    for tau in settings.thresholds:
        worse = natural_ok & (natural > tau)
        out["by_threshold"][f"{tau:g}"] = {
            "share_differing_between_orders": float(np.mean(spread[warm_ok] > tau)),
            "share_warm_natural_worse": float(np.mean(natural[natural_ok] > tau)),
            "share_warm_natural_better": float(np.mean(natural[natural_ok] < -tau)),
            "share_warm_any_order_worse": float(np.mean(worst_order[warm_ok] > tau)),
            "share_cold_single_start_worse": float(np.mean(cold[cold_ok] > tau)),
            "share_reference_better_than_multi_start": float(
                np.mean(reference[reference_ok] < -tau)
            ),
            "worse_warm_natural_relative_power": _quantiles(natural[worse]),
            "worse_warm_natural_abs_d_inlet_k": _quantiles(np.abs(pooled["d_inlet_k"][worse])),
            "worse_warm_natural_abs_d_gas_m_per_s": _quantiles(
                np.abs(pooled["d_gas_m_per_s"][worse])
            ),
        }
    optima = pooled["reference_optima"]
    several = optima >= 2
    out["local_optima"] = {
        "share_of_queries_with_several": float(np.mean(several)),
        "largest_number": int(optima.max()),
        "relative_power_spread_where_several": _quantiles(
            pooled["reference_spread"][several & np.isfinite(pooled["reference_spread"])]
        ),
    }
    out["set_points_between_orders"] = {
        "abs_d_inlet_k": _quantiles(pooled["d_inlet_orders_k"][warm_ok]),
        "abs_d_gas_m_per_s": _quantiles(pooled["d_gas_orders_m_per_s"][warm_ok]),
    }
    return out
