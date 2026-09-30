r"""SX1 (Befund 67): resolution threshold of Befund 63 with a t quantile instead of z = 1.96.

Pure evaluation of committed records, no training. Same comparisons, same medians, same
range-based sigma estimate as ``analyze_m2_resolution_verdicts.py`` (Befund 63, part g); only the
critical value changes:

.. math::
   \Delta_{\min} = t_{\nu,\,0.975}\,\sqrt{\frac{\hat\sigma_c^2 + \hat\sigma_r^2}{n}},
   \qquad
   \nu = \frac{(a+b)^2}{a^2/\nu_c + b^2/\nu_r},\quad a=\hat\sigma_c^2/n,\; b=\hat\sigma_r^2/n

(Welch-Satterthwaite). Degrees of freedom: the estimate is ``R/d2``, not the sample standard
deviation, so ``nu_c = nu_r`` is NOT ``n - 1 = 4``. The effective degrees of freedom of the range
estimator are obtained by matching the variance of ``(R/d2)^2`` to that of a scaled chi-square,
``nu_eff = 2 E[s^2]^2 / Var[s^2]`` with ``s = R/d2``, evaluated by Monte Carlo for normal samples
of size ``n`` (fixed seed; written to the artifact). The naive variant ``nu = n - 1`` per group is
reported next to it as a sensitivity, not used for the verdict.

Every verdict of Befund 63 (24 cells GRU/Transformer/xLSTM/Mamba vs. LSTM at n_train=64, 4 cells
Transformer vs. LSTM at 200/600 epochs) is recomputed; the z-based reproduction is checked against
the stored artifact ``analyze_m2_resolution_verdicts.json`` before any change is reported.

    python scripts/analyze_sx1_resolution_t.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import provenance, require_clean_tree, write_artifact
from analyze_m2_resolution_verdicts import (
    CHANNELS,
    EPOCH600_JSONL,
    EXTENDED_JSONL,
    N_TRAIN_HEADLINE,
    PROJECTED_JSONL,
    TEST_SETS,
    _metric_key,
    _spread,
    _values,
)

from twinlink_ml.evaluation.resolution import _HARTLEY_D2
from twinlink_ml.utils.tracking import read_jsonl

Z_95 = 1.960
ALPHA_TWO_SIDED = 0.05
MC_SAMPLES = 4_000_000
MC_SEED = 20260930
STORED = Path(__file__).resolve().parents[1] / "artifacts" / "runs"
STORED_VERDICTS = STORED / "analyze_m2_resolution_verdicts.json"
CANDIDATES = ("gru", "transformer", "xlstm", "mamba")


def range_estimator_dof(n: int) -> dict[str, float]:
    """Effective degrees of freedom of ``s = R / d2(n)`` for normal samples (Monte Carlo)."""
    rng = np.random.default_rng(MC_SEED)
    samples = rng.standard_normal((MC_SAMPLES, n))
    sigma_hat = (samples.max(axis=1) - samples.min(axis=1)) / _HARTLEY_D2[n]
    variance_hat = sigma_hat**2
    mean = float(variance_hat.mean())
    variance = float(variance_hat.var(ddof=1))
    return {
        "n": n,
        "mc_samples": MC_SAMPLES,
        "seed": MC_SEED,
        "mean_of_sigma_hat_squared": mean,
        "var_of_sigma_hat_squared": variance,
        "nu_eff_range": 2.0 * mean * mean / variance,
        "nu_eff_sample_std": float(n - 1),
    }


def _threshold(
    *, sigma_c: float, sigma_r: float, n: int, nu_group: float, median_ref: float
) -> dict[str, float]:
    a = sigma_c**2 / n
    b = sigma_r**2 / n
    nu_welch = (a + b) ** 2 / (a**2 / nu_group + b**2 / nu_group) if a + b > 0 else math.inf
    t_crit = float(stats.t.ppf(1.0 - ALPHA_TWO_SIDED / 2.0, nu_welch))
    scale = math.sqrt(a + b) / abs(median_ref)
    return {"nu_welch": nu_welch, "critical_value": t_crit, "threshold_relative": t_crit * scale}


def _cell(
    *,
    candidate: str,
    candidate_values: list[float],
    reference: str,
    reference_values: list[float],
    nu_eff: float,
) -> dict[str, Any]:
    cand = _spread(candidate_values)
    ref = _spread(reference_values)
    n = len(reference_values)
    improvement = (ref.median - cand.median) / abs(ref.median)
    sigma_c, sigma_r = cand.sigma_from_span, ref.sigma_from_span
    z_threshold = Z_95 * math.sqrt((sigma_c**2 + sigma_r**2) / n) / abs(ref.median)
    primary = _threshold(
        sigma_c=sigma_c, sigma_r=sigma_r, n=n, nu_group=nu_eff, median_ref=ref.median
    )
    naive = _threshold(
        sigma_c=sigma_c, sigma_r=sigma_r, n=n, nu_group=float(n - 1), median_ref=ref.median
    )
    return {
        "candidate": candidate,
        "reference": reference,
        "n_seeds": n,
        "improvement_relative": improvement,
        "sigma_relative_candidate": sigma_c / abs(cand.median) if cand.median else math.nan,
        "sigma_relative_reference": sigma_r / abs(ref.median),
        "z": {"threshold_relative": z_threshold, "resolved": abs(improvement) > z_threshold},
        "t_range_dof": {
            **primary,
            "resolved": abs(improvement) > primary["threshold_relative"],
        },
        "t_naive_dof_n_minus_1": {
            **naive,
            "resolved": abs(improvement) > naive["threshold_relative"],
        },
    }


def main() -> None:
    require_clean_tree()
    projected = read_jsonl(PROJECTED_JSONL)
    extended = read_jsonl(EXTENDED_JSONL)
    headline = projected + extended
    epoch600 = read_jsonl(EPOCH600_JSONL)
    stored = json.loads(STORED_VERDICTS.read_text(encoding="utf-8"))

    dof = range_estimator_dof(5)
    nu_eff = dof["nu_eff_range"]
    print(f"effective dof of R/d2 at n=5: {nu_eff:.3f} (sample std: 4)")

    cells: dict[str, Any] = {}
    for channel in CHANNELS:
        for test_set in TEST_SETS:
            lstm = _values(
                headline, model="lstm", n_train=N_TRAIN_HEADLINE, test_set=test_set, channel=channel
            )
            for candidate in CANDIDATES:
                values = _values(
                    headline,
                    model=candidate,
                    n_train=N_TRAIN_HEADLINE,
                    test_set=test_set,
                    channel=channel,
                )
                cells[f"{channel}/{test_set}/{candidate}"] = _cell(
                    candidate=candidate,
                    candidate_values=values,
                    reference="lstm",
                    reference_values=lstm,
                    nu_eff=nu_eff,
                )
    for channel in ("bed_temperature_celsius", "energy_target"):
        key = _metric_key(channel)
        for epochs, cand_values, ref_values in (
            (
                200,
                _values(
                    headline,
                    model="transformer",
                    n_train=N_TRAIN_HEADLINE,
                    test_set="constant_holdout",
                    channel=channel,
                ),
                _values(
                    headline,
                    model="lstm",
                    n_train=N_TRAIN_HEADLINE,
                    test_set="constant_holdout",
                    channel=channel,
                ),
            ),
            (
                600,
                sorted(
                    r.metrics[key]
                    for r in epoch600
                    if r.tags.get("model") == "transformer"
                    and r.tags.get("test_set") == "constant_holdout"
                    and key in r.metrics
                ),
                sorted(
                    r.metrics[key]
                    for r in epoch600
                    if r.tags.get("model") == "lstm"
                    and r.tags.get("test_set") == "constant_holdout"
                    and key in r.metrics
                ),
            ),
        ):
            cells[f"{channel}/constant_holdout/transformer_{epochs}ep"] = _cell(
                candidate=f"transformer_{epochs}ep",
                candidate_values=cand_values,
                reference=f"lstm_{epochs}ep",
                reference_values=ref_values,
                nu_eff=nu_eff,
            )

    # Regression: the z reproduction must equal the threshold stored by Befund 63 part g.
    max_dev = 0.0
    for channel in CHANNELS:
        for test_set in TEST_SETS:
            for candidate in CANDIDATES:
                old = stored["part_b_verdicts_n_train_64"][f"{channel}/{test_set}"][candidate]
                new = cells[f"{channel}/{test_set}/{candidate}"]["z"]["threshold_relative"]
                max_dev = max(max_dev, abs(old["threshold_relative"] - new))
    for channel, entry in stored["part_b_verdicts_epoch_budget"].items():
        for epochs in (200, 600):
            old = entry[f"epochs_{epochs}"]["threshold_relative"]
            new = cells[f"{channel}/constant_holdout/transformer_{epochs}ep"]["z"][
                "threshold_relative"
            ]
            max_dev = max(max_dev, abs(old - new))
    print(f"max |z threshold - stored threshold| = {max_dev:.3e}")
    if max_dev > 1e-9:
        raise SystemExit("z reproduction does not match the stored Befund 63 artifact")

    changed = {
        name: {
            "improvement_relative": cell["improvement_relative"],
            "z_threshold": cell["z"]["threshold_relative"],
            "t_threshold": cell["t_range_dof"]["threshold_relative"],
            "resolved_z": cell["z"]["resolved"],
            "resolved_t": cell["t_range_dof"]["resolved"],
        }
        for name, cell in cells.items()
        if cell["z"]["resolved"] != cell["t_range_dof"]["resolved"]
    }
    changed_naive = sorted(
        name
        for name, cell in cells.items()
        if cell["t_range_dof"]["resolved"] != cell["t_naive_dof_n_minus_1"]["resolved"]
    )
    for name, cell in cells.items():
        print(
            f"{name:70s} imp {cell['improvement_relative']:+8.1%}  "
            f"z {cell['z']['threshold_relative']:8.1%} ({'R' if cell['z']['resolved'] else '-'})  "
            f"t {cell['t_range_dof']['threshold_relative']:8.1%} "
            f"nu={cell['t_range_dof']['nu_welch']:.2f} "
            f"({'R' if cell['t_range_dof']['resolved'] else '-'})"
        )
    print("verdicts that change with t:", list(changed))
    print("verdicts that differ between range-dof and naive n-1 dof:", changed_naive)

    write_artifact(
        "sx1_resolution_t",
        {
            "_provenance": provenance(),
            "note": "resolved = |improvement| > threshold; 'R' in the console table",
            "dof_of_range_estimator": dof,
            "stored_artifact_regression_max_abs_deviation": max_dev,
            "cells": cells,
            "verdicts_that_change_with_t": changed,
            "verdicts_differing_between_range_dof_and_naive_dof": changed_naive,
        },
    )


if __name__ == "__main__":
    main()
