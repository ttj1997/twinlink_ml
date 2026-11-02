r"""P1/P2: operating-point-scheduled FIR model (GEPLANT) against a feature-count control.

NO network training -- pure OLS regression (``np.linalg.lstsq``). Data generation,
split mechanics, regime definition, FIR window logic, k=43, 3 training-pool seeds and
fixed constant-run holdout are imported unchanged from ``probe_n1_regime_crosstab.py``
(``_generate_pools``, ``_crosstab_for_k``, ``_fir_design``,
``_fit_mode_keep_mask``) instead of being typed out again -- bit-identical data basis
guaranteed, no risk of drift. The import works because both scripts lie in the same
``scripts/`` directory (on the call ``python scripts/x.py`` Python prepends the script
directory to ``sys.path``); the imported file has no module side effect (``main()`` sits
behind ``if __name__ == "__main__"``). NO change to ``src/`` or to
``probe_n1_regime_crosstab.py``.

Background (Befund 49, ``docs/04_stand_ap1.md``): locally fitted FIR models beat
the pooled model on ``product_moisture_kg_per_kg`` clearly (factor 66 in
residual variance on the HEISS holdout). Open question: operating-point dependence of the
parameters (then an operating-point-scheduled linear model suffices) or genuine
state dependence of the dynamics (then a justification for sequence models)?

P1 -- GEPLANT: pooled FIR model (k=43) whose feature set additionally contains the
interaction of every linear/quadratic FIR feature with the measured
``bed_temperature_celsius`` at time t, plus ``bed_temperature_celsius(t)``
itself as a main effect (standard practice for interaction models: always include the
main effect). The feature count thereby doubles exactly (see ``_fir_design_scheduled``).

IMPORTANT for the interpretation: for the channels ``energy_target``,
``product_moisture_kg_per_kg`` and ``d50_um``, ``bed_temperature_celsius(t)`` is a DIFFERENT
output channel than the target -- no autoregressive feedback of the target channel. For the
target ``bed_temperature_celsius`` itself, by contrast, the construction is autoregressive (the
model receives the target value at the prediction time directly as a feature) -- the number
there is NO evidence for operating-point scheduling but a leakage artifact, and is marked
accordingly.

P2 -- CONTROL: pooled FIR model without operating-point reference, but with a higher k,
so that the feature count is comparable to GEPLANT (nearest k, see
``_select_control_k``). Checks whether the GEPLANT gain is pure feature count rather than
operating-point reference.

Evaluated on the HEISS holdout and the KALT holdout as in N1: R2, Var(y) and
residual variance in physical units, feature count, 3 seeds, median and range.
POOLED and LOKAL (k=43) are reported alongside for this from ``_crosstab_for_k`` (N1,
unchanged), so that GEPLANT/CONTROL stand directly next to them.

Q1 -- conditioning: per training design matrix (POOLED/GEPLANT/CONTROL, per seed)
condition number (``np.linalg.cond``), effective rank (``np.linalg.matrix_rank``),
ratio of largest/smallest column norm and the rank reported by ``lstsq`` (see
``_design_matrix_diagnostics``). Additional control computation: the same three models
refitted with column-wise standardised features (mean 0, std 1 per training seed, bias
column excluded, see ``_standardize_design``) -- R2/residual variance on both
holdouts, NOT intended as a new standard of the FIR pipeline, only as a check whether the
deeply negative HEISS R2 (among others ``product_moisture_kg_per_kg``) can be partly explained
by poor conditioning of the design matrix rather than being purely structural.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from probe_n1_regime_crosstab import (
    K_MAIN,
    REGIMES,
    TARGET_CHANNELS,
    TRAIN_SEEDS,
    _crosstab_for_k,
    _fir_design,
    _fit_mode_keep_mask,
    _generate_pools,
    _population_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# actions, outputs, k, *, point_mask -> (design, run_idx, t_idx); see _fir_design_scheduled
# and _fir_design (N1) for the two concrete implementations.
DesignFn = Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]


def _fir_design_scheduled(
    actions: np.ndarray,
    outputs: np.ndarray,
    bed_temp_idx: int,
    k: int,
    *,
    point_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FIR feature matrix with bed temperature interaction.

    Window logic (lags, ``point_mask``, exclusion of empty runs) identical to
    ``probe_n1_regime_crosstab._fir_design``, only extended by the interaction columns
    ``[lin, quad] * T_bed(t)`` and the main effect ``T_bed(t)`` (before the
    final bias column).
    """
    n_runs, n_steps, _n_actions = actions.shape
    run_idx_list = []
    t_idx_list = []
    feat_list = []
    for run in range(n_runs):
        ts = np.arange(k, n_steps)
        if point_mask is not None:
            keep = ~point_mask[run, ts]
            ts = ts[keep]
        if ts.size == 0:
            continue
        lags = np.stack([actions[run, ts - i, :] for i in range(k + 1)], axis=1)
        lin = lags.reshape(ts.size, -1)
        quad = lin**2
        base = np.concatenate([lin, quad], axis=-1)
        t_bed = outputs[run, ts, bed_temp_idx]
        interaction = base * t_bed[:, None]
        feat_list.append(np.concatenate([base, interaction, t_bed[:, None]], axis=-1))
        run_idx_list.append(np.full(ts.size, run))
        t_idx_list.append(ts)
    features = np.concatenate(feat_list, axis=0)
    run_idx = np.concatenate(run_idx_list, axis=0)
    t_idx = np.concatenate(t_idx_list, axis=0)
    design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=-1)
    return design, run_idx, t_idx


def _design_plain(
    actions: np.ndarray,
    _outputs: np.ndarray,
    k: int,
    *,
    point_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unchanged FIR features, ``outputs`` only present for the ``DesignFn`` signature.

    N1's ``_fir_design`` unchanged -- used jointly for POOLED (k=``K_MAIN``) and CONTROL
    (k=``k_control``), both without reference to the operating point.
    """
    return _fir_design(actions, k, point_mask=point_mask)


def _n_features_plain(n_actions: int, k: int) -> int:
    return 2 * (k + 1) * n_actions + 1


def _select_control_k(n_actions: int, target_features: int) -> int:
    """Nearest k whose unchanged FIR feature count matches target_features.

    Searches over k=1..399, no closed form (the feature count is odd,
    ``target_features`` in general even -- exact equality cannot be reached).
    """
    best_k = 1
    best_diff = abs(_n_features_plain(n_actions, best_k) - target_features)
    for k_candidate in range(2, 400):
        diff = abs(_n_features_plain(n_actions, k_candidate) - target_features)
        if diff < best_diff:
            best_diff = diff
            best_k = k_candidate
    return best_k


def _design_matrix_diagnostics(design: np.ndarray) -> dict[str, float | int]:
    """Condition number, effective rank, column norm ratio and lstsq rank.

    The rank reported by ``lstsq`` depends only on the design matrix (SVD-based,
    independent of the right-hand side `b`) -- a zero vector as `b` yields the same rank
    as a real fit, without singling out a target channel.
    """
    condition_number = float(np.linalg.cond(design))
    effective_rank = int(np.linalg.matrix_rank(design))
    column_norms = np.linalg.norm(design, axis=0)
    norm_ratio = float(column_norms.max() / column_norms.min())
    _coefficients, _residuals, lstsq_rank, _singular_values = np.linalg.lstsq(
        design, np.zeros(design.shape[0]), rcond=None
    )
    return {
        "condition_number": condition_number,
        "effective_rank": effective_rank,
        "max_min_column_norm_ratio": norm_ratio,
        "lstsq_rank": int(lstsq_rank),
        "n_points": int(design.shape[0]),
        "n_features": int(design.shape[1]),
    }


def _conditioning_report(
    variant: str,
    k: int,
    design_fn: DesignFn,
    train_pools: dict[int, dict],
) -> list[dict[str, float | int | str]]:
    """Conditioning diagnostics of the pooled training design matrix, per seed.

    The same mask (``_fit_mode_keep_mask(..., "pooled")``) as in the actual fit in
    ``_pooled_variant_crosstab`` -- the design matrix is target-independent, so one
    diagnostic per seed suffices instead of per (seed, channel).
    """
    reports: list[dict[str, float | int | str]] = []
    for seed in TRAIN_SEEDS:
        pool = train_pools[seed]
        keep = _fit_mode_keep_mask(pool["masks"], "pooled")
        design_tr, _run_idx_tr, _t_idx_tr = design_fn(
            pool["actions"], pool["outputs"], k, point_mask=~keep
        )
        diagnostics = _design_matrix_diagnostics(design_tr)
        reports.append({"variant": variant, "seed": seed, **diagnostics})
    return reports


def _standardize_design(
    design_tr: np.ndarray, eval_designs: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Column-wise z-standardisation (mean 0, std 1), estimated from the training data.

    The last column (bias/intercept, constant 1) stays unchanged: it has std=0,
    standardising would divide by zero and destroy the intercept.
    """
    n_features = design_tr.shape[1]
    feature_cols = slice(0, n_features - 1)
    mean = design_tr[:, feature_cols].mean(axis=0)
    std = design_tr[:, feature_cols].std(axis=0)
    std_safe = np.where(std > 0.0, std, 1.0)

    def _apply(design: np.ndarray) -> np.ndarray:
        out = design.copy()
        out[:, feature_cols] = (design[:, feature_cols] - mean) / std_safe
        return out

    design_tr_std = _apply(design_tr)
    eval_designs_std = {regime: _apply(design) for regime, design in eval_designs.items()}
    return design_tr_std, eval_designs_std


def _pooled_variant_crosstab(
    variant: str,
    k: int,
    design_fn: DesignFn,
    channels: tuple[str, ...],
    holdout_actions: np.ndarray,
    holdout_outputs: np.ndarray,
    holdout_masks: dict[str, np.ndarray],
    train_pools: dict[int, dict],
    *,
    eval_regimes: tuple[str, ...] = REGIMES,
    standardize: bool = False,
) -> dict[str, dict[str, dict]]:
    """Pooled fit model (``fit_mode="pooled"``, see N1), features from ``design_fn``.

    ``standardize=True`` (Q1 control computation): design matrix z-standardised column-wise
    per seed (statistics from the training window of this seed, applied to
    training AND holdout of the same seed) -- therefore the holdout designs are
    transformed anew PER SEED here, unlike in the unstandardised case, instead of once upfront.
    """
    eval_designs_raw = {
        regime: design_fn(holdout_actions, holdout_outputs, k, point_mask=~holdout_masks[regime])
        for regime in eval_regimes
    }

    result_per_channel: dict[str, dict[str, dict]] = {}
    for channel in TARGET_CHANNELS:
        target_idx = channels.index(channel)
        eval_targets = {
            regime: holdout_outputs[run_idx, t_idx, target_idx]
            for regime, (_design, run_idx, t_idx) in eval_designs_raw.items()
        }

        per_regime_seed_values: dict[str, dict[str, list[float]]] = {
            regime: {"r2": [], "var_y": [], "residual_var": []} for regime in eval_regimes
        }
        for seed in TRAIN_SEEDS:
            pool = train_pools[seed]
            keep = _fit_mode_keep_mask(pool["masks"], "pooled")
            design_tr, run_idx_tr, t_idx_tr = design_fn(
                pool["actions"], pool["outputs"], k, point_mask=~keep
            )
            target_tr = pool["outputs"][run_idx_tr, t_idx_tr, target_idx]

            if standardize:
                eval_designs_raw_only = {r: d for r, (d, _ri, _ti) in eval_designs_raw.items()}
                design_tr_fit, eval_designs_fit = _standardize_design(
                    design_tr, eval_designs_raw_only
                )
            else:
                design_tr_fit = design_tr
                eval_designs_fit = {r: d for r, (d, _ri, _ti) in eval_designs_raw.items()}

            coefficients, *_ = np.linalg.lstsq(design_tr_fit, target_tr, rcond=None)

            for regime in eval_regimes:
                design_ev = eval_designs_fit[regime]
                target_ev = eval_targets[regime]
                variance = float(np.var(target_ev))
                residual = target_ev - design_ev @ coefficients
                residual_variance = float(np.var(residual))
                r2 = float("nan") if variance <= 0.0 else 1.0 - residual_variance / variance
                per_regime_seed_values[regime]["r2"].append(r2)
                per_regime_seed_values[regime]["var_y"].append(variance)
                per_regime_seed_values[regime]["residual_var"].append(residual_variance)

        cells: dict[str, dict] = {}
        for regime in eval_regimes:
            values = per_regime_seed_values[regime]
            n_points = int(eval_designs_raw[regime][0].shape[0])
            n_features = int(eval_designs_raw[regime][0].shape[1])
            cells[regime] = {
                "variant": variant,
                "eval": regime,
                "k": k,
                "standardized": standardize,
                "n_points": n_points,
                "n_features": n_features,
                "r2_by_seed": values["r2"],
                "r2_median": float(np.median(values["r2"])),
                "r2_range": [float(min(values["r2"])), float(max(values["r2"]))],
                "var_y_median": float(np.median(values["var_y"])),
                "var_y_range": [float(min(values["var_y"])), float(max(values["var_y"]))],
                "residual_var_median": float(np.median(values["residual_var"])),
                "residual_var_range": [
                    float(min(values["residual_var"])),
                    float(max(values["residual_var"])),
                ],
            }
        result_per_channel[channel] = cells
    return result_per_channel


def main() -> None:
    t0 = time.monotonic()
    config, channels, holdout_actions, holdout_outputs, holdout_masks, train_pools = (
        _generate_pools()
    )
    bed_temp_idx = channels.index("bed_temperature_celsius")
    n_actions = int(holdout_actions.shape[-1])

    population = _population_report(holdout_masks, train_pools)
    print(
        f"Population Trainingspool: heiss={population['train_pool']['heiss']:.4%}, "
        f"kalt={population['train_pool']['kalt']:.4%}"
    )
    print(
        f"Population Holdout:       heiss={population['holdout']['heiss']:.4%}, "
        f"kalt={population['holdout']['kalt']:.4%}"
    )

    baseline = _crosstab_for_k(
        K_MAIN, channels, holdout_actions, holdout_outputs, holdout_masks, train_pools
    )
    n_features_pooled = _n_features_plain(n_actions, K_MAIN)
    print(f"POOLED/LOKAL (aus N1, unveraendert): k={K_MAIN}, n_features={n_features_pooled}")

    def _design_scheduled(
        actions: np.ndarray,
        outputs: np.ndarray,
        k: int,
        *,
        point_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _fir_design_scheduled(actions, outputs, bed_temp_idx, k, point_mask=point_mask)

    geplant = _pooled_variant_crosstab(
        "geplant",
        K_MAIN,
        _design_scheduled,
        channels,
        holdout_actions,
        holdout_outputs,
        holdout_masks,
        train_pools,
    )
    n_features_geplant = geplant[TARGET_CHANNELS[0]]["heiss"]["n_features"]
    print(f"GEPLANT: k={K_MAIN}, n_features={n_features_geplant}")

    k_control = _select_control_k(n_actions, n_features_geplant)

    control = _pooled_variant_crosstab(
        "control",
        k_control,
        _design_plain,
        channels,
        holdout_actions,
        holdout_outputs,
        holdout_masks,
        train_pools,
    )
    n_features_control = control[TARGET_CHANNELS[0]]["heiss"]["n_features"]
    print(
        f"CONTROL: k={k_control} (naechstliegend zu GEPLANT-Merkmalszahl), "
        f"n_features={n_features_control}"
    )

    for channel in TARGET_CHANNELS:
        is_autoregressive = channel == "bed_temperature_celsius"
        tag = " [AUTOREGRESSIV/Leckage bei GEPLANT]" if is_autoregressive else ""
        print(f"-- {channel}{tag} --")
        for regime in REGIMES:
            pooled_cell = baseline[channel][f"fit_pooled__eval_{regime}"]
            local_cell = baseline[channel][f"fit_{regime}__eval_{regime}"]
            scheduled_cell = geplant[channel][regime]
            control_cell = control[channel][regime]
            cells_by_label = (
                ("POOLED", pooled_cell),
                ("LOKAL", local_cell),
                ("GEPLANT", scheduled_cell),
                ("CONTROL", control_cell),
            )
            parts = [
                f"{label} R2={cell['r2_median']:+.3f} ResidVar={cell['residual_var_median']:.5g}"
                for label, cell in cells_by_label
            ]
            print(f"  eval={regime:<5s} " + " | ".join(parts))

    moisture_hot_pooled = baseline["product_moisture_kg_per_kg"]["fit_pooled__eval_heiss"][
        "residual_var_median"
    ]
    moisture_hot_local = baseline["product_moisture_kg_per_kg"]["fit_heiss__eval_heiss"][
        "residual_var_median"
    ]
    moisture_hot_scheduled = geplant["product_moisture_kg_per_kg"]["heiss"]["residual_var_median"]
    gap = moisture_hot_pooled - moisture_hot_local
    closed = moisture_hot_pooled - moisture_hot_scheduled
    fraction_closed = closed / gap if gap != 0.0 else float("nan")
    print(
        "Kernfrage product_moisture_kg_per_kg (HEISS-Holdout): "
        f"POOLED ResidVar={moisture_hot_pooled:.6g}, LOKAL={moisture_hot_local:.6g}, "
        f"GEPLANT={moisture_hot_scheduled:.6g}, Luecke geschlossen={fraction_closed:.4%}"
    )

    print("-- Q1: Konditionierung der Trainings-Entwurfsmatrizen (gepoolte Maske, je Seed) --")
    k_by_variant = {"pooled": K_MAIN, "geplant": K_MAIN, "control": k_control}
    conditioning = (
        _conditioning_report("pooled", K_MAIN, _design_plain, train_pools)
        + _conditioning_report("geplant", K_MAIN, _design_scheduled, train_pools)
        + _conditioning_report("control", k_control, _design_plain, train_pools)
    )
    for entry in conditioning:
        variant_k = k_by_variant[str(entry["variant"])]
        print(
            f"  {entry['variant']:<8s} seed={entry['seed']} k={variant_k} "
            f"n={entry['n_points']:6d} p={entry['n_features']:4d} "
            f"cond={entry['condition_number']:.4g} rank_svd={entry['effective_rank']} "
            f"rank_lstsq={entry['lstsq_rank']} normratio={entry['max_min_column_norm_ratio']:.4g}"
        )

    print("-- Q1-Zusatz: Kontrollrechnung mit spaltenweise standardisierten Merkmalen --")
    standardized_pooled = _pooled_variant_crosstab(
        "pooled_standardized",
        K_MAIN,
        _design_plain,
        channels,
        holdout_actions,
        holdout_outputs,
        holdout_masks,
        train_pools,
        standardize=True,
    )
    standardized_scheduled = _pooled_variant_crosstab(
        "geplant_standardized",
        K_MAIN,
        _design_scheduled,
        channels,
        holdout_actions,
        holdout_outputs,
        holdout_masks,
        train_pools,
        standardize=True,
    )
    standardized_control = _pooled_variant_crosstab(
        "control_standardized",
        k_control,
        _design_plain,
        channels,
        holdout_actions,
        holdout_outputs,
        holdout_masks,
        train_pools,
        standardize=True,
    )
    for channel in TARGET_CHANNELS:
        is_autoregressive = channel == "bed_temperature_celsius"
        tag = " [AUTOREGRESSIV/Leckage bei GEPLANT_STD]" if is_autoregressive else ""
        print(f"-- {channel}{tag} (standardisiert) --")
        for regime in REGIMES:
            cells_by_label = (
                ("POOLED_STD", standardized_pooled[channel][regime]),
                ("GEPLANT_STD", standardized_scheduled[channel][regime]),
                ("CONTROL_STD", standardized_control[channel][regime]),
            )
            parts = [
                f"{label} R2={cell['r2_median']:+.3f} ResidVar={cell['residual_var_median']:.5g}"
                for label, cell in cells_by_label
            ]
            print(f"  eval={regime:<5s} " + " | ".join(parts))

    moisture_hot_pooled_std = standardized_pooled["product_moisture_kg_per_kg"]["heiss"][
        "r2_median"
    ]
    moisture_hot_scheduled_std = standardized_scheduled["product_moisture_kg_per_kg"]["heiss"][
        "r2_median"
    ]
    moisture_hot_control_std = standardized_control["product_moisture_kg_per_kg"]["heiss"][
        "r2_median"
    ]
    moisture_hot_pooled_r2 = baseline["product_moisture_kg_per_kg"]["fit_pooled__eval_heiss"][
        "r2_median"
    ]
    moisture_hot_scheduled_r2 = geplant["product_moisture_kg_per_kg"]["heiss"]["r2_median"]
    moisture_hot_control_r2 = control["product_moisture_kg_per_kg"]["heiss"]["r2_median"]
    print(
        "Kernfrage Q1 product_moisture_kg_per_kg (HEISS-Holdout), R2 unstandardisiert -> "
        "standardisiert: "
        f"POOLED {moisture_hot_pooled_r2:+.3f} -> {moisture_hot_pooled_std:+.3f}, "
        f"GEPLANT {moisture_hot_scheduled_r2:+.3f} -> {moisture_hot_scheduled_std:+.3f}, "
        f"CONTROL {moisture_hot_control_r2:+.3f} -> {moisture_hot_control_std:+.3f}"
    )

    result = {
        "fingerprint": config.fingerprint,
        "k_main": K_MAIN,
        "k_control": k_control,
        "n_actions": n_actions,
        "n_features_pooled": n_features_pooled,
        "n_features_geplant": n_features_geplant,
        "n_features_control": n_features_control,
        "target_channels": list(TARGET_CHANNELS),
        "population": population,
        "baseline_k43": baseline,
        "geplant": geplant,
        "control": control,
        "moisture_heiss_gap_closed_fraction": fraction_closed,
        "conditioning": conditioning,
        "standardized_control_check": {
            "pooled": standardized_pooled,
            "geplant": standardized_scheduled,
            "control": standardized_control,
        },
        "_elapsed_s": time.monotonic() - t0,
    }
    out_path = REPO_ROOT / "artifacts" / "runs" / "probe_p1_p2_scheduled_fir.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"geschrieben: {out_path} ({time.monotonic() - t0:.1f}s)")


if __name__ == "__main__":
    main()
