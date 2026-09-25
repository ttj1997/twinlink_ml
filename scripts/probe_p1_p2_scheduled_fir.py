r"""P1/P2: Arbeitspunktgeplantes FIR-Modell (GEPLANT) gegen eine Merkmalszahl-Kontrolle.

KEIN Netztraining -- reine OLS-Regression (``np.linalg.lstsq``). Datenerzeugung,
Split-Mechanik, Regime-Definition, FIR-Fensterlogik, k=43, 3 Trainingspool-Seeds und
fixer Konstantlauf-Holdout werden unveraendert aus ``probe_n1_regime_crosstab.py``
importiert (``_generate_pools``, ``_crosstab_for_k``, ``_fir_design``,
``_fit_mode_keep_mask``) statt erneut abgetippt -- bitgleiche Datenbasis garantiert, kein
Drift-Risiko. Import funktioniert, weil beide Skripte im selben ``scripts/``-Verzeichnis
liegen (Python haengt beim Aufruf ``python scripts/x.py`` das Skriptverzeichnis vorn an
``sys.path``); die importierte Datei hat keinen Modul-Seiteneffekt (``main()`` steht
hinter ``if __name__ == "__main__"``). KEINE Aenderung an ``src/`` oder an
``probe_n1_regime_crosstab.py``.

Hintergrund (Befund 49, ``docs/04_stand_ap1.md``): lokal gefittete FIR-Modelle schlagen
das gepoolte Modell auf ``product_moisture_kg_per_kg`` deutlich (Faktor 66 in
Residuenvarianz auf dem HEISS-Holdout). Offene Frage: Arbeitspunktabhaengigkeit der
Parameter (dann genuegt ein arbeitspunktgeplantes lineares Modell) oder echte
Zustandsabhaengigkeit der Dynamik (dann Begruendung fuer Sequenzmodelle)?

P1 -- GEPLANT: gepooltes FIR-Modell (k=43), dessen Merkmalssatz zusaetzlich die
Wechselwirkung jedes linearen/quadratischen FIR-Merkmals mit der gemessenen
``bed_temperature_celsius`` zum Zeitpunkt t enthaelt, plus ``bed_temperature_celsius(t)``
selbst als Haupteffekt (Standardpraxis bei Interaktionsmodellen: Haupteffekt immer mit
aufnehmen). Merkmalszahl verdoppelt sich dadurch exakt (siehe ``_fir_design_scheduled``).

WICHTIG fuer die Interpretation: ``bed_temperature_celsius(t)`` ist fuer die Kanaele
``energy_target``, ``product_moisture_kg_per_kg`` und ``d50_um`` ein ANDERER Ausgangskanal
als das Ziel -- keine autoregressive Ruckfuehrung des Zielkanals. Fuer das Ziel
``bed_temperature_celsius`` selbst ist die Konstruktion dagegen autoregressiv (das Modell
bekommt den Zielwert zum Vorhersagezeitpunkt direkt als Merkmal) -- die Zahl dort ist
KEIN Beleg fuer Arbeitspunktplanung, sondern ein Leckage-Artefakt, und wird entsprechend
gekennzeichnet.

P2 -- CONTROL: gepooltes FIR-Modell ohne Betriebspunktbezug, aber mit einem hoeheren k,
sodass die Merkmalszahl mit GEPLANT vergleichbar ist (naechstliegendes k, siehe
``_select_control_k``). Prueft, ob der GEPLANT-Gewinn reine Merkmalszahl ist statt
Arbeitspunktbezug.

Ausgewertet auf HEISS-Holdout und KALT-Holdout wie in N1: R2, Var(y) und
Residuenvarianz in physikalischen Einheiten, Merkmalszahl, 3 Seeds, Median und Spannweite.
POOLED und LOKAL (k=43) werden dafuer aus ``_crosstab_for_k`` (N1, unveraendert)
mitberichtet, damit GEPLANT/CONTROL direkt daneben stehen.

Q1 -- Konditionierung: je Trainings-Entwurfsmatrix (POOLED/GEPLANT/CONTROL, je Seed)
Konditionszahl (``np.linalg.cond``), effektiver Rang (``np.linalg.matrix_rank``),
Verhaeltnis groesste/kleinste Spaltennorm und der von ``lstsq`` gemeldete Rang (s.
``_design_matrix_diagnostics``). Zusatz-Kontrollrechnung: dieselben drei Modelle mit
spaltenweise standardisierten Merkmalen (Mittel 0, Std 1 je Trainings-Seed, Bias-Spalte
ausgenommen, s. ``_standardize_design``) neu gefittet -- R2/Residuenvarianz auf beiden
Holdouts, NICHT als neuer Standard der FIR-Pipeline gedacht, nur zur Kontrolle, ob der
tief negative HEISS-R2 (u.a. ``product_moisture_kg_per_kg``) durch schlechte
Konditionierung der Entwurfsmatrix miterklaerbar ist statt rein strukturell.
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

# actions, outputs, k, *, point_mask -> (design, run_idx, t_idx); s. _fir_design_scheduled
# und _fir_design (N1) fuer die beiden konkreten Implementierungen.
DesignFn = Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]


def _fir_design_scheduled(
    actions: np.ndarray,
    outputs: np.ndarray,
    bed_temp_idx: int,
    k: int,
    *,
    point_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FIR-Merkmalsmatrix mit Betttemperatur-Interaktion.

    Fensterlogik (Lags, ``point_mask``, Ausschluss leerer Laeufe) identisch zu
    ``probe_n1_regime_crosstab._fir_design``, nur um die Wechselwirkungsspalten
    ``[lin, quad] * T_bed(t)`` und den Haupteffekt ``T_bed(t)`` erweitert (vor der
    abschliessenden Bias-Spalte).
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
    """Unveraenderte FIR-Merkmale, ``outputs`` nur fuer die ``DesignFn``-Signatur da.

    N1s ``_fir_design`` unveraendert -- gemeinsam fuer POOLED (k=``K_MAIN``) und CONTROL
    (k=``k_control``) verwendet, beide ohne Betriebspunktbezug.
    """
    return _fir_design(actions, k, point_mask=point_mask)


def _n_features_plain(n_actions: int, k: int) -> int:
    return 2 * (k + 1) * n_actions + 1


def _select_control_k(n_actions: int, target_features: int) -> int:
    """Naechstliegendes k, dessen unveraenderte FIR-Merkmalszahl target_features trifft.

    Sucht ueber k=1..399, keine geschlossene Form (die Merkmalszahl ist ungerade,
    ``target_features`` i.A. gerade -- exakte Gleichheit ist nicht erreichbar).
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
    """Konditionszahl, effektiver Rang, Spaltennorm-Verhaeltnis und lstsq-Rang.

    Der von ``lstsq`` gemeldete Rang haengt nur von der Entwurfsmatrix ab (SVD-basiert,
    unabhaengig von der rechten Seite `b`) -- ein Nullvektor als `b` liefert denselben Rang
    wie ein echter Fit, ohne einen Zielkanal auszuzeichnen.
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
    """Konditionierungsdiagnostik der gepoolten Trainings-Entwurfsmatrix, je Seed.

    Dieselbe Maske (``_fit_mode_keep_mask(..., "pooled")``) wie beim eigentlichen Fit in
    ``_pooled_variant_crosstab`` -- die Entwurfsmatrix ist Ziel-unabhaengig, daher reicht
    eine Diagnostik je Seed statt je (Seed, Kanal).
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
    """Spaltenweise z-Standardisierung (Mittel 0, Std 1), aus den Trainingsdaten geschaetzt.

    Die letzte Spalte (Bias/Intercept, konstant 1) bleibt unveraendert: sie hat Std=0,
    Standardisieren wuerde durch Null teilen und den Intercept zerstoeren.
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
    """Gepooltes Fit-Modell (``fit_mode="pooled"``, s. N1), Merkmale aus ``design_fn``.

    ``standardize=True`` (Q1-Kontrollrechnung): Entwurfsmatrix je Seed spaltenweise
    z-standardisiert (Statistik aus dem Trainingsfenster dieses Seeds, angewendet auf
    Training UND Holdout desselben Seeds) -- deshalb werden die Holdout-Entwuerfe hier,
    anders als im unstandardisierten Fall, JE SEED neu transformiert statt einmal vorab.
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

    def _design_geplant(
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
        _design_geplant,
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
            lokal_cell = baseline[channel][f"fit_{regime}__eval_{regime}"]
            geplant_cell = geplant[channel][regime]
            control_cell = control[channel][regime]
            cells_by_label = (
                ("POOLED", pooled_cell),
                ("LOKAL", lokal_cell),
                ("GEPLANT", geplant_cell),
                ("CONTROL", control_cell),
            )
            parts = [
                f"{label} R2={cell['r2_median']:+.3f} ResidVar={cell['residual_var_median']:.5g}"
                for label, cell in cells_by_label
            ]
            print(f"  eval={regime:<5s} " + " | ".join(parts))

    moisture_heiss_pooled = baseline["product_moisture_kg_per_kg"]["fit_pooled__eval_heiss"][
        "residual_var_median"
    ]
    moisture_heiss_lokal = baseline["product_moisture_kg_per_kg"]["fit_heiss__eval_heiss"][
        "residual_var_median"
    ]
    moisture_heiss_geplant = geplant["product_moisture_kg_per_kg"]["heiss"]["residual_var_median"]
    gap = moisture_heiss_pooled - moisture_heiss_lokal
    closed = moisture_heiss_pooled - moisture_heiss_geplant
    fraction_closed = closed / gap if gap != 0.0 else float("nan")
    print(
        "Kernfrage product_moisture_kg_per_kg (HEISS-Holdout): "
        f"POOLED ResidVar={moisture_heiss_pooled:.6g}, LOKAL={moisture_heiss_lokal:.6g}, "
        f"GEPLANT={moisture_heiss_geplant:.6g}, Luecke geschlossen={fraction_closed:.4%}"
    )

    print("-- Q1: Konditionierung der Trainings-Entwurfsmatrizen (gepoolte Maske, je Seed) --")
    k_by_variant = {"pooled": K_MAIN, "geplant": K_MAIN, "control": k_control}
    conditioning = (
        _conditioning_report("pooled", K_MAIN, _design_plain, train_pools)
        + _conditioning_report("geplant", K_MAIN, _design_geplant, train_pools)
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
    standardized_geplant = _pooled_variant_crosstab(
        "geplant_standardized",
        K_MAIN,
        _design_geplant,
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
                ("GEPLANT_STD", standardized_geplant[channel][regime]),
                ("CONTROL_STD", standardized_control[channel][regime]),
            )
            parts = [
                f"{label} R2={cell['r2_median']:+.3f} ResidVar={cell['residual_var_median']:.5g}"
                for label, cell in cells_by_label
            ]
            print(f"  eval={regime:<5s} " + " | ".join(parts))

    moisture_heiss_pooled_std = standardized_pooled["product_moisture_kg_per_kg"]["heiss"][
        "r2_median"
    ]
    moisture_heiss_geplant_std = standardized_geplant["product_moisture_kg_per_kg"]["heiss"][
        "r2_median"
    ]
    moisture_heiss_control_std = standardized_control["product_moisture_kg_per_kg"]["heiss"][
        "r2_median"
    ]
    moisture_heiss_pooled_r2 = baseline["product_moisture_kg_per_kg"]["fit_pooled__eval_heiss"][
        "r2_median"
    ]
    moisture_heiss_geplant_r2 = geplant["product_moisture_kg_per_kg"]["heiss"]["r2_median"]
    moisture_heiss_control_r2 = control["product_moisture_kg_per_kg"]["heiss"]["r2_median"]
    print(
        "Kernfrage Q1 product_moisture_kg_per_kg (HEISS-Holdout), R2 unstandardisiert -> "
        "standardisiert: "
        f"POOLED {moisture_heiss_pooled_r2:+.3f} -> {moisture_heiss_pooled_std:+.3f}, "
        f"GEPLANT {moisture_heiss_geplant_r2:+.3f} -> {moisture_heiss_geplant_std:+.3f}, "
        f"CONTROL {moisture_heiss_control_r2:+.3f} -> {moisture_heiss_control_std:+.3f}"
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
            "geplant": standardized_geplant,
            "control": standardized_control,
        },
        "_elapsed_s": time.monotonic() - t0,
    }
    out_path = REPO_ROOT / "artifacts" / "runs" / "probe_p1_p2_scheduled_fir.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"geschrieben: {out_path} ({time.monotonic() - t0:.1f}s)")


if __name__ == "__main__":
    main()
