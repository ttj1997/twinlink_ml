r"""N1: Regime-Kreuztabelle -- schlaegt ein lokal gefittetes FIR-Modell das gepoolte.

KEIN Netztraining -- reine OLS-Regression (``np.linalg.lstsq``), Datenbasis/Merkmalslogik
unveraendert gegenueber ``scripts/probe_l1_clip_sensitivity.py`` (fp 82f65f576a2e5253,
Δt=25 s, 3 Trainingspool-Seeds, fixer Konstantlauf-Holdout, FIR linear+quadratisch
elementweise, OLS via ``lstsq``). KEINE Aenderung an ``src/``.

Hintergrund: Befund 47 (``docs/04_stand_ap1.md``) zeigt an drei Beharrungszustaenden
(Eintrittstemperatur 60/90/120 °C -> Betttemperatur 37.38/62.55/87.79 °C), dass die
langsame Mode des Teilsystems (Wassermasse, Betttemperatur) stark vom Betriebspunkt
abhaengt (37.0 s/1194.9 s, 30.2 s/359.9 s, 38.8 s/133.5 s -- schnelle Mode 30-39 s
uebereinstimmend, langsame Mode um Faktor 9 verschieden). Hypothese: ein einzelnes LZI-
Modell kann beide Regime nicht gleichzeitig abbilden -- die mechanistische Erklaerung
dafuer, dass eine FIR-Regression auf gepoolten Daten bei ``product_moisture_kg_per_kg``
schlecht abschneidet (Befund 47/48: R2_holdout 0.354 bei k=43), waehrend sie auf einer
eingeschraenkten Teilmenge deutlich besser wird (Befund 48: 0.785, dort allerdings ueber
die Verletzungsmaske, nicht ueber die hier verwendete Betriebspunkt-Regimegrenze).

Regime-Definition, punktweise, getrennt von der Verletzungsmaske aus Befund 46:
    HEISS = bed_temperature_celsius > 70 °C
    KALT  = bed_temperature_celsius < 55 °C
    dazwischen: Uebergangszone, verworfen.

Kreuztabelle bei k=43, fuer alle vier Ausgangskanaele: drei Modelle (auf HEISS, auf KALT,
auf HEISS+KALT gepoolt gefittet) gegen zwei Auswertemengen (HEISS-Holdout, KALT-Holdout)
-- sechs Zellen je Kanal. Je Zelle: R2, Var(y) und Residuenvarianz in physikalischen
Einheiten (Var(y) und Residuenvarianz sind Pflicht, weil R2 zwischen Zellen mit
verschiedenen Auswertemengen nicht direkt vergleichbar ist), Anzahl Punkte, 3 Seeds,
Median und Spannweite.

Zusatz bei k=143 (~ 3 * 1194.9 s / 25 s ~ 143.4, als runder Wert getestet; die exakte
ceil(3*tau/dt) liegt bei 144, siehe N2/diagnostics.time_constant_report): nur die Zelle
"KALT gefittet, KALT ausgewertet", Pruefung auf Saettigung des kalten Regimes bei
ausreichend langem Fenster.
"""

from __future__ import annotations

import dataclasses
import json
import time
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.evaluation import DataSpec, generate_data
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.simulation.config import ProcessConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml"
HOLDOUT_SEED = 999
TRAIN_SEEDS = (1, 2, 3)
K_MAIN = 43
K_EXTRA = 143
HOT_THRESHOLD_CELSIUS = 70.0
COLD_THRESHOLD_CELSIUS = 55.0
TARGET_CHANNELS = (
    "energy_target",
    "bed_temperature_celsius",
    "product_moisture_kg_per_kg",
    "d50_um",
)
REGIMES = ("heiss", "kalt")
FIT_MODES = ("heiss", "kalt", "pooled")


def _excited_spec() -> DataSpec:
    return DataSpec(
        sample_time_s=25.0,
        run_duration_s=28_800.0,
        n_train_max=64,
        n_val=32,
        n_test=32,
        history_length=32,
        horizon=36,
        stride=8,
    )


def _holdout_spec(excited: DataSpec) -> DataSpec:
    return dataclasses.replace(
        excited, n_train_max=2, n_val=2, n_test=excited.n_test, constant_run_fraction=1.0
    )


def _fir_design(
    actions: np.ndarray, k: int, *, point_mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FIR-Merkmalsmatrix, unveraendert gegenueber ``probe_e2_e3_e4_lag_depth.py``."""
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
        feat_list.append(np.concatenate([lin, quad], axis=-1))
        run_idx_list.append(np.full(ts.size, run))
        t_idx_list.append(ts)
    features = np.concatenate(feat_list, axis=0)
    run_idx = np.concatenate(run_idx_list, axis=0)
    t_idx = np.concatenate(t_idx_list, axis=0)
    design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=-1)
    return design, run_idx, t_idx


def _load_config() -> ProcessConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return load_process_config(CONFIG_PATH)


def _regime_masks(bed_temperature: np.ndarray) -> dict[str, np.ndarray]:
    """Punktweise Regime-Zugehoerigkeit (Form wie ``bed_temperature``), s. Modul-Docstring."""
    return {
        "heiss": bed_temperature > HOT_THRESHOLD_CELSIUS,
        "kalt": bed_temperature < COLD_THRESHOLD_CELSIUS,
    }


def _regime_fractions(heiss_count: int, kalt_count: int, total: int) -> dict[str, float]:
    heiss_fraction = heiss_count / total
    kalt_fraction = kalt_count / total
    return {
        "heiss": heiss_fraction,
        "kalt": kalt_fraction,
        "transition": 1.0 - heiss_fraction - kalt_fraction,
    }


def _generate_pools() -> tuple[
    ProcessConfig,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[int, dict[str, np.ndarray | dict[str, np.ndarray]]],
]:
    config = _load_config()
    excited_spec = _excited_spec()
    holdout_spec = _holdout_spec(excited_spec)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        holdout = generate_data(config, holdout_spec, master_seed=HOLDOUT_SEED)
    holdout_actions = holdout.trajectory.actions[4:36]
    holdout_outputs = holdout.trajectory.outputs[4:36]
    channels = holdout.trajectory.metadata["output_names"]
    bed_temp_idx = channels.index("bed_temperature_celsius")
    holdout_masks = _regime_masks(holdout_outputs[:, :, bed_temp_idx])

    train_pools: dict[int, dict] = {}
    for seed in TRAIN_SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PlaceholderConfigWarning)
            excited = generate_data(config, excited_spec, master_seed=seed)
        train_actions = excited.trajectory.actions[:64]
        train_outputs = excited.trajectory.outputs[:64]
        train_masks = _regime_masks(train_outputs[:, :, bed_temp_idx])
        train_pools[seed] = {
            "actions": train_actions,
            "outputs": train_outputs,
            "masks": train_masks,
        }

    return config, channels, holdout_actions, holdout_outputs, holdout_masks, train_pools


def _population_report(
    holdout_masks: dict[str, np.ndarray], train_pools: dict[int, dict]
) -> dict[str, dict[str, float]]:
    holdout_total = next(iter(holdout_masks.values())).size
    holdout_heiss = int(np.sum(holdout_masks["heiss"]))
    holdout_kalt = int(np.sum(holdout_masks["kalt"]))

    train_total = 0
    train_heiss = 0
    train_kalt = 0
    for pool in train_pools.values():
        masks = pool["masks"]
        train_total += masks["heiss"].size
        train_heiss += int(np.sum(masks["heiss"]))
        train_kalt += int(np.sum(masks["kalt"]))

    return {
        "holdout": _regime_fractions(holdout_heiss, holdout_kalt, holdout_total),
        "train_pool": _regime_fractions(train_heiss, train_kalt, train_total),
    }


def _fit_mode_keep_mask(masks: dict[str, np.ndarray], fit_mode: str) -> np.ndarray:
    if fit_mode == "pooled":
        return masks["heiss"] | masks["kalt"]
    return masks[fit_mode]


def _crosstab_for_k(
    k: int,
    channels: tuple[str, ...],
    holdout_actions: np.ndarray,
    holdout_outputs: np.ndarray,
    holdout_masks: dict[str, np.ndarray],
    train_pools: dict[int, dict],
    *,
    fit_modes: tuple[str, ...] = FIT_MODES,
    eval_regimes: tuple[str, ...] = REGIMES,
) -> dict[str, dict[str, dict]]:
    eval_designs = {
        regime: _fir_design(holdout_actions, k, point_mask=~holdout_masks[regime])
        for regime in eval_regimes
    }

    result_per_channel: dict[str, dict[str, dict]] = {}
    for channel in TARGET_CHANNELS:
        target_idx = channels.index(channel)
        eval_targets = {
            regime: holdout_outputs[run_idx, t_idx, target_idx]
            for regime, (_design, run_idx, t_idx) in eval_designs.items()
        }

        cells: dict[str, dict] = {}
        for fit_mode in fit_modes:
            per_regime_seed_values: dict[str, dict[str, list[float]]] = {
                regime: {"r2": [], "var_y": [], "residual_var": []} for regime in eval_regimes
            }
            for seed in TRAIN_SEEDS:
                pool = train_pools[seed]
                keep = _fit_mode_keep_mask(pool["masks"], fit_mode)
                design_tr, run_idx_tr, t_idx_tr = _fir_design(pool["actions"], k, point_mask=~keep)
                target_tr = pool["outputs"][run_idx_tr, t_idx_tr, target_idx]
                coefficients, *_ = np.linalg.lstsq(design_tr, target_tr, rcond=None)

                for regime in eval_regimes:
                    design_ev, _run_idx_ev, _t_idx_ev = eval_designs[regime]
                    target_ev = eval_targets[regime]
                    variance = float(np.var(target_ev))
                    residual = target_ev - design_ev @ coefficients
                    residual_variance = float(np.var(residual))
                    r2 = float("nan") if variance <= 0.0 else 1.0 - residual_variance / variance
                    per_regime_seed_values[regime]["r2"].append(r2)
                    per_regime_seed_values[regime]["var_y"].append(variance)
                    per_regime_seed_values[regime]["residual_var"].append(residual_variance)

            for regime in eval_regimes:
                values = per_regime_seed_values[regime]
                n_points = int(eval_designs[regime][0].shape[0])
                cells[f"fit_{fit_mode}__eval_{regime}"] = {
                    "fit": fit_mode,
                    "eval": regime,
                    "n_points": n_points,
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

    population = _population_report(holdout_masks, train_pools)
    print(
        f"Population Trainingspool: heiss={population['train_pool']['heiss']:.4%}, "
        f"kalt={population['train_pool']['kalt']:.4%}, "
        f"Uebergang={population['train_pool']['transition']:.4%}"
    )
    print(
        f"Population Holdout:       heiss={population['holdout']['heiss']:.4%}, "
        f"kalt={population['holdout']['kalt']:.4%}, "
        f"Uebergang={population['holdout']['transition']:.4%}"
    )

    crosstab_k43 = _crosstab_for_k(
        K_MAIN, channels, holdout_actions, holdout_outputs, holdout_masks, train_pools
    )
    for channel, cells in crosstab_k43.items():
        print(f"-- {channel} (k={K_MAIN}) --")
        for cell in cells.values():
            print(
                f"  fit={cell['fit']:<7s} eval={cell['eval']:<5s} n={cell['n_points']:6d} "
                f"R2_median={cell['r2_median']:+.3f} [{cell['r2_range'][0]:+.3f}, "
                f"{cell['r2_range'][1]:+.3f}] "
                f"Var(y)_median={cell['var_y_median']:.5g} "
                f"ResidVar_median={cell['residual_var_median']:.5g}"
            )

    crosstab_k143_kalt_kalt = _crosstab_for_k(
        K_EXTRA,
        channels,
        holdout_actions,
        holdout_outputs,
        holdout_masks,
        train_pools,
        fit_modes=("kalt",),
        eval_regimes=("kalt",),
    )
    for channel, cells in crosstab_k143_kalt_kalt.items():
        cell = cells["fit_kalt__eval_kalt"]
        print(
            f"-- {channel} (k={K_EXTRA}, kalt/kalt) -- "
            f"n={cell['n_points']} R2_median={cell['r2_median']:+.3f} "
            f"ResidVar_median={cell['residual_var_median']:.5g}"
        )

    result = {
        "fingerprint": config.fingerprint,
        "hot_threshold_celsius": HOT_THRESHOLD_CELSIUS,
        "cold_threshold_celsius": COLD_THRESHOLD_CELSIUS,
        "target_channels": list(TARGET_CHANNELS),
        "population": population,
        "k_main": K_MAIN,
        "crosstab_k43": crosstab_k43,
        "k_extra": K_EXTRA,
        "crosstab_k143_kalt_kalt": crosstab_k143_kalt_kalt,
        "_elapsed_s": time.monotonic() - t0,
    }
    out_path = REPO_ROOT / "artifacts" / "runs" / "probe_n1_regime_crosstab.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"geschrieben: {out_path} ({time.monotonic() - t0:.1f}s)")


if __name__ == "__main__":
    main()
