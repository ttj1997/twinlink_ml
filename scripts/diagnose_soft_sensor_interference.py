r"""Diagnosis of Befund 25 (``04_stand_ap1.md``): negative transfer in the soft-sensor case.

Observation this script checks
-------------------------------------
In soft-sensor operation, ``energy_target`` for GRU and LSTM is demonstrably worse than ARX
at *every* learning-curve point, although the same cores clearly hit the same target better
than ARX in the inline case (see ``artifacts/runs/arch_gru.jsonl``,
``observation/mode``). ``src/twinlink_ml/models/torch_sequence.py`` trains all targets with
a shared encoder and a single scalar loss (``_weighted_mse``), averaged **without**
per-target weighting -- confirmed by reading the trainer, not by this script. In
soft-sensor operation, ``product_moisture_kg_per_kg`` and ``d50_um`` are practically
unsolvable for every model checked so far (climatology NRMSE there close to 1.0). The
hypothesis: the gradient noise of these two hard, barely learnable sub-tasks degrades,
via the shared encoder, the otherwise good solution for ``energy_target`` -- negative
transfer between sub-tasks of unequal difficulty.

Setup
------
Three GRU runs on an **identical** data pool and seed per repetition, soft-sensor operation,
``n_train = 32``:

a) as in the existing architecture comparison: ``target_names`` = all three targets, one
   shared network (``torch_sequence.GruSequenceModel``).
b) ``target_names`` = only ``("energy_target",)`` -- the head does not see the two hard
   sub-tasks at all.
c) ``CompositeSequenceModel`` (Befund 26, option 2): a separate network per target, all
   three targets as in (a), but without shared state.

(a) and (b) share trajectory, split and model seed (``dataclasses.replace`` on the same
:class:`~twinlink_ml.evaluation.GeneratedData`); only ``spec.target_names`` differs. An
improvement in (b) can therefore not come from a different data pool or a different
initialisation, only from the removed auxiliary loss. (c) runs on the same data pool as
(a), with the same ``target_names`` set -- the only difference is the architecture.
Reference values (ARX, climatology) run along on the same pool, solely for context.

Invocation::

    python scripts/diagnose_soft_sensor_interference.py --seeds 1 2 3

``--gradient-clip`` allows the follow-up from Befund 26 (04_stand_ap1.md): whether the
global ``clip_grad_norm_`` over all network parameters (instead of separately per target
group) contributes to the poor performance of case (a) can be checked by setting the clip
norm so high that in practice it no longer takes effect (e.g. 100.0 instead of the default
1.0) -- deliberately not disabled, so as not to introduce divergence as a new interfering
signal::

    python scripts/diagnose_soft_sensor_interference.py --seeds 1 2 3 --gradient-clip 100.0

Case (a) and case (c) run in the **same** script execution, so that the additional
training-time cost of (c) over (a) is measured on the same machine, under the same load and
at the same point in time -- a comparison across two separate runs could be distorted by
background load on the machine.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
import warnings
from pathlib import Path

import numpy as np

from twinlink_ml.evaluation import (
    DataSpec,
    GeneratedData,
    ObservationMode,
    RolloutEvaluator,
    ValidationPolicy,
    build_datasets,
    generate_data,
    spec_for_mode,
)
from twinlink_ml.models.base import SequenceModel
from twinlink_ml.models.baselines import ClimatologyModel, LinearARXModel
from twinlink_ml.models.composite import CompositeGruSequenceModel
from twinlink_ml.models.torch_sequence import GruSequenceModel, TorchSequenceConfig
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"

HORIZON = 24
TARGET = "energy_target"
ALL_TARGETS = ("energy_target", "product_moisture_kg_per_kg", "d50_um")


def parse_arguments() -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--n-train", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--run-duration-s", type=float, default=28_800.0)
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="An beide GRU-Läufe (a und b) durchgereicht, siehe Befund 26 Fortsetzung "
        "in 04_stand_ap1.md: mit einem hohen Wert (z.B. 100.0) wirkt clip_grad_norm_ "
        "de facto nicht mehr, ohne das Clipping ganz abzuschalten.",
    )
    return parser.parse_args()


def _soft_sensor_variant(base: DataSpec, target_names: tuple[str, ...]) -> DataSpec:
    """Soft-sensor specification for a given set of targets.

    ``spec_for_mode`` observes everything except the two quality variables; for
    ``target_names = ("energy_target",)`` that is the only existing target anyway,
    so the behaviour stays consistent between (a) and (b).
    """
    variant = dataclasses.replace(base, target_names=target_names)
    return spec_for_mode(variant, ObservationMode.SOFT_SENSOR)


def _fit_and_evaluate(
    data: GeneratedData, n_train: int, model: SequenceModel, evaluator: RolloutEvaluator
) -> tuple[dict[int, float], object, float]:
    """Fit, evaluate, and return the pure ``fit`` time (for the cost comparison)."""
    train, validation, test = build_datasets(data, n_train, policy=ValidationPolicy.FIXED_POOL)
    start = time.perf_counter()
    model.fit(train, seed=data.seeds.model_seed, validation=validation)
    fit_seconds = time.perf_counter() - start
    result = evaluator.evaluate(model, test)
    return result.nrmse_by_horizon[TARGET], model, fit_seconds


def main() -> None:
    """Run both variants over several seeds and compare energy_target."""
    arguments = parse_arguments()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        process_config = load_process_config(arguments.config)

    base = DataSpec(
        run_duration_s=arguments.run_duration_s,
        n_train_max=arguments.n_train,
        n_val=arguments.n_train // 2,
        n_test=arguments.n_train // 2,
        horizon=HORIZON,
    )
    evaluator = RolloutEvaluator(horizons=(1, HORIZON))

    case_a_nrmse: list[float] = []
    case_b_nrmse: list[float] = []
    case_c_nrmse: list[float] = []
    arx_nrmse: list[float] = []
    climatology_nrmse: list[float] = []
    fit_seconds_a: list[float] = []
    fit_seconds_c: list[float] = []
    epochs_a: list[int] = []
    epochs_c: list[dict[str, int]] = []
    last_history_a: tuple[dict[str, float], ...] = ()

    for seed in arguments.seeds:
        print(f"\n=== seed {seed} ===")
        pool_spec = dataclasses.replace(
            base, n_train_max=base.n_train_max, n_val=base.n_val, n_test=base.n_test
        )
        data_all = generate_data(process_config, pool_spec, master_seed=seed)

        data_a = dataclasses.replace(data_all, spec=_soft_sensor_variant(base, ALL_TARGETS))
        data_b = dataclasses.replace(data_all, spec=_soft_sensor_variant(base, (TARGET,)))

        model_a = GruSequenceModel(
            TorchSequenceConfig(core="gru", gradient_clip=arguments.gradient_clip)
        )
        nrmse_a, model_a, seconds_a = _fit_and_evaluate(
            data_a, arguments.n_train, model_a, evaluator
        )
        print(
            f"  (a) alle drei Zielgrößen:      nrmse[{TARGET}, h{HORIZON}] = "
            f"{nrmse_a[HORIZON]:.4f}  ({seconds_a:.1f}s fit, "
            f"{len(model_a.training_history)} Epochen)"
        )
        case_a_nrmse.append(nrmse_a[HORIZON])
        fit_seconds_a.append(seconds_a)
        epochs_a.append(len(model_a.training_history))
        last_history_a = model_a.training_history

        model_b = GruSequenceModel(
            TorchSequenceConfig(core="gru", gradient_clip=arguments.gradient_clip)
        )
        nrmse_b, _, _ = _fit_and_evaluate(data_b, arguments.n_train, model_b, evaluator)
        print(
            f"  (b) nur energy_target:         nrmse[{TARGET}, h{HORIZON}] = {nrmse_b[HORIZON]:.4f}"
        )
        case_b_nrmse.append(nrmse_b[HORIZON])

        model_c = CompositeGruSequenceModel(
            TorchSequenceConfig(core="gru", gradient_clip=arguments.gradient_clip)
        )
        nrmse_c, model_c, seconds_c = _fit_and_evaluate(
            data_a, arguments.n_train, model_c, evaluator
        )
        seed_epochs_c = {name: len(history) for name, history in model_c.training_history.items()}
        print(
            f"  (c) CompositeSequenceModel:    nrmse[{TARGET}, h{HORIZON}] = "
            f"{nrmse_c[HORIZON]:.4f}  ({seconds_c:.1f}s fit)"
        )
        print(f"      Epochen je Teilmodell:      {seed_epochs_c}")
        case_c_nrmse.append(nrmse_c[HORIZON])
        fit_seconds_c.append(seconds_c)
        epochs_c.append(seed_epochs_c)

        arx_model = LinearARXModel()
        nrmse_arx, _, _ = _fit_and_evaluate(data_a, arguments.n_train, arx_model, evaluator)
        print(
            f"  Referenz ARX:                  nrmse[{TARGET}, h{HORIZON}] = "
            f"{nrmse_arx[HORIZON]:.4f}"
        )
        arx_nrmse.append(nrmse_arx[HORIZON])

        climatology_model = ClimatologyModel()
        nrmse_clim, _, _ = _fit_and_evaluate(
            data_a, arguments.n_train, climatology_model, evaluator
        )
        print(
            f"  Referenz Climatology:          nrmse[{TARGET}, h{HORIZON}] = "
            f"{nrmse_clim[HORIZON]:.4f}"
        )
        climatology_nrmse.append(nrmse_clim[HORIZON])

    print("\n\n############ Zusammenfassung ############")
    print(f"n_train={arguments.n_train}, Soft-Sensor, {TARGET}, h={HORIZON}, gepoolter NRMSE\n")
    print(
        f"  (a) alle drei Zielgrößen (Median über Seeds): {np.median(case_a_nrmse):.4f}"
        f"  [{', '.join(f'{v:.4f}' for v in case_a_nrmse)}]"
    )
    print(
        f"  (b) nur energy_target    (Median über Seeds): {np.median(case_b_nrmse):.4f}"
        f"  [{', '.join(f'{v:.4f}' for v in case_b_nrmse)}]"
    )
    print(
        f"  (c) CompositeSequenceModel      (Median): {np.median(case_c_nrmse):.4f}"
        f"  [{', '.join(f'{v:.4f}' for v in case_c_nrmse)}]"
    )
    print(
        f"  ARX-Referenz             (Median über Seeds): {np.median(arx_nrmse):.4f}"
        f"  [{', '.join(f'{v:.4f}' for v in arx_nrmse)}]"
    )
    print(
        f"  Climatology-Referenz     (Median über Seeds): {np.median(climatology_nrmse):.4f}"
        f"  [{', '.join(f'{v:.4f}' for v in climatology_nrmse)}]"
    )

    print("\nTrainingszeit fit() (Median über Seeds, dieselbe Skriptausführung):")
    print(
        f"  (a) gemeinsames Netz:        {np.median(fit_seconds_a):.2f}s"
        f"  [{', '.join(f'{v:.2f}' for v in fit_seconds_a)}]"
    )
    print(
        f"  (c) CompositeSequenceModel:  {np.median(fit_seconds_c):.2f}s"
        f"  [{', '.join(f'{v:.2f}' for v in fit_seconds_c)}]"
    )
    print(
        f"  Faktor (c)/(a):              {np.median(fit_seconds_c) / np.median(fit_seconds_a):.2f}x"
    )

    print("\nTatsächlich gelaufene Epochen je Seed (max_epochs=200, patience=20):")
    print(f"  (a) gemeinsames Netz (alle drei Zielgrößen in einem Trainingslauf): {epochs_a}")
    for target in ALL_TARGETS:
        per_seed = [record[target] for record in epochs_c]
        print(f"  (c) Teilmodell {target:<28} {per_seed}  (Median {np.median(per_seed):.0f})")
    sum_c = [sum(record.values()) for record in epochs_c]
    print(f"  (c) Summe der drei Teilmodelle je Seed: {sum_c}")
    print(
        f"  Verhältnis Summe(c)/(a) je Seed: "
        f"{[round(s / a, 2) for s, a in zip(sum_c, epochs_a, strict=True)]}"
        "  (Erwartung bei gleicher Kosten je Epoche: nahe am Zeitfaktor oben)"
    )

    print("\nLetzter Seed, letzte Trainingsepoche, Verlust pro Zielgröße (Fall a):")
    if last_history_a:
        last = last_history_a[-1]
        for key, value in last.items():
            if key.startswith("train_loss"):
                print(f"  {key:<32} {value:.4f}")


if __name__ == "__main__":
    main()
