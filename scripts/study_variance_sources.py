r"""Decompose the spread by source and state the resolving power (step 5, 1+2).

Usage::

    python scripts/study_variance_sources.py --log artifacts/runs/variance.jsonl

So far one master seed varied dataset realisation, sensor noise and
model initialisation together. That is the conservative statement and remains the reported
curve -- but it does not answer whether a difference between two architectures comes from the
training or from the dataset realisation. This study varies the three sources
individually (``SeedPlan``) and reports three ranges separately.

The resolving power falls out of the same calculation: the spread of the **parameter-free**
persistence is pure dataset realisation and thus the lower bound of every comparison.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from twinlink_ml.evaluation import (
    DataPoolCache,
    DataSpec,
    RolloutEvaluator,
    run_variance_decomposition,
)
from twinlink_ml.evaluation.resolution import resolution_report
from twinlink_ml.models import LinearARXModel, PersistenceModel, StaticRegressionModel
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils import JsonlExperimentLogger, SeedPlan

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"

FACTORIES = {
    "persistence": PersistenceModel,
    "static": StaticRegressionModel,
    "arx": LinearARXModel,
}


def parse_arguments() -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Prozess-Config. Erlaubt einen Lauf gegen eine korrigierte "
        "Parametrisierung (z. B. conf/process/greybox_lab_moisture.yaml) "
        "ohne Code-Aenderung.",
    )
    parser.add_argument("--n-train-max", type=int, default=64)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--run-duration-s", type=float, default=28_800.0)
    parser.add_argument("--log", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Compute the decomposition and print the report."""
    arguments = parse_arguments()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(arguments.config)

    spec = DataSpec(
        run_duration_s=arguments.run_duration_s,
        n_train_max=arguments.n_train_max,
        n_val=arguments.n_train_max // 2,
        n_test=arguments.n_train_max // 2,
        horizon=arguments.horizon,
    )
    plan = SeedPlan(master_seeds=tuple(arguments.seeds))
    cache = DataPoolCache(config, spec)
    logger = JsonlExperimentLogger(arguments.log) if arguments.log else None

    records = run_variance_decomposition(
        config,
        FACTORIES,
        spec=spec,
        plan=plan,
        sizes=arguments.sizes,
        evaluator=RolloutEvaluator(horizons=(1, arguments.horizon)),
        logger=logger,
        cache=cache,
    )

    print(
        f"Pool {spec.pool_size} Läufe, Δt = {spec.sample_time_s:.0f} s, "
        f"Master-Seeds {plan.master_seeds}\n"
        f"Aufwand: {cache.n_simulations} Simulationen, "
        f"{cache.n_remeasurements} Neuvermessungen "
        f"(ohne Cache wären es {cache.n_remeasurements} Simulationen)"
    )
    for target in spec.target_names:
        print()
        print(resolution_report(records, target=target, horizon=arguments.horizon))

    if logger is not None:
        logger.close()
        print(f"\n{len(records)} Einträge in {arguments.log}")


if __name__ == "__main__":
    main()
