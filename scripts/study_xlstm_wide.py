r"""Capacity-matched xLSTM variant -- complement to ``study_architectures.py``.

Why a second, smaller script instead of a new ``--hidden-size`` flag in
``study_architectures.py``
-------------------------------------------------------------------------------
``study_architectures.py`` runs all candidates of a label with the same,
architecture-independent ``TorchSequenceConfig`` -- capacity is deliberately not a
study factor there (docstring of ``CompositeSequenceModel``, reason 3). For the
xLSTM comparison (``docs/04_stand_ap1.md``, Befund 29) exactly that is needed: the same
architecture (``core="xlstm"``) at two different ``hidden_size`` values, so that the
comparison with GRU/LSTM covers both "same width" and "same parameter count".
A new CLI flag in ``study_architectures.py`` would have opened this capacity freedom for
*all* architectures -- exactly the freedom that deliberately does not exist there.
This script instead calls the same library function
(:func:`~twinlink_ml.evaluation.protocol.run_architecture_comparison`) with a
self-built factory mapping that contains only the label ``xlstm`` --
``persistence``/``climatology`` do **not** run again here; they are already
in the target file from the preceding run with ``--models xlstm persistence
climatology``.

Parameters exactly as in the reference run
-------------------------------------
The same ``DataSpec`` (defaults: ``n_train_max=64``, ``horizon=24``,
``run_duration_s=28800``, like ``arch_composite.jsonl``), the same ``SeedPlan``, the same
process config, the same support points -- otherwise a comparison of the figures across
two runs would not be admissible. Unlike ``study_architectures.py``, this
script writes **appending** to an already existing log file (``JsonlExperimentLogger`` is
appending anyway, see there) -- the call presupposes that the file from the
preceding run already exists.

Usage (single-point sample, see Befund 29)::

    python scripts/study_xlstm_wide.py --hidden-size 317 --label xlstm_h317 \
        --sizes 64 --seeds 1 2 3 --modes inline --sources joint --with-baselines \
        --log artifacts/runs/arch_xlstm.jsonl
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from twinlink_ml.evaluation import (
    DataSpec,
    ModelFactory,
    ObservationMode,
    RolloutEvaluator,
    run_architecture_comparison,
)
from twinlink_ml.models.baselines import LinearARXModel, PersistenceModel
from twinlink_ml.models.composite import CompositeSequenceModel
from twinlink_ml.models.torch_sequence import TorchSequenceConfig
from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config
from twinlink_ml.utils import JsonlExperimentLogger, SeedPlan
from twinlink_ml.utils.seeding import VarianceSource

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "process" / "greybox_placeholder.yaml"

_MODE_BY_NAME = {"inline": ObservationMode.INLINE, "soft_sensor": ObservationMode.SOFT_SENSOR}
_SOURCE_BY_NAME = {"joint": VarianceSource.JOINT, "model": VarianceSource.MODEL}


def _existing_target_names(log_path: Path) -> tuple[str, ...] | None:
    """``data/target_names`` of the first record in ``log_path``.

    ``None`` for an empty or missing file.

    By docstring contract this script appends to an already existing log file
    (e.g. a continuation of ``study_architectures.py``). An appending run with
    a different target channel count than the records already logged would produce a log with
    inconsistent channel count -- worthless, and nobody notices later.
    """
    if not log_path.exists():
        return None
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            names = record["params"].get("data/target_names")
            return tuple(names) if names is not None else None
    return None


def _check_target_names_consistency(log_path: Path, target_names: tuple[str, ...]) -> None:
    """Aborts hard if ``log_path`` already carries records with a different target channel count.

    Separate function (not inline in ``main()``) so that the schema check is testable without a
    full CLI call.
    """
    existing = _existing_target_names(log_path)
    if existing is not None and existing != target_names:
        raise RuntimeError(
            f"{log_path}: bestehende Records tragen data/target_names={existing}, "
            f"dieser Lauf wuerde {target_names} anfuegen -- ein Log mit uneinheitlicher "
            "Kanalzahl ueber seine Records ist wertlos und faellt spaeter niemandem auf."
        )


def parse_arguments() -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--n-train-max", type=int, default=64)
    parser.add_argument("--run-duration-s", type=float, default=28_800.0)
    parser.add_argument(
        "--modes", type=str, nargs="+", choices=sorted(_MODE_BY_NAME), default=["inline"]
    )
    parser.add_argument(
        "--sources", type=str, nargs="+", choices=sorted(_SOURCE_BY_NAME), default=["joint"]
    )
    parser.add_argument(
        "--with-baselines",
        action="store_true",
        help="persistence und arx zusätzlich mitfahren (billig, deterministisch) -- "
        "damit architecture_verdicts aus dieser einen Logdatei allein auswertbar ist, "
        "ohne über zwei Läufe zusammenzuführen.",
    )
    parser.add_argument("--log", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run only the candidate ``xlstm`` at a specific width."""
    arguments = parse_arguments()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(arguments.config)

    base = DataSpec(
        run_duration_s=arguments.run_duration_s,
        n_train_max=arguments.n_train_max,
        n_val=arguments.n_train_max // 2,
        n_test=arguments.n_train_max // 2,
        horizon=arguments.horizon,
    )
    _check_target_names_consistency(arguments.log, base.target_names)

    plan = SeedPlan(master_seeds=tuple(arguments.seeds))
    logger = JsonlExperimentLogger(arguments.log)
    evaluator = RolloutEvaluator(horizons=(1, arguments.horizon))

    net_config = TorchSequenceConfig(core="xlstm", hidden_size=arguments.hidden_size)
    factories: dict[str, ModelFactory] = {
        arguments.label: lambda: CompositeSequenceModel(net_config)
    }
    if arguments.with_baselines:
        factories["persistence"] = PersistenceModel
        factories["arx"] = LinearARXModel

    def report_progress(*, source: str, replicate: int, n_train: int) -> None:
        print(
            f"  Achse {source}  Replikat {replicate + 1}/{plan.n_replicates}  n_train={n_train}",
            flush=True,
        )

    modes = tuple(_MODE_BY_NAME[name] for name in arguments.modes)
    sources = tuple(_SOURCE_BY_NAME[name] for name in arguments.sources)
    print(
        f"=== xLSTM, label={arguments.label!r}, hidden_size={arguments.hidden_size}, "
        f"modes={arguments.modes}, sources={arguments.sources}, "
        f"baselines={'ja' if arguments.with_baselines else 'nein'} -> {arguments.log} ==="
    )
    records = run_architecture_comparison(
        config,
        factories,
        spec=base,
        plan=plan,
        sizes=arguments.sizes,
        modes=modes,
        sources=sources,
        evaluator=evaluator,
        logger=logger,
        on_point=report_progress,
    )
    logger.close()
    print(f"\n{len(records)} Einträge angefügt an {arguments.log}")


if __name__ == "__main__":
    main()
