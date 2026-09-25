r"""GA1: Haelt der FA2-Agent eine Sicherheitsmarge zur Feuchtegrenze? Nur Auswertung,
kein Training -- verwendet die bereits gesicherten FA2-Checkpoints bei 10000 Schritten.

Vermuteter Mechanismus (Auftrag): das Orakel-Optimum liegt an 17 von 25 Befund-52-Punkten
exakt auf der Gutsfeuchtegrenze (``binding_moisture``, Befund 62 (a)). Ein strafbasierter
Agent meidet Nebenbedingungsgrenzen (Strafe wirkt schon vor der Grenze, nicht erst danach)
und erkauft Abstand mit hoeherer Gasgeschwindigkeit -- er uebertrocknet systematisch, statt
am Optimum zu sitzen.

Je FA2-Seed, an den 25 Befund-52-Testkontexten, fuer die deterministische Agentenaktion
UND die Orakelaktion:

  a) Produktfeuchte als Abstand zur kritischen Gutsfeuchte (``critical - moisture``,
     positiv = Sicherheitsmarge, ~0 = bindend, negativ = Verletzung)
  b) Betttemperatur als Abstand zur 80-°C-Grenze, ebenso
  c) Gasgeschwindigkeit, Agent gegen Orakel (die Groesse, mit der ein Agent Abstand
     "erkauft", siehe Umgebungsdocstring: Gutsfeuchte bindet bei NIEDRIGER
     Gasgeschwindigkeit)

Kernfrage (Auftrag, ein Satz): liegt der Agent an den Kontexten, an denen das Orakel auf
der Feuchtegrenze liegt, systematisch mit deutlichem Abstand darunter?

Aufruf::

    python scripts/probe_bandit_env_ga1_safety_margin.py \
        --output artifacts/runs/rl_bandit_ga1_safety_margin.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.rl import OperatingPointBanditEnv, load_test_points
from twinlink_ml.simulation import PlaceholderConfigWarning
from twinlink_ml.utils.tracking import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_JSONLS = tuple(
    REPO_ROOT / "artifacts" / "runs" / f"rl_bandit_sac_learning_curve_fa2_seed{seed}.jsonl"
    for seed in (0, 1, 2)
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_ga1_safety_margin.json"
CHECKPOINT_STEP = 10_000


def _suppress_placeholder_warning() -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)


def _checkpoint_path_at_step(seed_jsonl: Path, step: int) -> Path:
    for record in read_jsonl(seed_jsonl):
        if int(record.params["n_env_steps"]) == step:
            return REPO_ROOT / record.params["checkpoint/path"]
    raise ValueError(f"Kein Record mit n_env_steps={step} in {seed_jsonl}")


def _median(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed-jsonls", type=Path, nargs=3, default=list(DEFAULT_SEED_JSONLS))
    parser.add_argument("--checkpoint-step", type=int, default=CHECKPOINT_STEP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    _suppress_placeholder_warning()

    from stable_baselines3 import SAC

    env = OperatingPointBanditEnv()
    critical_moisture = env._base_config.material.critical_moisture_kg_per_kg
    bed_temperature_limit = env._regret_module.BED_TEMPERATURE_LIMIT_C
    points = load_test_points()

    # Orakel-Kennzahlen sind seedunabhaengig -- einmal berechnen.
    oracle_per_point: list[dict[str, Any]] = []
    for point in points:
        _, info = env.evaluate_operating_point(
            inlet_temperature_celsius=point.oracle_inlet_temperature_celsius,
            gas_velocity_m_per_s=point.oracle_gas_velocity_m_per_s,
            ambient_temperature_celsius=point.ambient_temperature_celsius,
            ambient_relative_humidity=point.ambient_relative_humidity,
        )
        check = info["constraint_check"]
        oracle_per_point.append(
            {
                "moisture_margin_kg_per_kg": critical_moisture - check["moisture_kg_per_kg"],
                "bed_temperature_margin_k": bed_temperature_limit - check["bed_temperature_celsius"],
                "gas_velocity_m_per_s": point.oracle_gas_velocity_m_per_s,
                "binding_moisture": bool(check["binding_moisture"]),
            }
        )
    n_oracle_binding_moisture = sum(1 for p in oracle_per_point if p["binding_moisture"])

    print("=" * 88)
    print(
        f"GA1: Sicherheitsmarge zur Feuchtegrenze (kritisch={critical_moisture} kg/kg, "
        f"Bettemperaturgrenze={bed_temperature_limit} C)"
    )
    print(f"Orakel bindet Gutsfeuchte an {n_oracle_binding_moisture}/{len(points)} Punkten")
    print("=" * 88)

    per_seed: dict[str, Any] = {}
    for seed_index, seed_jsonl in enumerate(arguments.seed_jsonls):
        checkpoint_path = _checkpoint_path_at_step(seed_jsonl, arguments.checkpoint_step)
        model = SAC.load(checkpoint_path)

        agent_per_point: list[dict[str, Any]] = []
        for point in points:
            observation = np.array(
                [point.ambient_temperature_celsius, point.ambient_relative_humidity],
                dtype=np.float32,
            )
            action, _ = model.predict(observation, deterministic=True)
            t_in = env._inlet_bound.minimum + 0.5 * (float(action[0]) + 1.0) * env._inlet_bound.span
            u_gas = env._gas_bound.minimum + 0.5 * (float(action[1]) + 1.0) * env._gas_bound.span
            _, info = env.evaluate_operating_point(
                inlet_temperature_celsius=t_in,
                gas_velocity_m_per_s=u_gas,
                ambient_temperature_celsius=point.ambient_temperature_celsius,
                ambient_relative_humidity=point.ambient_relative_humidity,
            )
            if info.get("solve_failed"):
                agent_per_point.append(
                    {
                        "solve_failed": True,
                        "moisture_margin_kg_per_kg": float("nan"),
                        "bed_temperature_margin_k": float("nan"),
                        "gas_velocity_m_per_s": u_gas,
                        "binding_moisture": False,
                    }
                )
                continue
            check = info["constraint_check"]
            agent_per_point.append(
                {
                    "solve_failed": False,
                    "moisture_margin_kg_per_kg": critical_moisture - check["moisture_kg_per_kg"],
                    "bed_temperature_margin_k": bed_temperature_limit - check["bed_temperature_celsius"],
                    "gas_velocity_m_per_s": u_gas,
                    "binding_moisture": bool(check["binding_moisture"]),
                }
            )

        # Kernvergleich: NUR an den Punkten, an denen das Orakel die Feuchtegrenze bindet.
        binding_indices = [i for i, p in enumerate(oracle_per_point) if p["binding_moisture"]]
        agent_moisture_margin_all = [p["moisture_margin_kg_per_kg"] for p in agent_per_point]
        agent_moisture_margin_binding = [
            agent_per_point[i]["moisture_margin_kg_per_kg"] for i in binding_indices
        ]
        agent_bed_margin_all = [p["bed_temperature_margin_k"] for p in agent_per_point]
        gas_delta_all = [
            agent_per_point[i]["gas_velocity_m_per_s"] - oracle_per_point[i]["gas_velocity_m_per_s"]
            for i in range(len(points))
        ]
        gas_delta_binding = [
            agent_per_point[i]["gas_velocity_m_per_s"] - oracle_per_point[i]["gas_velocity_m_per_s"]
            for i in binding_indices
        ]
        n_agent_binding_moisture = sum(1 for p in agent_per_point if p["binding_moisture"])

        seed_key = str(seed_index)
        per_seed[seed_key] = {
            "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
            "median_moisture_margin_agent_all_kg_per_kg": _median(agent_moisture_margin_all),
            "median_moisture_margin_agent_binding_subset_kg_per_kg": _median(
                agent_moisture_margin_binding
            ),
            "median_bed_temperature_margin_agent_all_k": _median(agent_bed_margin_all),
            "median_gas_velocity_delta_agent_minus_oracle_all_m_per_s": _median(gas_delta_all),
            "median_gas_velocity_delta_agent_minus_oracle_binding_subset_m_per_s": _median(
                gas_delta_binding
            ),
            "n_agent_binding_moisture": n_agent_binding_moisture,
            "n_binding_subset": len(binding_indices),
            "per_point": agent_per_point,
        }

        print(f"\n--- Seed {seed_index} ({checkpoint_path.name}) " + "-" * 20)
        print(
            f"  Median Feuchtemarge Agent, ALLE 25 Punkte: "
            f"{per_seed[seed_key]['median_moisture_margin_agent_all_kg_per_kg']:.4f} kg/kg"
        )
        print(
            f"  Median Feuchtemarge Agent, NUR die {len(binding_indices)} Orakel-bindenden "
            f"Punkte: {per_seed[seed_key]['median_moisture_margin_agent_binding_subset_kg_per_kg']:.4f} "
            "kg/kg (Orakel dort per Definition ~0.0)"
        )
        print(
            f"  Median Delta u_gas (Agent - Orakel), bindende Punkte: "
            f"{per_seed[seed_key]['median_gas_velocity_delta_agent_minus_oracle_binding_subset_m_per_s']:+.4f} m/s"
        )
        print(f"  Agent bindet Gutsfeuchte selbst an {n_agent_binding_moisture}/{len(points)} Punkten")

    median_across_seeds_binding_subset = _median(
        [per_seed[str(s)]["median_moisture_margin_agent_binding_subset_kg_per_kg"] for s in range(3)]
    )
    median_gas_delta_binding_subset = _median(
        [
            per_seed[str(s)]["median_gas_velocity_delta_agent_minus_oracle_binding_subset_m_per_s"]
            for s in range(3)
        ]
    )

    print("\n" + "=" * 88)
    print(
        f"Median ueber Seeds, Feuchtemarge Agent an bindenden Punkten: "
        f"{median_across_seeds_binding_subset:.4f} kg/kg (kritisch={critical_moisture} kg/kg)"
    )
    print(f"Median ueber Seeds, Delta u_gas an bindenden Punkten: {median_gas_delta_binding_subset:+.4f} m/s")

    output = {
        "critical_moisture_kg_per_kg": critical_moisture,
        "bed_temperature_limit_c": bed_temperature_limit,
        "n_oracle_binding_moisture": n_oracle_binding_moisture,
        "n_points": len(points),
        "oracle_per_point": oracle_per_point,
        "per_seed": per_seed,
        "median_across_seeds": {
            "moisture_margin_agent_binding_subset_kg_per_kg": median_across_seeds_binding_subset,
            "gas_velocity_delta_agent_minus_oracle_binding_subset_m_per_s": median_gas_delta_binding_subset,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
