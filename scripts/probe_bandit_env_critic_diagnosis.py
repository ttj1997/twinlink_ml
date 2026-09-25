r"""BB1: Mechanismus des Eckenkollaps pruefen -- NUR mit vorhandenen Checkpoints, kein Training.

Zwei vermutete Mechanismen (Auftrag):

* **Optimismus durch Initialisierung**: alle Belohnungen sind negativ, der Kritiker startet
  nahe null; unbesuchte Aktionen erscheinen dadurch besser als besuchte, der Akteur laeuft
  in die Ecken.
* **tanh-Saettigung**: in der Ecke waechst die Voraktivierung, der Gradient durch tanh
  verschwindet, der Akteur kommt nicht zurueck.

Drei Messungen an den XA2-Checkpoints (1000 und 10000 Schritte), fuer alle 25 Befund-52-
Testkontexte:

a) Voraktivierung ``mu`` des Akteurs VOR tanh (``Actor.get_action_dist_params``), je
   Aktionsdimension. ``|mu| > 3`` bedeutet ``tanh' < 0.01`` (nahezu verschwindender
   Gradient) -- siehe :data:`SATURATION_MU_THRESHOLD`.
b) Kritikerwert ``Q`` (beide Kritiker-Netze) an der tatsaechlich ausgegebenen (Ecken-)
   Aktion UND an der Orakelaktion desselben Kontexts (normiert eingespeist, wie ein Agent
   sie saehe). ``Q(Ecke) > Q(Orakel)`` trotz Unzulaessigkeit der Ecke waere eine
   Kritikerfehleinschaetzung.
c) ``Q`` an einer Handvoll Aktionen ohne besonderen Bezug zur ausgegebenen Politik (Mitte,
   Gegenecke, gemischte Ecken) -- ACHTUNG: der Replay-Buffer wurde in XA2 NICHT gesichert
   (nur die Politik-Gewichte, siehe ``utils/checkpoints.py``-Konvention), "tatsaechlich nie
   besucht" ist deshalb NICHT direkt pruefbar. Diese Aktionen sind ein PROXY (weit von der
   ausgegebenen Politik entfernt), keine verifizierte Negativliste -- siehe Auftragsantwort
   fuer die Einordnung dieser Einschraenkung.

Aufruf::

    python scripts/probe_bandit_env_critic_diagnosis.py \
        --output artifacts/runs/rl_bandit_critic_diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.rl import OperatingPointBanditEnv, load_test_points
from twinlink_ml.rl.env import _normalise
from twinlink_ml.simulation import PlaceholderConfigWarning

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints" / "rl_bandit_sac_learning_curve"
ANCHOR_ARTIFACT_PATH = REPO_ROOT / "artifacts" / "runs" / "bb1_bb3_operating_point_regret.json"
OUT_PATH = REPO_ROOT / "artifacts" / "runs" / "rl_bandit_critic_diagnosis.json"
DEFAULT_CHECKPOINT_STEPS = (1_000, 10_000)
SATURATION_MU_THRESHOLD = 3.0
"""``tanh'(3.0) = 1 - tanh(3.0)^2 ~= 0.0099 < 0.01`` -- Schwelle aus dem Auftrag."""

PROXY_UNVISITED_ACTIONS: dict[str, tuple[float, float]] = {
    "mitte": (0.0, 0.0),
    "gegenecke_kalt": (-1.0, -1.0),
    "gemischt_kalt_gas_hoch": (-1.0, 1.0),
    "gemischt_heiss_gas_niedrig": (1.0, -1.0),
}


def _suppress_placeholder_warning() -> None:
    warnings.filterwarnings("ignore", category=PlaceholderConfigWarning)


def _find_checkpoint(step: int, run_uid: str | None) -> Path:
    """Siehe ``probe_bandit_env_policy_inspection.py::_find_checkpoint`` -- ``run_uid``
    fuer exakten Treffer statt mtime-basierter Auswahl, noetig bei parallel laufenden
    Trainingslaeufen.
    """
    if run_uid is not None:
        path = CHECKPOINT_DIR / f"{run_uid}_step{step:06d}.zip"
        if not path.exists():
            raise FileNotFoundError(f"{path} nicht gefunden.")
        return path
    candidates = sorted(
        CHECKPOINT_DIR.glob(f"*_step{step:06d}.zip"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(
            f"Kein Checkpoint fuer Schritt {step} unter {CHECKPOINT_DIR} gefunden -- "
            "scripts/run_rl_bandit_learning_curve.py muss vorher gelaufen sein."
        )
    return candidates[-1]


def _oracle_lookup() -> dict[tuple[float, float], dict[str, Any]]:
    anchor = json.loads(ANCHOR_ARTIFACT_PATH.read_text(encoding="utf-8"))
    return {
        (round(g["ambient_temperature_celsius"], 6), round(g["ambient_relative_humidity"], 6)): g
        for g in anchor["grid_results"]
    }


def diagnose_checkpoint(path: Path, env: OperatingPointBanditEnv) -> dict[str, Any]:
    import torch
    from stable_baselines3 import SAC

    model = SAC.load(path)
    policy = model.policy
    actor = policy.actor
    points = load_test_points()
    oracle_lookup = _oracle_lookup()

    mus: list[list[float]] = []
    q_corner: list[list[float]] = []
    q_oracle: list[list[float]] = []
    per_point: list[dict[str, Any]] = []

    for point in points:
        obs = np.array(
            [point.ambient_temperature_celsius, point.ambient_relative_humidity],
            dtype=np.float32,
        )
        obs_tensor, _ = policy.obs_to_tensor(obs)

        with torch.no_grad():
            mean_actions, _log_std, _ = actor.get_action_dist_params(obs_tensor)
        mu = mean_actions.numpy()[0]
        mus.append(mu.tolist())

        deterministic_action, _ = model.predict(obs, deterministic=True)
        action_tensor = torch.as_tensor(deterministic_action, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_at_corner = model.critic(obs_tensor, action_tensor)
        q_corner_values = [float(q.item()) for q in q_at_corner]
        q_corner.append(q_corner_values)

        key = (
            round(point.ambient_temperature_celsius, 6),
            round(point.ambient_relative_humidity, 6),
        )
        oracle_point = oracle_lookup[key]
        oracle_normalised = np.array(
            [
                _normalise(
                    oracle_point["inlet_temperature_celsius"],
                    env._inlet_bound.minimum,
                    env._inlet_bound.maximum,
                ),
                _normalise(
                    oracle_point["gas_velocity_m_per_s"],
                    env._gas_bound.minimum,
                    env._gas_bound.maximum,
                ),
            ],
            dtype=np.float32,
        )
        oracle_tensor = torch.as_tensor(oracle_normalised).unsqueeze(0)
        with torch.no_grad():
            q_at_oracle = model.critic(obs_tensor, oracle_tensor)
        q_oracle_values = [float(q.item()) for q in q_at_oracle]
        q_oracle.append(q_oracle_values)

        per_point.append(
            {
                "ambient_temperature_celsius": point.ambient_temperature_celsius,
                "ambient_relative_humidity": point.ambient_relative_humidity,
                "mu": mu.tolist(),
                "deterministic_action_normalised": deterministic_action.tolist(),
                "q_at_corner": q_corner_values,
                "q_at_oracle": q_oracle_values,
                "min_q_corner_gt_min_q_oracle": bool(
                    min(q_corner_values) > min(q_oracle_values)
                ),
            }
        )

    mus_arr = np.asarray(mus)
    q_corner_arr = np.asarray(q_corner)
    q_oracle_arr = np.asarray(q_oracle)

    # (c) Proxy-Aktionen fuer "nie besucht", am ersten Kontext.
    obs0 = np.array(
        [points[0].ambient_temperature_celsius, points[0].ambient_relative_humidity],
        dtype=np.float32,
    )
    obs0_tensor, _ = policy.obs_to_tensor(obs0)
    proxy_q: dict[str, list[float]] = {}
    for label, action in PROXY_UNVISITED_ACTIONS.items():
        action_tensor = torch.as_tensor(np.array(action, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            q_proxy = model.critic(obs0_tensor, action_tensor)
        proxy_q[label] = [float(q.item()) for q in q_proxy]

    return {
        "checkpoint_path": str(path.relative_to(REPO_ROOT)),
        "n_contexts": len(points),
        "mu_mean": mus_arr.mean(axis=0).tolist(),
        "mu_std": mus_arr.std(axis=0).tolist(),
        "mu_abs_max": np.abs(mus_arr).max(axis=0).tolist(),
        "n_contexts_saturated": (np.abs(mus_arr) > SATURATION_MU_THRESHOLD).sum(axis=0).tolist(),
        "saturation_mu_threshold": SATURATION_MU_THRESHOLD,
        "q_corner_mean": q_corner_arr.mean(axis=0).tolist(),
        "q_oracle_mean": q_oracle_arr.mean(axis=0).tolist(),
        "n_contexts_corner_ranked_above_oracle": int(
            sum(1 for p in per_point if p["min_q_corner_gt_min_q_oracle"])
        ),
        "proxy_unvisited_action_q_at_context_0": proxy_q,
        "q_corner_at_context_0": q_corner[0],
        "per_point": per_point,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint-steps", type=int, nargs="+", default=list(DEFAULT_CHECKPOINT_STEPS)
    )
    parser.add_argument(
        "--run-uid",
        type=str,
        default=None,
        help="exakter run_uid statt mtime-basierter Auswahl -- noetig bei parallel "
        "laufenden Trainingslaeufen.",
    )
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    _suppress_placeholder_warning()
    env = OperatingPointBanditEnv()

    results: dict[str, Any] = {}
    for step in arguments.checkpoint_steps:
        checkpoint_path = _find_checkpoint(step, arguments.run_uid)
        print(f"\n--- Schritt {step} ({checkpoint_path.name}) " + "-" * 10)
        result = diagnose_checkpoint(checkpoint_path, env)
        results[str(step)] = result

        print(f"(a) mu Mittelwert: {result['mu_mean']}  |mu| max: {result['mu_abs_max']}")
        print(f"    Kontexte mit |mu| > {SATURATION_MU_THRESHOLD} (gesaettigt): "
              f"{result['n_contexts_saturated']} von je {result['n_contexts']}")
        print(f"(b) Q(Ecke) Mittelwert:   {[round(v, 2) for v in result['q_corner_mean']]}")
        print(f"    Q(Orakel) Mittelwert: {[round(v, 2) for v in result['q_oracle_mean']]}")
        print(f"    Kontexte mit Q(Ecke) > Q(Orakel): "
              f"{result['n_contexts_corner_ranked_above_oracle']} von {result['n_contexts']}")
        print(f"(c) Q an Proxy-Aktionen (Kontext 0, Q(Ecke) dort = "
              f"{[round(v, 2) for v in result['q_corner_at_context_0']]}):")
        for label, q in result["proxy_unvisited_action_q_at_context_0"].items():
            print(f"    {label:30s}: {[round(v, 2) for v in q]}")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nErgebnis geschrieben: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
