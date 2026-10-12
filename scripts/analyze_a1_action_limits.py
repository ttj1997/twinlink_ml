r"""How often does the simulator limit the requested actions? Data of the first pilot (A1.3).

Pure analysis of data that is regenerated deterministically (no model, no training). Written before
the run: this file is committed before it is executed (clean tree, ``git/dirty`` is logged).

Data: exactly those of ``scripts/run_a1_pilot_pp.py`` -- the P0 pretraining pool (master seed 101,
64 / 32 / 32 runs) and the PP plant pools (master seeds 1-3, 16 / 8 / 8 runs), process config
``greybox_gab_placeholder.yaml`` with the gas-velocity projection ON, ``DataSpec.new_study``. The split
digest of the P0 pool is checked against the digest recorded in ``a1_pilot_pp.json``.

What counts as "limited" (rates are per sample = run x time step, and per run)
-----------------------------------------------------------------------------
1. Safety projection of the gas velocity to ``[u_mf(d50), u_terminal(d50)]`` (``project_action``, state
   dependent): requested != realised. Lower limit = the request was below ``u_mf`` and was raised,
   upper limit = the request was above ``u_terminal`` and was lowered. Recorded in
   ``Trajectory.requested_actions`` / ``actions``.
2. Box limits of the four actions (``enforce_action_bounds``, a clip inside ``step`` that is NOT
   recorded in the trajectory): the requested and the realised action compared with the bounds of the
   config, per channel and per bound.
3. The other three channels: requested != realised (the projection touches only the gas velocity).
4. The simulator inside the hybrid model (P0, driven with the set points of the PP runs, started in
   the steady state at the set points): the same projection rates, and how often it disagrees with the
   plant on whether a sample is projected.

    tools/run_here.sh scripts/analyze_a1_action_limits.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))

from _common import RUNS_DIR, provenance, require_clean_tree, write_artifact
from run_a1_pilot_pp import PRETRAIN_SEED, generate, load_config, make_plan

from twinlink_ml.models import hybrid
from twinlink_ml.simulation.base import Trajectory
from twinlink_ml.simulation.greybox import GreyBoxFluidBedGranulator

OUTPUT_NAME = "a1_action_limits_pilot1"
EPS = 1e-12


def rates(
    trajectory: Trajectory, action_names: tuple[str, ...], lower: Any, upper: Any
) -> dict[str, Any]:
    requested = (
        trajectory.requested_actions
        if trajectory.requested_actions is not None
        else trajectory.actions
    )
    realised = trajectory.actions
    n_runs, n_steps, _ = realised.shape
    gas = action_names.index("gas_velocity_m_per_s")
    raised = requested[..., gas] < realised[..., gas] - EPS
    lowered = requested[..., gas] > realised[..., gas] + EPS
    projected = raised | lowered
    quarter = n_steps // 4
    box = {
        name: {
            "requested_below_min_pct": 100.0 * float((requested[..., i] < lower[i] - EPS).mean()),
            "requested_above_max_pct": 100.0 * float((requested[..., i] > upper[i] + EPS).mean()),
            "realised_below_min_pct": 100.0 * float((realised[..., i] < lower[i] - EPS).mean()),
            "realised_above_max_pct": 100.0 * float((realised[..., i] > upper[i] + EPS).mean()),
        }
        for i, name in enumerate(action_names)
    }
    other = {
        name: 100.0 * float((np.abs(requested[..., i] - realised[..., i]) > EPS).mean())
        for i, name in enumerate(action_names)
        if i != gas
    }
    return {
        "n_runs": int(n_runs),
        "n_samples": int(n_runs * n_steps),
        "projection_active": bool(trajectory.projection_active),
        "gas_projection": {
            "lower_limit_u_mf_samples_pct": 100.0 * float(raised.mean()),
            "upper_limit_u_terminal_samples_pct": 100.0 * float(lowered.mean()),
            "any_samples_pct": 100.0 * float(projected.mean()),
            "runs_affected_pct": 100.0 * float(projected.any(axis=1).mean()),
            "runs_lower_pct": 100.0 * float(raised.any(axis=1).mean()),
            "runs_upper_pct": 100.0 * float(lowered.any(axis=1).mean()),
            "share_of_projected_samples_in_first_quarter_pct": 100.0
            * float(projected[:, :quarter].sum())
            / max(int(projected.sum()), 1),
        },
        "other_channels_changed_by_projection_samples_pct": other,
        "box_limits": box,
    }


def main() -> None:
    require_clean_tree()
    plan = make_plan(False)
    config = load_config()
    names = tuple(config.actions.names)
    bounds = config.actions.bounds
    lower = [b.minimum for b in bounds]
    upper = [b.maximum for b in bounds]

    recorded = json.loads((RUNS_DIR / "a1_pilot_pp.json").read_text(encoding="utf-8"))
    results: dict[str, Any] = {}

    p0 = generate(config, "P0", plan.pretrain_spec, PRETRAIN_SEED)
    digest_matches = p0.split.digest == recorded["pretraining"]["split_digest"]
    if not digest_matches:
        raise RuntimeError("P0 pool differs from the pool of the pilot (split digest)")
    results["P0_pretraining_pool_seed101"] = rates(p0.trajectory, names, lower, upper)

    for pool in plan.pools:
        data = generate(config, "PP", plan.pool_spec, pool)
        trajectory = data.trajectory
        results[f"PP_plant_pool_seed{pool}"] = rates(trajectory, names, lower, upper)

        set_points = (
            trajectory.requested_actions
            if trajectory.requested_actions is not None
            else trajectory.actions
        )
        states, solved, _ = hybrid.stationary_initial_states(
            data.base_config, set_points[:, 0, :], trajectory.disturbances[:, 0, :]
        )
        quiet = dataclasses.replace(
            data.base_config,
            measurement=dataclasses.replace(
                data.base_config.measurement,
                relative_noise_std=0.0,
                absolute_noise_std_temperature_k=0.0,
            ),
        )
        simulator = GreyBoxFluidBedGranulator(quiet)
        simulated = simulator.rollout(
            set_points,
            trajectory.disturbances,
            run_id=trajectory.run_id,
            seed=0,
            initial_state=simulator.backend.asarray(states),
        )
        entry = rates(simulated, names, lower, upper)
        gas = names.index("gas_velocity_m_per_s")
        plant_projected = np.abs(set_points[..., gas] - trajectory.actions[..., gas]) > EPS
        sim_projected = (
            np.abs(simulated.requested_actions[..., gas] - simulated.actions[..., gas]) > EPS
        )
        entry["plant_vs_simulator_projection_disagree_samples_pct"] = 100.0 * float(
            (plant_projected != sim_projected).mean()
        )
        entry["gas_velocity_realised_mean_abs_difference_plant_vs_simulator_m_per_s"] = float(
            np.abs(trajectory.actions[..., gas] - simulated.actions[..., gas]).mean()
        )
        entry["steady_state_start_solved_runs"] = int(solved.sum())
        results[f"P0_simulator_in_hybrid_channel_pool{pool}"] = entry

    write_artifact(
        OUTPUT_NAME,
        {
            "_provenance": provenance(),
            "design": {
                "data": "identical to scripts/run_a1_pilot_pp.py (pilot 1)",
                "p0_pool_split_digest_equals_pilot_artifact": digest_matches,
                "action_names": list(names),
                "box_lower": lower,
                "box_upper": upper,
                "epsilon": EPS,
            },
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
