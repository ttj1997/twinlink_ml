r"""B1: how often are requested actions limited? Rates per plan B0-B4, channel and bound (report only).

Question of the project lead (2026-10-12): in the pilot data, how often does the simulator limit the
requested actions, per channel and bound; are B0-B4 limited differently often (then they are not
equally expensive)? Data: the plans B0-B4 for the master seeds 1-5 of the pilot (the pilot's data
seeds), N = 4 runs each, ``DataSpec.new_study()``, process config of the studies. Measured, per plan
and channel, as shares of all steps:

* simulator side: ``sim_clip_lower/upper`` (realised action differs from the requested one: box clip,
  ``enforce_action_bounds``, or gas-velocity projection if it were on) and the config switches;
* design side: ``req_outside`` (the plan's pre-limit signal outside the box), ``ratelim_changed``
  (the rate limiter of the design changed the signal), ``ratelim_binding`` (the change per step sits at
  the rate limit), ``at_lower/at_upper`` (the requested action sits on a box bound);
* ``fluidisation_*``: shares of steps at which the realised gas velocity lies below the minimum
  fluidisation velocity or above the terminal velocity of the current particle size, i.e. where the
  safety projection of Befund 78 would intervene if it were on (it is off in the study config).

    tools/run_here.sh scripts/report_b1_limit_rates.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import (
    CONFIG_PATH,
    RUNS_DIR,
    provenance,
    require_clean_tree,
    write_artifact,
)

from twinlink_ml.evaluation.experiments import DataSpec
from twinlink_ml.evaluation.growing_pool import GrowingPool, prepare_process_config
from twinlink_ml.simulation import (
    PlaceholderConfigWarning,
    excitation,
    excitation_plans,
    load_process_config,
)
from twinlink_ml.simulation.excitation_plans import PLAN_NAMES, make_plan
from twinlink_ml.utils.seeding import SeedBundle

require_clean_tree()
if (RUNS_DIR / "b1_limit_rates.json").exists():
    raise SystemExit("artifact exists")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", PlaceholderConfigWarning)
    cfg = load_process_config(CONFIG_PATH)
print(
    "enforce_action_bounds",
    cfg.numerics.enforce_action_bounds,
    "projection",
    cfg.numerics.enforce_gas_velocity_projection,
)
spec = DataSpec.new_study(n_train_max=4, n_val=0, n_test=0)
prep = prepare_process_config(cfg, spec)
bounds = prep.actions.bounds
names = prep.actions.names
dt = spec.sample_time_s
recorded = []
orig = excitation.apply_rate_limit


def rec(signal: np.ndarray, rate: float, st: float) -> np.ndarray:
    out = orig(signal, rate, st)
    recorded.append((signal.copy(), out.copy(), rate))
    return out


excitation.apply_rate_limit = rec
excitation_plans.apply_rate_limit = rec
res = {}
for name in PLAN_NAMES:
    acc = {
        k: np.zeros(4)
        for k in [
            "sim_clip_lower",
            "sim_clip_upper",
            "at_lower",
            "at_upper",
            "req_outside",
            "ratelim_changed",
            "ratelim_binding",
            "steps",
        ]
    }
    margin_viol = []
    n_proj = []
    for master in range(1, 6):
        seeds = SeedBundle.from_master(master)
        plan = make_plan(
            name,
            prep,
            sample_time_s=dt,
            run_duration_s=spec.run_duration_s,
            solid_fraction_process="independent",
        )
        recorded.clear()
        sch = plan.build(4, seed=seeds.data_seed)
        pre = dict.fromkeys(range(4))
        # recorded has one entry per channel (denormalised, pre-limit) for b0,b2,b3,b4 (per channel call) and b1 (per channel call)
        assert len(recorded) == 4, len(recorded)
        pool = GrowingPool(cfg, spec, sch, noise_seed=seeds.noise_seed)
        pool.add([0, 1, 2, 3])
        tr = pool.trajectory()
        for c, b in enumerate(bounds):
            pre_c, post_c, rate = recorded[c]
            acc["steps"][c] += post_c.size
            acc["req_outside"][c] += np.sum((pre_c < b.minimum - 1e-9) | (pre_c > b.maximum + 1e-9))
            acc["ratelim_changed"][c] += np.sum(np.abs(pre_c - post_c) > 1e-9 * b.span)
            d = np.abs(np.diff(post_c, axis=1)) / dt
            acc["ratelim_binding"][c] += np.sum(d >= 0.999 * rate)
            req = sch.actions[:, :, c]
            real = tr.actions[:, :, c]
            acc["sim_clip_upper"][c] += np.sum(real < req - 1e-9 * b.span)
            acc["sim_clip_lower"][c] += np.sum(real > req + 1e-9 * b.span)
            acc["at_lower"][c] += np.sum(req <= b.minimum + 1e-9 * b.span)
            acc["at_upper"][c] += np.sum(req >= b.maximum - 1e-9 * b.span)
        lo = tr.diagnostics.get("fluidization_margin_lower_m_per_s")
        up = tr.diagnostics.get("fluidization_margin_upper_m_per_s")
        margin_viol.append(
            ((lo < 0) | (up < 0)).mean(axis=1) if lo is not None else np.array([np.nan])
        )
        margin_viol.append((lo < 0).mean(axis=1))
        margin_viol.append((up < 0).mean(axis=1))
    mv = np.array(margin_viol).reshape(5, 3, -1)  # seed, [any, lower, upper], run
    res[name] = {k: (v / acc["steps"]).tolist() for k, v in acc.items() if k != "steps"}
    res[name]["fluidisation_window_violation_any"] = float(mv[:, 0].mean())
    res[name]["fluidisation_below_u_mf"] = float(mv[:, 1].mean())
    res[name]["fluidisation_above_u_t"] = float(mv[:, 2].mean())
print(names)
for name, r in res.items():
    print("\n==", name)
    for k, v in r.items():
        print(f"  {k:36s}", [round(x, 4) for x in v] if isinstance(v, list) else round(v, 4))
write_artifact(
    "b1_limit_rates",
    {
        "_provenance": provenance(),
        "purpose": "limit rates of the plans B0-B4 (report only)",
        "channels": list(names),
        "master_seeds": [1, 2, 3, 4, 5],
        "n_runs_per_plan_and_seed": 4,
        "enforce_action_bounds": bool(cfg.numerics.enforce_action_bounds),
        "enforce_gas_velocity_projection": bool(cfg.numerics.enforce_gas_velocity_projection),
        "rates": res,
    },
)
