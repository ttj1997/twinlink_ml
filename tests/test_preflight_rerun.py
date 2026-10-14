"""Pre-flight gate of the full re-run: data checks before any training."""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import preflight_rerun as pre  # noqa: E402
import run_arch_rerun_sweep as sweep  # noqa: E402
import run_arch_rerun_worker as worker  # noqa: E402

from twinlink_ml.simulation import PlaceholderConfigWarning, load_process_config  # noqa: E402


@pytest.fixture(scope="module")
def config():  # type: ignore[no-untyped-def]
    import dataclasses

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        base = load_process_config(worker.old_worker.CONFIG_PATH)
    return dataclasses.replace(
        base, numerics=dataclasses.replace(base.numerics, enforce_gas_velocity_projection=True)
    )


@pytest.fixture(scope="module")
def old() -> list[dict]:
    return pre.load_old()


@pytest.fixture(scope="module")
def results(config, old):  # type: ignore[no-untyped-def]
    return pre.check_a1(config, 1, old, 1e-6), pre.check_n(config, 1, old)


def test_the_real_pools_of_seed_1_pass(results) -> None:  # type: ignore[no-untyped-def]
    a1, n_arm = results
    assert a1["failures"] == [], a1["failures"]
    assert n_arm["failures"] == [], n_arm["failures"]
    assert a1["guard_pool"]["projected_steps"] > 0  # legacy trajectories are limited
    assert n_arm["guard_pool"]["projected_steps"] == 0
    assert n_arm["guard_holdout"]["projected_steps"] == 0
    assert all(v["worst"] < 1e-6 for v in a1["persistence_vs_old"].values())


def test_a_changed_old_record_is_detected(config, old) -> None:  # type: ignore[no-untyped-def]
    broken = copy.deepcopy(old)
    for r in broken:
        if r["params"]["model/label"] == "persistence":
            key = next(k for k in r["metrics"] if k.startswith("rmse_run_median_physical"))
            r["metrics"][key] *= 1.01
    assert pre.check_a1(config, 1, broken, 1e-6)["failures"]
    wrong = copy.deepcopy(old)
    for r in wrong:
        r["params"]["split/digest"] = "x"
    assert any("split/digest" in f for f in pre.check_a1(config, 1, wrong, 1e-6)["failures"])


def test_a_guard_that_acts_is_detected(config, old, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    real = pre._gen

    def acting(cfg, spec, seed):  # type: ignore[no-untyped-def]
        data = real(cfg, spec, seed)
        if spec.new_study_version == 2:
            # pretend the guard acted: swap in the legacy pool, which is limited
            return real(cfg, worker.legacy_spec(), seed)
        return data

    monkeypatch.setattr(pre, "_gen", acting)
    failures = pre.check_n(config, 1, old)["failures"]
    assert any("the guard acted" in f for f in failures)


def test_the_sweep_refuses_without_a_passed_preflight(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        sys,
        "argv",
        ["sweep", "--run-name", "unit", "--arms", "A1", "--require-preflight", "no_such_preflight"],
    )
    with pytest.raises(SystemExit, match="preflight artifact"):
        sweep.main()
