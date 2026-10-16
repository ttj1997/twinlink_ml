"""Study module of the exhaust recirculation (docs/22): recovery layouts, booking, tasks, resume."""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from twinlink_ml.evaluation import exhaust_recirculation as er
from twinlink_ml.evaluation.carrier_energy import carrier_energy_from_diagnostics
from twinlink_ml.simulation import PlaceholderConfigWarning, ProcessConfig, load_process_config
from twinlink_ml.simulation.oracle_search import evaluate_point

REPO_ROOT = Path(__file__).resolve().parents[1]
K = 271.797


def _config() -> ProcessConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        return load_process_config(REPO_ROOT / "conf" / "process" / "greybox_gab_placeholder.yaml")


def _solve(config: ProcessConfig, r: float | None):
    return evaluate_point(
        90.0,
        1.26,
        1.0,
        20.0,
        0.5,
        max_iterations=30,
        tolerance=1e-7,
        base_config=er.with_recirculation(config, r),
    )


def test_layouts_agree_without_recirculation() -> None:
    config = _config()
    solution = _solve(config, 0.0)
    fresh = er.recovery_balance(
        solution, config, er.RecoverySpec(K, "fresh_air"), t_amb=20.0, t_in=90.0
    )
    mixed = er.recovery_balance(
        solution, config, er.RecoverySpec(K, "mixed_air"), t_amb=20.0, t_in=90.0
    )
    for key in ("recovered_kw", "gas_kw", "co2_kg_per_h"):
        assert fresh[key] == pytest.approx(mixed[key], rel=1e-9)
    assert fresh["recovered_kw"] > 0.0


@pytest.mark.parametrize("r", [0.0, 0.5])
def test_booking_without_recovery_equals_the_carrier_booking_of_m1(r: float) -> None:
    config = _config()
    solution = _solve(config, r)
    balance = er.recovery_balance(solution, config, er.RecoverySpec(K), t_amb=20.0, t_in=90.0)
    carrier = carrier_energy_from_diagnostics(
        solution.diagnostics,
        heater="gas",
        process_heater_efficiency=config.energy.heater_efficiency,
        gas_heater_efficiency=0.80,
    )
    assert balance["co2_without_recovery_kg_per_h"] == pytest.approx(K * carrier.co2_kg_per_h)
    assert balance["gas_kw"] == pytest.approx(
        (balance["duty_kw"] - balance["recovered_kw"]) / 0.80, rel=1e-9
    )


def test_recirculation_halves_the_purge_and_mixed_layout_loses_the_pinch() -> None:
    config = _config()
    solution = _solve(config, 0.5)
    fresh = er.recovery_balance(
        solution, config, er.RecoverySpec(K, "fresh_air"), t_amb=20.0, t_in=90.0
    )
    mixed = er.recovery_balance(
        solution, config, er.RecoverySpec(K, "mixed_air"), t_amb=20.0, t_in=90.0
    )
    assert fresh["purge_dry_air_kg_per_s"] == pytest.approx(
        0.5 * K * solution.dry_air_mass_flow_kg_per_s
    )
    # the mixture is warmer than the fresh air: less (here no) recovery within dT_min
    assert mixed["recovered_kw"] < fresh["recovered_kw"]


def test_fog_ratio_is_zero_without_recirculation_and_below_one_at_a_dry_point() -> None:
    config = _config()
    assert er.mixing_saturation_ratio(_solve(config, None), config.gas) == 0.0
    assert 0.0 < er.mixing_saturation_ratio(_solve(config, 0.5), config.gas) < 1.0


def test_task_needs_exactly_one_of_fixed_r_and_box() -> None:
    common: dict[str, Any] = {
        "key": "x",
        "context_index": 0,
        "ambient_temperature_celsius": 20.0,
        "ambient_relative_humidity": 0.5,
        "bounds_t_in": (60.0, 120.0),
        "bounds_u": (1.26, 2.5),
        "quality_threshold_kg_per_kg": 0.144,
    }
    with pytest.raises(ValueError, match="exactly one"):
        er.RecirculationTask(**common, fixed_r=None)
    with pytest.raises(ValueError, match="exactly one"):
        er.RecirculationTask(**common, fixed_r=0.2, r_box=(0.0, 0.9))
    with pytest.raises(ValueError, match="fresh-air path"):
        er.RecirculationTask(**common, fixed_r=0.2, fresh_air_path=True)


def test_overrides_replace_single_fields() -> None:
    config = er.apply_overrides(_config(), (("material", "max_saturation_degree", 0.45),))
    assert config.material.max_saturation_degree == 0.45
    assert config.material.critical_moisture_kg_per_kg == 0.20


def test_run_tasks_resumes_from_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake(task: er.RecirculationTask) -> dict[str, Any]:
        calls.append(task.key)
        return {"key": task.key, "value": task.context_index}

    tasks = [
        er.RecirculationTask("a", 0, 20.0, 0.5, (60.0, 120.0), (1.26, 2.5), 0.144),
        er.RecirculationTask("b", 1, 20.0, 0.5, (60.0, 120.0), (1.26, 2.5), 0.144),
    ]
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps({"key": "a", "value": 0}) + "\n", encoding="utf-8")

    class SerialPool:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> SerialPool:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def imap_unordered(self, _function: Any, items: list[er.RecirculationTask]) -> Any:
            return (fake(item) for item in items)

    monkeypatch.setattr(er, "Pool", SerialPool)
    results = er.run_tasks(_config(), tasks, workers=1, cache_path=cache, label="t")
    assert calls == ["b"]
    assert [r["key"] for r in results] == ["a", "b"]
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 2


def test_task_is_picklable_for_the_spawn_pool() -> None:
    import pickle

    task = er.RecirculationTask(
        "x", 0, 20.0, 0.5, (60.0, 120.0), (1.26, 2.5), 0.144, recovery=er.RecoverySpec(K)
    )
    assert pickle.loads(pickle.dumps(task)) == task
    assert dataclasses.asdict(er.RecoverySpec(K))["layout"] == "fresh_air"
