"""S0.2: optional quality limit of the product moisture, decoupled from the kinetics.

``MaterialConfig.product_moisture_limit_kg_per_kg`` is optional. Unset, the quality limit
is the critical moisture as before and the field is NOT part of the fingerprint, so every
existing config keeps its fingerprint. Set, it enters the hash. The drying kinetics keep
using ``critical_moisture_kg_per_kg``; only the constraint reads the quality limit.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import pytest

from twinlink_ml.simulation import (
    MaterialConfig,
    PlaceholderConfigWarning,
    ProcessConfig,
    dump_process_config,
    load_process_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "conf" / "process"

# Fingerprints computed with the code of commit 7939a7d (before the field existed):
# (file, sample_time_s=25 + projection flag or None = as loaded).
FINGERPRINTS_BEFORE_S02: dict[tuple[str, bool | None], str] = {
    ("greybox_gab_placeholder.yaml", None): "2acd06f38963c3af",
    ("greybox_gab_placeholder.yaml", False): "6ecedc6bd993fb3d",
    ("greybox_gab_placeholder.yaml", True): "bddca37f716f5cfc",
    ("greybox_lab_moisture.yaml", None): "ebbc22316fe248c0",
    ("greybox_lab_moisture.yaml", False): "f9a4455aa9539a91",
    ("greybox_lab_moisture.yaml", True): "ba85b873034ff983",
    ("greybox_placeholder.yaml", None): "034ee446f64b5159",
    ("greybox_placeholder.yaml", False): "eed1084074c7180c",
    ("greybox_placeholder.yaml", True): "a87281ef999aa37a",
}
MATERIAL_SECTION_FINGERPRINT_GAB = "185338da51ec0388"


def _load(name: str, projection: bool | None) -> ProcessConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        config = load_process_config(CONFIG_DIR / name)
    if projection is None:
        return config
    return dataclasses.replace(
        config,
        numerics=dataclasses.replace(
            config.numerics, sample_time_s=25.0, enforce_gas_velocity_projection=projection
        ),
    )


def _with_limit(config: ProcessConfig, limit: float | None) -> ProcessConfig:
    return dataclasses.replace(
        config,
        material=dataclasses.replace(config.material, product_moisture_limit_kg_per_kg=limit),
    )


@pytest.mark.parametrize(("name", "projection"), sorted(FINGERPRINTS_BEFORE_S02, key=str))
def test_existing_configs_keep_their_fingerprints(name: str, projection: bool | None) -> None:
    assert _load(name, projection).fingerprint == FINGERPRINTS_BEFORE_S02[(name, projection)]


def test_material_section_fingerprint_is_unchanged_while_unset() -> None:
    config = _load("greybox_gab_placeholder.yaml", None)
    assert config.section_fingerprints["material"] == MATERIAL_SECTION_FINGERPRINT_GAB


def test_unset_limit_equals_critical_moisture() -> None:
    config = _load("greybox_gab_placeholder.yaml", None)
    assert config.material.product_moisture_limit_kg_per_kg is None
    assert (
        config.material.quality_moisture_limit_kg_per_kg
        == config.material.critical_moisture_kg_per_kg
    )


def test_set_limit_enters_the_fingerprint_and_only_the_material_section() -> None:
    base = _load("greybox_gab_placeholder.yaml", None)
    limited = _with_limit(base, 0.19)
    assert limited.material.quality_moisture_limit_kg_per_kg == 0.19
    assert limited.fingerprint != base.fingerprint
    assert limited.section_fingerprints["material"] != base.section_fingerprints["material"]
    for section in ("geometry", "gas", "energy", "actions", "disturbances", "numerics"):
        assert limited.section_fingerprints[section] == base.section_fingerprints[section]
    # Setting it to the critical value is still a set field and therefore hashed.
    same_value = _with_limit(base, base.material.critical_moisture_kg_per_kg)
    assert same_value.fingerprint != base.fingerprint


def test_limit_must_exceed_equilibrium_moisture() -> None:
    with pytest.raises(ValueError, match="product_moisture_limit_kg_per_kg"):
        MaterialConfig(equilibrium_moisture_kg_per_kg=0.003, product_moisture_limit_kg_per_kg=0.003)


@pytest.mark.parametrize("limit", [None, 0.15])
def test_dump_and_load_round_trip(tmp_path: Path, limit: float | None) -> None:
    config = _with_limit(_load("greybox_gab_placeholder.yaml", None), limit)
    path = tmp_path / "config.yaml"
    dump_process_config(config, path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        loaded = load_process_config(path)
    assert loaded.material.product_moisture_limit_kg_per_kg == limit
    assert loaded.fingerprint == config.fingerprint


def test_kinetics_are_unchanged_and_only_the_constraint_moves() -> None:
    """Same energy (kinetics untouched), but the constraint check uses the quality limit."""
    from twinlink_ml.simulation import GreyBoxFluidBedGranulator
    from twinlink_ml.simulation.operating_point import (
        DEFAULT_SEARCH_MAX_ITERATIONS,
        DEFAULT_SEARCH_TOLERANCE,
        build_variant_config,
        check_constraints,
    )
    from twinlink_ml.simulation.steady_state import solve_steady_state

    base = _load("greybox_gab_placeholder.yaml", None)
    variant = build_variant_config(
        base, t_in=75.0, u_gas=1.2, p_nozzle=1.0026, t_amb=20.0, phi_amb=0.5
    )
    tightened = _with_limit(variant, 0.05)

    solutions = [
        solve_steady_state(
            GreyBoxFluidBedGranulator(config),
            max_iterations=DEFAULT_SEARCH_MAX_ITERATIONS,
            tolerance=DEFAULT_SEARCH_TOLERANCE,
        )
        for config in (variant, tightened)
    ]
    assert solutions[0].diagnostics["energy_target"] == solutions[1].diagnostics["energy_target"]
    assert solutions[0].particle_moisture_kg_per_kg == solutions[1].particle_moisture_kg_per_kg

    moisture = solutions[0].particle_moisture_kg_per_kg
    assert moisture > 0.05  # the point must violate the tightened limit for this test
    for config, solution in zip((variant, tightened), solutions, strict=True):
        check = check_constraints(
            solution,
            critical_moisture_kg_per_kg=config.material.quality_moisture_limit_kg_per_kg,
            gas_velocity_m_per_s=1.2,
        )
        expected = max(0.0, moisture - config.material.quality_moisture_limit_kg_per_kg)
        assert check.moisture_violation_kg_per_kg == pytest.approx(expected)
    assert expected > 0.0


def test_bandit_env_reads_the_quality_limit(tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    from twinlink_ml.rl import BanditEnvConfig, OperatingPointBanditEnv

    path = tmp_path / "limited.yaml"
    dump_process_config(_with_limit(_load("greybox_gab_placeholder.yaml", None), 0.05), path)
    point = {
        "inlet_temperature_celsius": 75.0,
        "gas_velocity_m_per_s": 1.2,
        "ambient_temperature_celsius": 20.0,
        "ambient_relative_humidity": 0.5,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        default_env = OperatingPointBanditEnv()
        limited_env = OperatingPointBanditEnv(BanditEnvConfig(process_config_path=path))
    _, info_default = default_env.evaluate_operating_point(**point)
    _, info_limited = limited_env.evaluate_operating_point(**point)
    assert info_default["energy_target"] == info_limited["energy_target"]
    default_violation = info_default["constraint_check"]["moisture_violation_kg_per_kg"]
    limited_violation = info_limited["constraint_check"]["moisture_violation_kg_per_kg"]
    assert limited_violation > default_violation


def test_bandit_env_config_override_equals_a_config_file_limit(tmp_path: Path) -> None:
    pytest.importorskip("gymnasium")
    from twinlink_ml.rl import BanditEnvConfig, OperatingPointBanditEnv

    point = {
        "inlet_temperature_celsius": 75.0,
        "gas_velocity_m_per_s": 1.2,
        "ambient_temperature_celsius": 20.0,
        "ambient_relative_humidity": 0.5,
    }
    path = tmp_path / "limited.yaml"
    dump_process_config(_with_limit(_load("greybox_gab_placeholder.yaml", None), 0.05), path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlaceholderConfigWarning)
        from_file = OperatingPointBanditEnv(BanditEnvConfig(process_config_path=path))
        from_override = OperatingPointBanditEnv(
            BanditEnvConfig(product_moisture_limit_kg_per_kg=0.05)
        )
        default = OperatingPointBanditEnv()
    r_file, info_file = from_file.evaluate_operating_point(**point)
    r_over, info_over = from_override.evaluate_operating_point(**point)
    assert r_file == r_over
    assert info_file["constraint_check"] == info_over["constraint_check"]
    assert from_override._base_config.material.critical_moisture_kg_per_kg == 0.20
    assert default._base_config.material.product_moisture_limit_kg_per_kg is None
    assert from_override._base_config.fingerprint == from_file._base_config.fingerprint
