"""Provenance block of the scripts (Befund 79): solver bound, dT_min check, network fingerprint."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from twinlink_ml.evaluation.site_network import SiteSpec
from twinlink_ml.simulation.steady_state import BED_TEMPERATURE_BRACKET_RULE

_COMMON = Path(__file__).resolve().parents[1] / "scripts" / "verify" / "_common.py"


def _common() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_common_under_test", _COMMON)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_provenance_names_solver_bound_and_heat_network() -> None:
    common = _common()
    site = SiteSpec(271.797)
    block = common.provenance(site)
    assert block["solver/bed_temperature_bracket"] == BED_TEMPERATURE_BRACKET_RULE
    assert block["heatnetwork/approach_check"] == "along_exchanger"
    assert block["heatnetwork/fingerprint"] == site.fingerprint
    legacy = common.provenance(SiteSpec(271.797, approach_check="warm_end"))
    assert legacy["heatnetwork/approach_check"] == "warm_end"
    assert legacy["heatnetwork/fingerprint"] != block["heatnetwork/fingerprint"]


def test_provenance_without_or_unrecorded_heat_network() -> None:
    common = _common()
    none = common.provenance(common.NO_HEAT_NETWORK)
    assert none["heatnetwork/approach_check"] == none["heatnetwork/fingerprint"] == "not used"
    old = common.provenance()
    assert old["heatnetwork/approach_check"] is None and old["heatnetwork/fingerprint"] is None
    assert old["solver/bed_temperature_bracket"] == BED_TEMPERATURE_BRACKET_RULE
