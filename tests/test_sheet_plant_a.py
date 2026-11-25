"""Process 1 of the sweep sheet against measured points: mapping, roots, readings, records.

Synthetic data only: a laboratory-scale line that is over-saturated by construction stands for
the sheet's process, invented segment means stand for the measured days. No plant value, no
value of a partner sheet.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import Any

import pytest

from twinlink_ml.dyssol_steady import plant_check as pc
from twinlink_ml.dyssol_steady.materials import (
    DEFAULT_HEAT_CAPACITY,
    DEFAULT_MOLAR_MASS,
    DyssolProperties,
)
from twinlink_ml.dyssol_steady.reference import LineInputs

PROPS = DyssolProperties(
    cp_gas=1006.0,
    cp_vapour=1906.0,
    cp_liquid=4185.0,
    cp_solid=DEFAULT_HEAT_CAPACITY,
    heat_of_vaporization_j_per_kg=40660.5 / DEFAULT_MOLAR_MASS,
    boiling_point_k=373.15,
    gas_enthalpy_linear=(1005.0, 0.0),
    vapour_enthalpy_shomate=(
        1670.36,
        0.379262,
        0.000377093,
        -1.40685e-07,
        4.5594e06,
        -1.3926e07,
        -1.34234e07,
    ),
    gas_enthalpy_range_k=(200.0, 2773.0),
    vapour_enthalpy_range_k=(290.0, 1700.0),
)
SHEET_LINE = LineInputs(
    dry_gas_kg_per_h=123.4,
    gas_humidity_g_per_kg=6.1,
    gas_celsius=11.3,
    dryer_inlet_celsius=81.7,
    spray_kg_per_h=6.9,
    solid_fraction=0.41,
    spray_celsius=29.1,
    medium_kg_per_h=0.0,
    equilibrium_moisture_g_per_kg=31.0,
    heater_efficiency_percent=79.0,
    calorific_value_kj_per_kg=44100.0,
)
DENSITY = 1.2922
SOLIDS = {"0.65": 0.65, "0.6": 0.6, pc.SHEET: SHEET_LINE.solid_fraction}
HUMIDITIES = {pc.HUMIDITY_REFERENCE: 0.024, pc.SHEET: SHEET_LINE.gas_humidity_g_per_kg}
REFERENCE = ("0.65", pc.HUMIDITY_REFERENCE)


def _point(label: str, air: float, supply: float, spray: float, exhaust: float) -> pc.MeasuredPoint:
    return pc.MeasuredPoint(
        label=label,
        process_air_nm3h=air,
        supply_celsius=supply,
        spray_kg_h=spray,
        ambient_celsius=4.7,
        exhaust_celsius=exhaust,
        bed_celsius=exhaust + 7.3,
    )


SEGMENTS = {
    "day_1": [
        _point("day_1_s1", 101.3, 85.9, 3.1, 60.7),
        _point("day_1_s2", 102.9, 86.3, 3.3, 61.9),
    ],
    "day_2": [
        _point("day_2_s1", 97.1, 83.3, 6.1, 33.7),
        _point("day_2_s2", 96.3, 83.9, 6.3, 34.9),
    ],
}


def _study() -> tuple[dict[str, LineInputs], dict[str, Any], dict[str, Any], dict[str, Any]]:
    points = pc.build_points(SHEET_LINE, SEGMENTS, SOLIDS, HUMIDITIES, REFERENCE, DENSITY)
    start = points[pc.SHEET_POINT]
    roots = {
        "spray_kg_per_h": pc.admissibility_root(
            PROPS, start, "spray_kg_per_h", 1e-6 * start.spray_kg_per_h, start.spray_kg_per_h
        ),
        "solid_fraction": pc.admissibility_root(
            PROPS, start, "solid_fraction", start.solid_fraction, 0.99
        ),
        "dryer_inlet_celsius": pc.admissibility_root(
            PROPS, start, "dryer_inlet_celsius", start.dryer_inlet_celsius, 231.7
        ),
    }
    outputs = {tag: pc.rebuild(PROPS, line) for tag, line in points.items()}
    minimum = {
        humidity: {
            p.label: pc.admissibility_root(
                PROPS, points[pc.point_tag(p.label, "0.65", humidity)], "solid_fraction", 0.01, 0.99
            )
            for pts in SEGMENTS.values()
            for p in pts
        }
        for humidity in HUMIDITIES
    }
    repo, local = pc.summarise(
        points=points,
        outputs=outputs,
        segments=SEGMENTS,
        solids=SOLIDS,
        humidities=HUMIDITIES,
        reference=REFERENCE,
        threshold_k=2.0,
        roots=roots,
        minimum_solid=minimum,
        predictions={"F": "F0", "T": "T1:unit warmer", "A": "A0:solid_fraction"},
    )
    return points, roots, repo, local


def test_the_synthetic_sheet_line_is_oversaturated() -> None:
    assert pc.saturation_ratio(PROPS, SHEET_LINE) > 1.0


def test_unit_inputs_take_the_measured_values_and_keep_the_rest_of_the_template() -> None:
    point = SEGMENTS["day_1"][0]
    line = pc.unit_inputs(
        point,
        replace(SHEET_LINE, medium_kg_per_h=500.0),
        solid_fraction=0.65,
        humidity_g_per_kg=0.024,
        normal_density_kg_per_m3=DENSITY,
    )
    assert line.dry_gas_kg_per_h == pytest.approx(point.process_air_nm3h * DENSITY)
    assert line.dryer_inlet_celsius == point.supply_celsius
    assert line.spray_kg_per_h == point.spray_kg_h
    assert line.gas_celsius == point.ambient_celsius
    assert (line.solid_fraction, line.gas_humidity_g_per_kg, line.medium_kg_per_h) == (
        0.65,
        0.024,
        0.0,
    )
    for name in (
        "spray_celsius",
        "equilibrium_moisture_g_per_kg",
        "heater_efficiency_percent",
        "calorific_value_kj_per_kg",
    ):
        assert getattr(line, name) == getattr(SHEET_LINE, name)


def test_measured_point_from_statistics_and_day_aggregate() -> None:
    statistics = {
        "process_air_flow": {"mean": 101.3, "std": 0.1},
        "supply_air_temperature": {"mean": 85.9, "std": 0.1},
        "spray_rate": {"mean": 3.1, "std": 0.1},
        "ambient_temperature": {"mean": 4.7, "std": 0.1},
        "exhaust_outlet_temperature": {"mean": 60.7, "std": 0.1},
        "bed_temperature": {"mean": 68.0, "std": 0.1},
        "atomising_air_flow": {"mean": 2.0, "std": 0.1},
    }
    point = pc.MeasuredPoint.from_statistics("day_1_s1", statistics)
    assert point == SEGMENTS["day_1"][0]
    day = pc.mean_point("day_1", SEGMENTS["day_1"])
    assert day.label == "day_1"
    assert day.process_air_nm3h == pytest.approx((101.3 + 102.9) / 2.0)
    assert day.exhaust_celsius == pytest.approx((60.7 + 61.9) / 2.0)
    with pytest.raises(ValueError, match="no point"):
        pc.mean_point("empty", [])


def test_water_load_is_the_evaporated_water_per_kg_of_dry_gas() -> None:
    solid = 6.9 * 0.41
    expected = 1000.0 * (6.9 - solid - solid * 0.031) / 123.4
    assert pc.water_load_g_per_kg(SHEET_LINE) == pytest.approx(expected)
    out = pc.rebuild(PROPS, SHEET_LINE)
    assert out.dryer_outlet_humidity_g_per_kg - SHEET_LINE.gas_humidity_g_per_kg == pytest.approx(
        expected
    )


def test_points_cover_the_sheet_every_variant_and_every_single_replacement() -> None:
    points, _, _, _ = _study()
    measured = [tag for tag in points if tag.startswith("M|")]
    assert len(measured) == 4 * len(SOLIDS) * len(HUMIDITIES)
    assert points[pc.SHEET_POINT] == SHEET_LINE
    for day in SEGMENTS:
        for name in pc.REPLACED:
            line = points[pc.replacement_tag(day, name)]
            changed = [f for f in asdict(line) if getattr(line, f) != getattr(SHEET_LINE, f)]
            assert changed == [name]
    day_1 = pc.mean_point("day_1", SEGMENTS["day_1"])
    assert points[pc.replacement_tag("day_1", "spray_kg_per_h")].spray_kg_per_h == day_1.spray_kg_h
    assert points[pc.replacement_tag("day_2", "solid_fraction")].solid_fraction == 0.65
    assert (
        points[pc.replacement_tag("day_2", "gas_humidity_g_per_kg")].gas_humidity_g_per_kg == 0.024
    )


@pytest.mark.parametrize(
    ("name", "lower", "upper"),
    [
        ("spray_kg_per_h", 1e-5, SHEET_LINE.spray_kg_per_h),
        ("solid_fraction", SHEET_LINE.solid_fraction, 0.99),
        ("dryer_inlet_celsius", SHEET_LINE.dryer_inlet_celsius, 231.7),
    ],
)
def test_root_lies_on_the_admissible_side_of_saturation(
    name: str, lower: float, upper: float
) -> None:
    root = pc.admissibility_root(PROPS, SHEET_LINE, name, lower, upper)
    assert root is not None
    assert lower < root < upper
    ratio = pc.saturation_ratio(PROPS, replace(SHEET_LINE, **{name: root}))
    assert ratio <= 1.0
    assert ratio == pytest.approx(1.0, abs=1e-9)


def test_root_is_none_when_both_ends_lie_on_the_same_side() -> None:
    assert pc.admissibility_root(PROPS, SHEET_LINE, "solid_fraction", 0.41, 0.42) is None


def test_reading_f() -> None:
    assert pc.reading_f([False, False]) == "F0"
    assert pc.reading_f([True, False]) == "F1"
    assert pc.reading_f([True, True]) == "F2"
    with pytest.raises(ValueError, match="at least one"):
        pc.reading_f([])


def test_reading_t() -> None:
    assert pc.reading_t([1.9, -2.0], 2.0) == "T0"
    assert pc.reading_t([1.9, 2.1], 2.0) == "T1"
    assert pc.reading_t([], 2.0) == "not applicable"


def test_reading_a() -> None:
    one = pc.reading_a({"a": [True, True], "b": [True, False], "c": [False, False]})
    assert (one["code"], one["named_input"]) == ("A0", "a")
    assert one["removes_on_some_day_only"] == ["b"]
    two = pc.reading_a({"a": [True, True], "b": [True, True]})
    assert (two["code"], two["named_input"]) == ("A1", None)
    partly = pc.reading_a({"a": [True, False], "b": [False, False]})
    assert partly["code"] == "A1"
    none = pc.reading_a({"a": [False, False], "b": [False, False]})
    assert none["code"] == "A2"


def test_control_deviation_counts_missing_points_and_finds_a_shifted_output() -> None:
    points, _, _, _ = _study()
    outputs = {tag: pc.rebuild(PROPS, line) for tag, line in points.items()}
    same = pc.control_deviation(dict(outputs), outputs)
    assert same["points_compared"] == same["points"] == len(points)
    assert same["max_temperature_k"] == 0.0
    assert same["max_relative_percent"] == 0.0
    shifted: dict[str, Any] = dict(outputs)
    shifted[pc.SHEET_POINT] = replace(
        outputs[pc.SHEET_POINT],
        dryer_outlet_celsius=outputs[pc.SHEET_POINT].dryer_outlet_celsius + 0.25,
    )
    first = next(tag for tag in points if tag.startswith("M|"))
    shifted[first] = None
    result = pc.control_deviation(shifted, outputs)
    assert result["points_compared"] == len(points) - 1
    assert result["max_temperature_k"] == pytest.approx(0.25)


def test_readings_of_the_synthetic_study() -> None:
    _, roots, repo, local = _study()
    assert repo["reading_f"]["code"] == "F0"
    assert repo["reading_f"]["oversaturated_per_day"] == {"day_1": 0, "day_2": 0}
    assert set(repo["reading_t"]["unit_minus_exhaust_k"]) == {
        "day_1_s1",
        "day_1_s2",
        "day_2_s1",
        "day_2_s2",
    }
    assert repo["reading_a"]["removed"]["solid_fraction"] == {"day_1": True, "day_2": True}
    assert repo["reading_a"]["removed"]["spray_kg_per_h"] == {"day_1": True, "day_2": False}
    assert repo["reading_a"]["code"] == "A0"
    assert repo["predictions"]["F"]["confirmed"] is True
    assert repo["predictions"]["A"]["found"] == "A0:solid_fraction"
    distance = repo["without_reading"]["distance_to_admissibility"]
    assert distance["spray_relative_change_percent"] < 0.0
    assert distance["dryer_inlet_change_k"] > 0.0
    assert distance["solid_fraction_root_exists"] is True
    assert local["roots"] == roots
    assert local["points"][pc.SHEET_POINT] == asdict(SHEET_LINE)


def _numbers(node: Any) -> list[float]:
    if isinstance(node, dict):
        return [x for value in node.values() for x in _numbers(value)]
    if isinstance(node, list | tuple):
        return [x for value in node for x in _numbers(value)]
    if isinstance(node, bool) or not isinstance(node, int | float):
        return []
    return [float(node)]


def test_repository_record_holds_no_input_of_the_sheet_and_no_measured_value() -> None:
    points, roots, repo, _ = _study()
    json.dumps(repo)  # serialisable as it is
    forbidden = {float(v) for v in asdict(SHEET_LINE).values()} - {0.0}
    forbidden |= {
        float(v)
        for pts in SEGMENTS.values()
        for p in pts
        for v in asdict(p).values()
        if not isinstance(v, str)
    }
    forbidden |= {float(v) for v in roots.values() if v is not None}
    forbidden |= {points[tag].dry_gas_kg_per_h for tag in points}
    found = [x for x in _numbers(repo) if any(abs(x - f) <= 1e-12 * abs(f) for f in forbidden)]
    assert found == []
    for key, entry in repo["without_reading"]["variants"].items():
        depends_on_sheet = pc.SHEET in key.split("|")
        assert ("saturation_ratio" in entry) is not depends_on_sheet
        assert ("unit_minus_exhaust_k" in entry) is not depends_on_sheet
    assert "saturation_ratio" not in json.dumps(repo["reading_a"])
    minimum = repo["without_reading"]["minimum_solid_fraction_of_the_unit"]
    assert set(minimum[pc.HUMIDITY_REFERENCE]) == {
        "segments",
        "segments_without_root",
        "segments_with_minimum_above_the_sheet_value",
    }
