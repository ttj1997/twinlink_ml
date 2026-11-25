"""Process 1 of the partner sweep sheet against measured stationary points of a plant line.

Design and pre-registered reading: ``docs/NN_sheet_plant_a.md`` (docstring of
``scripts/report_sheet_plant_a.py``).

The module turns measured stationary points into inputs of the one-line units
(:func:`unit_inputs`), builds every point of the study (:func:`build_points`), finds the value
of one input at which the dryer exhaust of the unit is exactly saturated
(:func:`admissibility_root`), applies the three readings (:func:`reading_f`, :func:`reading_t`,
:func:`reading_a`) and assembles two records (:func:`summarise`): one for the repository and one
that stays local.

**What the repository record may hold** (decision of the project lead, 2026-11-25): saturation
ratios, counts, temperature differences in K, relative changes, flags. Whatever depends on an
input of the sheet is reduced to counts, flags and reading codes; the exceptions are the
saturation ratio of the sheet's own point (already in the artifact of Befund 130) and the two
relative distances to admissibility. No input of the sheet, no measured value, no ratio "sheet
over day" of an input.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from .materials import DyssolProperties
from .reference import LineInputs, LineOutputs, evaluate

__all__ = [
    "CONTROL_RELATIVE",
    "CONTROL_TEMPERATURES",
    "HUMIDITY_REFERENCE",
    "MEASURED_INPUTS",
    "REPLACED",
    "SHEET",
    "MeasuredPoint",
    "admissibility_root",
    "build_points",
    "control_deviation",
    "mean_point",
    "point_tag",
    "reading_a",
    "reading_f",
    "reading_t",
    "rebuild",
    "replacement_tag",
    "saturation_ratio",
    "summarise",
    "unit_inputs",
    "water_load_g_per_kg",
]

REPLACED: tuple[str, ...] = (
    "dry_gas_kg_per_h",
    "dryer_inlet_celsius",
    "spray_kg_per_h",
    "solid_fraction",
    "gas_humidity_g_per_kg",
)
"""Inputs of the sheet's line that reading A replaces one at a time by the value of a day."""
MEASURED_INPUTS: tuple[str, ...] = ("dry_gas_kg_per_h", "dryer_inlet_celsius", "spray_kg_per_h")
"""The replaced inputs that are measured at the plant (the other two are placeholders)."""
SHEET = "sheet"
"""Label of a variant that takes the value of the sheet."""
HUMIDITY_REFERENCE = "dried_supply_air"
"""Label of the reference humidity (the dried-supply-air placeholder of Befund 117)."""
SHEET_POINT = "S"
CONTROL_TEMPERATURES = ("dryer_outlet_celsius", "exhaust_celsius", "preheated_celsius")
CONTROL_RELATIVE = ("heat_recovered_kw", "heater_duty_kw", "fuel_kg_per_h")
_STATISTICS = {
    "process_air_nm3h": "process_air_flow",
    "supply_celsius": "supply_air_temperature",
    "spray_kg_h": "spray_rate",
    "ambient_celsius": "ambient_temperature",
    "exhaust_celsius": "exhaust_outlet_temperature",
    "bed_celsius": "bed_temperature",
}


@dataclass(frozen=True)
class MeasuredPoint:
    """Means of one stationary segment (or of a day): degC, air in Nm3/h, spray in kg/h."""

    label: str
    process_air_nm3h: float
    supply_celsius: float
    spray_kg_h: float
    ambient_celsius: float
    exhaust_celsius: float
    bed_celsius: float

    @classmethod
    def from_statistics(cls, label: str, statistics: Mapping[str, Any]) -> MeasuredPoint:
        """From the result of :func:`twinlink_ml.plant_scale.data.segment_statistics`."""
        return cls(label, **{f: float(statistics[role]["mean"]) for f, role in _STATISTICS.items()})


def mean_point(label: str, points: Sequence[MeasuredPoint]) -> MeasuredPoint:
    """Day aggregate: the mean of every field over the segments of a day (as docs/34)."""
    if not points:
        raise ValueError("no point to average")
    return MeasuredPoint(
        label, **{f: math.fsum(getattr(p, f) for p in points) / len(points) for f in _STATISTICS}
    )


def unit_inputs(
    point: MeasuredPoint,
    template: LineInputs,
    *,
    solid_fraction: float,
    humidity_g_per_kg: float,
    normal_density_kg_per_m3: float,
) -> LineInputs:
    """Inputs of the unit at a measured point.

    Measured: process air (normal cubic metres of dry air are a mass, as Befund 117), supply-air
    temperature, spray rate, intake temperature. Not measured and therefore arguments: solid
    fraction of the spray and humidity of the gas. Spray temperature, equilibrium moisture,
    heater efficiency and calorific value stay those of ``template`` (the line of the sheet). No
    medium flow: the heat recovery is not part of the comparison.
    """
    return replace(
        template,
        dry_gas_kg_per_h=point.process_air_nm3h * normal_density_kg_per_m3,
        gas_humidity_g_per_kg=humidity_g_per_kg,
        gas_celsius=point.ambient_celsius,
        dryer_inlet_celsius=point.supply_celsius,
        spray_kg_per_h=point.spray_kg_h,
        solid_fraction=solid_fraction,
        medium_kg_per_h=0.0,
    )


def water_load_g_per_kg(line: LineInputs) -> float:
    """Water the dryer unit evaporates per kg of dry gas [g/kg]."""
    solid = line.spray_kg_per_h * line.solid_fraction
    left = solid * line.equilibrium_moisture_g_per_kg / 1000.0
    return 1000.0 * max(line.spray_kg_per_h - solid - left, 0.0) / line.dry_gas_kg_per_h


def rebuild(props: DyssolProperties, line: LineInputs) -> LineOutputs:
    """One line with its own heat recovery through the Python rebuild of the units."""
    return evaluate(props, [line], (0,))[0]


def saturation_ratio(props: DyssolProperties, line: LineInputs) -> float:
    """Humidity of the dryer exhaust over the saturation humidity at its temperature."""
    return rebuild(props, line).saturation_ratio


def admissibility_root(
    props: DyssolProperties,
    line: LineInputs,
    name: str,
    lower: float,
    upper: float,
    *,
    iterations: int = 200,
) -> float | None:
    """Value of the input ``name`` in ``[lower, upper]`` at which the saturation ratio is 1.

    Bisection on the sign of ``ratio - 1``; ``None`` if both ends lie on the same side.
    """

    def above(value: float) -> bool:
        return saturation_ratio(props, replace(line, **{name: value})) > 1.0

    low_above, high_above = above(lower), above(upper)
    if low_above == high_above:
        return None
    for _ in range(iterations):
        middle = 0.5 * (lower + upper)
        if middle in (lower, upper):
            break
        if above(middle) == low_above:
            lower = middle
        else:
            upper = middle
    # the end on the admissible side, so that the returned point is not over-saturated
    return upper if low_above else lower


# ------------------------------------------------------------------------------------ points


def point_tag(segment: str, solid: str, humidity: str) -> str:
    """Tag of a measured point in one variant."""
    return f"M|{segment}|{solid}|{humidity}"


def replacement_tag(day: str, name: str) -> str:
    """Tag of the sheet's point with one input replaced by the value of a day."""
    return f"R|{day}|{name}"


def build_points(
    sheet_line: LineInputs,
    segments: Mapping[str, Sequence[MeasuredPoint]],
    solids: Mapping[str, float],
    humidities: Mapping[str, float],
    reference: tuple[str, str],
    normal_density_kg_per_m3: float,
) -> dict[str, LineInputs]:
    """All points of the study by tag: S, the measured points M, the replacements R.

    ``segments``: day -> its stationary segments; ``solids`` / ``humidities``: label -> value of
    the variants; ``reference``: the labels of the reference variant, whose day aggregates
    provide the replaced values.
    """
    points: dict[str, LineInputs] = {SHEET_POINT: replace(sheet_line, medium_kg_per_h=0.0)}
    for day, day_points in segments.items():
        for point in day_points:
            for solid_label, solid in solids.items():
                for humidity_label, humidity in humidities.items():
                    points[point_tag(point.label, solid_label, humidity_label)] = unit_inputs(
                        point,
                        sheet_line,
                        solid_fraction=solid,
                        humidity_g_per_kg=humidity,
                        normal_density_kg_per_m3=normal_density_kg_per_m3,
                    )
        day_line = unit_inputs(
            mean_point(day, day_points),
            sheet_line,
            solid_fraction=solids[reference[0]],
            humidity_g_per_kg=humidities[reference[1]],
            normal_density_kg_per_m3=normal_density_kg_per_m3,
        )
        for name in REPLACED:
            points[replacement_tag(day, name)] = replace(
                points[SHEET_POINT], **{name: getattr(day_line, name)}
            )
    return points


# ------------------------------------------------------------------------------------ readings


def reading_f(oversaturated: Sequence[bool]) -> str:
    """F0 no segment over-saturated, F2 all of them, F1 otherwise."""
    if not oversaturated:
        raise ValueError("reading F needs at least one segment")
    if not any(oversaturated):
        return "F0"
    return "F2" if all(oversaturated) else "F1"


def reading_t(differences_k: Sequence[float], threshold_k: float) -> str:
    """T0 every difference within the threshold, T1 otherwise; without a segment not applicable."""
    if not differences_k:
        return "not applicable"
    return "T0" if all(abs(d) <= threshold_k for d in differences_k) else "T1"


def reading_a(removed: Mapping[str, Sequence[bool]]) -> dict[str, Any]:
    """Attribution: which single replacement removes the over-saturation.

    ``removed``: input -> one flag per day. A0 exactly one input removes it on every day; A1 more
    than one does, or none does on every day but at least one does on some day; A2 none on any day.
    """
    every_day = [name for name, flags in removed.items() if all(flags)]
    some_day = [name for name, flags in removed.items() if any(flags)]
    if len(every_day) == 1:
        code = "A0"
    elif every_day or some_day:
        code = "A1"
    else:
        code = "A2"
    return {
        "code": code,
        "removes_on_every_day": every_day,
        "removes_on_some_day_only": [n for n in some_day if n not in every_day],
        "named_input": every_day[0] if code == "A0" else None,
    }


def control_deviation(
    dyssol: Mapping[str, LineOutputs | None], reference: Mapping[str, LineOutputs]
) -> dict[str, Any]:
    """Largest deviation of Dyssol from the rebuild over the points that ran."""
    temperature = relative = 0.0
    compared = 0
    for tag, ours in reference.items():
        theirs = dyssol.get(tag)
        if theirs is None:
            continue
        compared += 1
        for name in CONTROL_TEMPERATURES:
            temperature = max(temperature, abs(getattr(theirs, name) - getattr(ours, name)))
        for name in CONTROL_RELATIVE:
            value = getattr(ours, name)
            relative = max(
                relative, 100.0 * abs(getattr(theirs, name) - value) / max(abs(value), 1.0)
            )
    return {
        "points": len(reference),
        "points_compared": compared,
        "max_temperature_k": temperature,
        "max_relative_percent": relative,
    }


# ------------------------------------------------------------------------------------ records


def _span(values: Sequence[float]) -> list[float] | None:
    return [min(values), max(values)] if values else None


def _variant(
    labels: Sequence[str],
    days: Mapping[str, Sequence[MeasuredPoint]],
    outputs: Mapping[str, LineOutputs],
    solid: str,
    humidity: str,
    threshold_k: float,
) -> dict[str, Any]:
    by_label = {p.label: p for pts in days.values() for p in pts}
    out = {label: outputs[point_tag(label, solid, humidity)] for label in labels}
    over = {label: o.oversaturated for label, o in out.items()}
    admissible = [label for label in labels if not over[label]]
    exhaust = {
        label: out[label].dryer_outlet_celsius - by_label[label].exhaust_celsius for label in labels
    }
    bed = {label: out[label].dryer_outlet_celsius - by_label[label].bed_celsius for label in labels}
    read = [exhaust[label] for label in admissible]
    return {
        "segments": len(labels),
        "oversaturated_segments": sum(over.values()),
        "oversaturated_per_day": {
            day: sum(over[p.label] for p in pts) for day, pts in days.items()
        },
        "reading_f": reading_f(list(over.values())),
        "reading_t": reading_t(read, threshold_k),
        "unit_warmer_at_every_read_segment": bool(read) and all(d > 0.0 for d in read),
        "saturation_ratio": {label: o.saturation_ratio for label, o in out.items()},
        "unit_minus_exhaust_k": exhaust,
        "unit_minus_bed_k": bed,
        "unit_minus_exhaust_k_per_day": {
            day: _span([exhaust[p.label] for p in pts if not over[p.label]])
            for day, pts in days.items()
        },
    }


_COUNTS_ONLY = (
    "segments",
    "oversaturated_segments",
    "oversaturated_per_day",
    "reading_f",
    "reading_t",
    "unit_warmer_at_every_read_segment",
)


def summarise(
    *,
    points: Mapping[str, LineInputs],
    outputs: Mapping[str, LineOutputs],
    segments: Mapping[str, Sequence[MeasuredPoint]],
    solids: Mapping[str, float],
    humidities: Mapping[str, float],
    reference: tuple[str, str],
    threshold_k: float,
    roots: Mapping[str, float | None],
    minimum_solid: Mapping[str, Mapping[str, float | None]],
    predictions: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Readings and reported numbers: the record for the repository and the local record.

    ``outputs``: the rebuild at every tag of ``points``; ``roots``: per input of the sheet's line
    the value at which its saturation ratio is 1 (or ``None``); ``minimum_solid``: per humidity
    label and segment the smallest solid fraction at which the unit is admissible;
    ``predictions``: reading -> predicted code (``A`` as ``"A0:<input>"``).
    """
    labels = [p.label for pts in segments.values() for p in pts]
    days = list(segments)
    sheet_line = points[SHEET_POINT]
    variants = {
        (s, h): _variant(labels, segments, outputs, s, h, threshold_k)
        for s in solids
        for h in humidities
    }
    ref = variants[reference]

    ratio = {
        name: {day: outputs[replacement_tag(day, name)].saturation_ratio for day in days}
        for name in REPLACED
    }
    removed = {name: {day: ratio[name][day] <= 1.0 for day in days} for name in REPLACED}
    attribution = reading_a({name: list(flags.values()) for name, flags in removed.items()})

    found_a = attribution["code"] + (
        f":{attribution['named_input']}" if attribution["named_input"] else ""
    )
    found_t = ref["reading_t"] + (
        ":unit warmer"
        if ref["reading_t"] == "T1" and ref["unit_warmer_at_every_read_segment"]
        else ""
    )
    found = {"F": ref["reading_f"], "T": found_t, "A": found_a}
    checks = {
        key: {
            "predicted": predictions[key],
            "found": found[key],
            "confirmed": predictions[key] == found[key],
        }
        for key in ("F", "T", "A")
    }

    sheet_solid = sheet_line.solid_fraction
    minimum_counts = {
        humidity: {
            "segments": len(values),
            "segments_without_root": sum(v is None for v in values.values()),
            "segments_with_minimum_above_the_sheet_value": sum(
                v is not None and v > sheet_solid for v in values.values()
            ),
        }
        for humidity, values in minimum_solid.items()
    }
    spray_root, inlet_root = roots.get("spray_kg_per_h"), roots.get("dryer_inlet_celsius")
    distance = {
        "spray_relative_change_percent": None
        if spray_root is None
        else 100.0 * (spray_root - sheet_line.spray_kg_per_h) / sheet_line.spray_kg_per_h,
        "dryer_inlet_change_k": None
        if inlet_root is None
        else inlet_root - sheet_line.dryer_inlet_celsius,
        "solid_fraction_root_exists": roots.get("solid_fraction") is not None,
    }

    def variant_record(key: tuple[str, str], *, local: bool) -> dict[str, Any]:
        entry = variants[key]
        if local or SHEET not in key:
            return dict(entry)
        return {name: entry[name] for name in _COUNTS_ONLY}

    repo: dict[str, Any] = {
        "test_set": {
            "days": {day: len(pts) for day, pts in segments.items()},
            "segments": labels,
            "reference_variant": {"solid_fraction": reference[0], "humidity": reference[1]},
        },
        "sheet_point": {"saturation_ratio": outputs[SHEET_POINT].saturation_ratio},
        "reading_f": {
            "code": ref["reading_f"],
            "segments": ref["segments"],
            "oversaturated_segments": ref["oversaturated_segments"],
            "oversaturated_per_day": ref["oversaturated_per_day"],
        },
        "reading_t": {
            "threshold_k": threshold_k,
            "code": ref["reading_t"],
            "unit_minus_exhaust_k": {
                label: ref["unit_minus_exhaust_k"][label]
                for label in labels
                if ref["saturation_ratio"][label] <= 1.0
            },
            "unit_minus_exhaust_k_per_day": ref["unit_minus_exhaust_k_per_day"],
            "unit_warmer_at_every_read_segment": ref["unit_warmer_at_every_read_segment"],
        },
        "reading_a": {**attribution, "removed": removed, "measured_inputs": list(MEASURED_INPUTS)},
        "predictions": checks,
        "without_reading": {
            "variants": {
                f"{s}|{h}": variant_record((s, h), local=False) for s in solids for h in humidities
            },
            "minimum_solid_fraction_of_the_unit": minimum_counts,
            "distance_to_admissibility": distance,
        },
    }
    local: dict[str, Any] = {
        "points": {tag: asdict(line) for tag, line in points.items()},
        "outputs": {tag: asdict(out) for tag, out in outputs.items()},
        "measured": {day: [asdict(p) for p in pts] for day, pts in segments.items()},
        "day_aggregates": {day: asdict(mean_point(day, pts)) for day, pts in segments.items()},
        "solid_fractions": dict(solids),
        "humidities_g_per_kg": dict(humidities),
        "water_load_g_per_kg": {tag: water_load_g_per_kg(line) for tag, line in points.items()},
        "variants": {
            f"{s}|{h}": variant_record((s, h), local=True) for s in solids for h in humidities
        },
        "reading_a_saturation_ratio": ratio,
        "roots": dict(roots),
        "minimum_solid_fraction_of_the_unit": {h: dict(v) for h, v in minimum_solid.items()},
    }
    return repo, local
