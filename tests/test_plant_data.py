"""Tests für ``plant_data`` -- Messstellenzuordnung und Excel-Import."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import openpyxl
import pytest

from twinlink_ml.plant_data import (
    GRANULATOR_D3204_CHANNELS,
    ambient_temperature_tag,
    bed_temperature_tags,
    channels_by_tag,
    inlet_temperature_tags,
    load_plant_export,
    outlet_temperature_tag,
    process_air_flow_tag,
    spray_rate_tag,
    upper_position_temperature_tags,
)


def test_channel_tags_unique() -> None:
    tags = [channel.tag for channel in GRANULATOR_D3204_CHANNELS]
    assert len(tags) == len(set(tags))


def test_semantic_accessors_resolve_against_mapping() -> None:
    by_tag = channels_by_tag()
    accessor_tags = (
        process_air_flow_tag(),
        ambient_temperature_tag(),
        *inlet_temperature_tags(),
        *bed_temperature_tags(),
        *upper_position_temperature_tags(),
        outlet_temperature_tag(),
        spray_rate_tag(),
    )
    for tag in accessor_tags:
        assert tag in by_tag


def test_bed_temperature_confirmed_upper_position_not() -> None:
    by_tag = channels_by_tag()
    assert all(by_tag[tag].confirmed for tag in bed_temperature_tags())
    assert all(not by_tag[tag].confirmed for tag in upper_position_temperature_tags())


def test_flow_channels_carry_corrected_unit_not_celsius() -> None:
    by_tag = channels_by_tag()
    for tag in ("FIRAL-323805", "FIRAL-323604"):
        assert by_tag[tag].unit == "m3/h"


def _write_sample_export(path: Path) -> None:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.append(["Time", "TIR-322412", "FIRAL-323604"])
    worksheet.append([None, "[°C]", "[°C]"])
    worksheet.append([dt.datetime(2026, 3, 26, 0, 0), 4.9, 36.6])
    worksheet.append([dt.datetime(2026, 3, 26, 0, 1), 4.8, 36.7])
    workbook.save(path)


def test_load_plant_export_reads_columns_and_overrides_mislabelled_unit(
    tmp_path: Path,
) -> None:
    export_path = tmp_path / "sample.xlsx"
    _write_sample_export(export_path)

    series = load_plant_export(export_path)

    assert len(series.timestamps) == 2
    assert series.columns["TIR-322412"] == pytest.approx((4.9, 4.8))
    assert series.units["TIR-322412"] == "°C"
    # Quelle beschriftet FIRAL-323604 faelschlich als [°C] -- Loader korrigiert auf m3/h.
    assert series.units["FIRAL-323604"] == "m3/h"


def test_load_plant_export_rejects_missing_time_column(tmp_path: Path) -> None:
    export_path = tmp_path / "bad.xlsx"
    workbook = openpyxl.Workbook()
    workbook.active.append(["Not-Time", "X"])
    workbook.save(export_path)

    with pytest.raises(ValueError, match="Time"):
        load_plant_export(export_path)
