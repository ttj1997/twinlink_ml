"""Inputs of ``Model_v4.1.xlsx``: the plant inputs of v4 plus the two circuit mass flows.

The cell layout of the sheet ``Inputs`` is the one of ``Model_v4.xlsx`` for the dryers, the
material properties and the ambient conditions, so the reader of :mod:`twinlink_ml.partner_v4`
is reused. New in v4.1 are ``Inputs!E20`` and ``E21`` (mass flow of the heat transfer medium of
circuit 1 and 2; circuit *i* belongs to the exhaust of dryer *i*, wherever its heat goes) and
column ``K`` (not referenced by any formula, see :mod:`.deviations`).

``cp`` of the heat transfer medium is ``Inputs!I12`` and is part of the v4 material inputs
(``cp_heating_medium_kj_per_kg_k``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from twinlink_ml.partner_v4.inputs import ExcelInputs, PartnerV4Inputs, load_excel_inputs

__all__ = ["ExcelInputsV41", "SiInputsV41", "load_excel_inputs_v41"]


@dataclass(frozen=True)
class SiInputsV41:
    """Plant inputs in SI plus the circuits: mass flows in kg/s, ``cp`` in J/(kg K)."""

    plant: PartnerV4Inputs
    htm_flow_kg_per_s: tuple[float, float]
    cp_htm_j_per_kg_k: float


@dataclass(frozen=True)
class ExcelInputsV41:
    """The sheet ``Inputs`` in the sheet's units (kg/h, g/kg, kJ, percent, atm)."""

    plant: ExcelInputs
    htm_flow_kg_per_h: tuple[float, float]
    """``Inputs!E20``, ``E21``."""

    def to_si(self) -> SiInputsV41:
        """Convert to SI (the only place where the circuits' units are converted)."""
        return SiInputsV41(
            plant=self.plant.to_si(),
            htm_flow_kg_per_s=(
                self.htm_flow_kg_per_h[0] / 3600.0,
                self.htm_flow_kg_per_h[1] / 3600.0,
            ),
            cp_htm_j_per_kg_k=self.plant.material.cp_heating_medium_kj_per_kg_k * 1e3,
        )

    def with_htm_flow(self, kg_per_h: tuple[float, float]) -> ExcelInputsV41:
        """Same inputs with other circuit mass flows (the flows are adjustable)."""
        return ExcelInputsV41(self.plant, kg_per_h)


def load_excel_inputs_v41(path: str | Path) -> ExcelInputsV41:
    """Read the sheet ``Inputs`` of ``Model_v4.1.xlsx`` (needs ``openpyxl``)."""
    import openpyxl

    sheet = openpyxl.load_workbook(Path(path), data_only=True)["Inputs"]
    flows = []
    for address in ("E20", "E21"):
        value = sheet[address].value
        if not isinstance(value, int | float):
            raise ValueError(f"Inputs!{address} is not numeric: {value!r}")
        flows.append(float(value))
    return ExcelInputsV41(load_excel_inputs(path), (flows[0], flows[1]))
