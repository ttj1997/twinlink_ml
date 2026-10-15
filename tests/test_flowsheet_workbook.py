"""The flowsheet with every switch ``"partner"`` against the workbook ``Model_v4.1.xlsx``.

Needs ``Model v4.1.xlsx`` (also accepted: ``Model_v4_1.xlsx``) from ``local/`` (confidential,
never committed; ``TWINLINK_LOCAL_DIR`` names the directory, e.g. in a worktree). Skipped when the
file is missing. Tolerance as in ``test_partner_v4_1``: ``rel=1e-12, abs=1e-12`` for the
as-built reproduction of every stored number.

``gas_formula="partner"`` is the formula of the sheet as built (108.9 times too high); with the
corrected formula only rows 26, 39 and 41 differ, by exactly that factor.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from twinlink_ml.flowsheet import Flowsheet, FlowsheetConfig, Switches
from twinlink_ml.flowsheet.sheet import sheet_scenario_cells
from twinlink_ml.partner_v4_1 import deviations as dev
from twinlink_ml.partner_v4_1.inputs import load_excel_inputs_v41

LOCAL = Path(os.environ.get("TWINLINK_LOCAL_DIR", Path(__file__).resolve().parents[1] / "local"))
WORKBOOK = next(
    (LOCAL / n for n in ("Model v4.1.xlsx", "Model_v4_1.xlsx") if (LOCAL / n).exists()), None
)


@pytest.fixture(scope="module")
def workbook():  # type: ignore[no-untyped-def]
    if WORKBOOK is None:
        pytest.skip("local/Model v4.1.xlsx not available")
    pytest.importorskip("openpyxl")
    return load_excel_inputs_v41(WORKBOOK), WORKBOOK


def test_all_partner_reproduces_every_stored_number_of_scenarios(workbook) -> None:  # type: ignore[no-untyped-def]
    """Scenarios 1-6, every numeric cell of the sheet ``Scenarios`` (230 cells), as built."""
    inputs, path = workbook
    cells = sheet_scenario_cells(Flowsheet(FlowsheetConfig(inputs.to_si(), Switches.all_partner())))
    excel = dev.read_formula_cells(path, "Scenarios")
    assert set(cells) == set(excel)
    for address, (_, stored) in excel.items():
        assert cells[address] == pytest.approx(stored, rel=1e-12, abs=1e-12), address


def test_with_the_corrected_gas_formula_only_the_gas_rows_differ(workbook) -> None:  # type: ignore[no-untyped-def]
    inputs, path = workbook
    fixed = Switches(gas_formula="physical")
    cells = sheet_scenario_cells(Flowsheet(FlowsheetConfig(inputs.to_si(), fixed)))
    excel = dev.read_formula_cells(path, "Scenarios")
    differing = {
        a for a, (_, v) in excel.items() if cells[a] != pytest.approx(v, rel=1e-9, abs=1e-12)
    }
    assert {int(re.sub(r"[A-Z]+", "", a)) for a in differing} == {26, 39, 41}
    for address in differing:
        assert excel[address][1] / cells[address] == pytest.approx(13.2 / 1.212 * 10.0, rel=1e-9)
