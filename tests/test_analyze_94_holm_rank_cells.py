"""Holm counts of the rank cells (addendum to Befund 94) on synthetic cells with known p values."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "verify"))

import analyze_94_holm_rank_cells as holm_cells  # noqa: E402


def _cell(candidate: str, channel: str, p: float, verdict: str, horizon: int = 36):
    return {
        "candidate": candidate,
        "channel": channel,
        "p_value": p,
        "verdict": verdict,
        "horizon": horizon,
    }


def _cells():
    cells = [_cell("gru", "energy_target", 0.04, "candidate better")]
    cells += [_cell("gru", "energy_target", 0.5, "unresolved") for _ in range(3)]
    cells += [_cell("mamba", "energy_target", 1e-6, "candidate worse") for _ in range(4)]
    cells += [_cell("mamba", "energy_target", 0.9, "unresolved") for _ in range(2)]
    return cells


def test_main_family_holm_keeps_only_the_strong_candidate():
    result = holm_cells.analyse({"A0": _cells(), "A1": _cells(), "N": _cells()})
    gru = result["N"]["gru_vs_lstm"]["all"]
    assert gru["main_family"]["better_uncorrected"] == 1
    assert gru["main_family"]["better_holm"] == 0
    mamba = result["N"]["by_candidate"]["mamba"]["all"]
    assert mamba["worse_uncorrected"] == 4
    assert mamba["worse_holm"] == 4


def test_sensitivity_family_is_the_gru_cells_only():
    result = holm_cells.analyse({"A0": _cells(), "A1": _cells(), "N": _cells()})
    sens = result["N"]["gru_vs_lstm"]["all"]["sensitivity_family"]
    assert sens["cells"] == 4
    assert sens["better_holm"] == 0
    assert result["N"]["by_candidate"]["gru"]["all"]["cells"] == 4
