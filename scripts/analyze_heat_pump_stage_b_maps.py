r"""Heat-pump study: Stage-B foresight maps per operating mode, aggregated from ``gate_heat_pump``.

Pure aggregation of ``artifacts/runs/gate_heat_pump.json`` (nothing recomputed). The gate maps of
the main artifact pool all modes; in mode ``shift`` the rule gap exists already at a constant
signal (source scarcity at night, see the post-hoc analysis), so the question "from which signal
swing and storage size does foresight become valuable" is answered here PER MODE: in mode
``flat`` (all streams constant) a gap can only come from the signal.

Tables: max and median rule gap D and max storage gain G per (mode, criterion kind, amplitude,
capacity) over all runs (realistic or not; the amplitude band is marked), the smallest amplitude
with D >= 1 and >= 2 points per (mode, kind, capacity), the ten largest gaps with their axes, and
the smallest share of heat-pump heat delivered in pinch-straddling steps per Stage-A criterion.

Call::

    tools/run_here.sh scripts/analyze_heat_pump_stage_b_maps.py --name gate_heat_pump_stage_b_maps
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import (  # type: ignore[import-untyped]
    NO_HEAT_NETWORK,
    REPO_ROOT,
    provenance,
    require_clean_tree,
    write_artifact,
)

SOURCE = REPO_ROOT / "artifacts" / "runs" / "gate_heat_pump.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    require_clean_tree()
    src = json.loads(SOURCE.read_text(encoding="utf-8"))
    grid = src["stage_b"]["grid"]
    bands = src["stage_b"]["gate"]["amplitude_bands"]
    gaps: dict[tuple[str, str, float, float], list[float]] = defaultdict(list)
    gains: dict[tuple[str, str, float, float], list[float]] = defaultdict(list)
    cells: list[dict[str, Any]] = []
    for key, r in grid.items():
        m = r["meta"]
        for i, cap in enumerate(r["capacities_kwh"]):
            k = (m["mode"], m["kind"], float(m["amplitude"]), float(cap))
            gaps[k].append(r["gap_fraction"][i])
            gains[k].append(r["storage_gain_fraction"][i])
            cells.append(
                {
                    "key": key,
                    **m,
                    "capacity_kwh": cap,
                    "realistic": r["realistic"],
                    "gap": r["gap_fraction"][i],
                    "gain": r["storage_gain_fraction"][i],
                }
            )
    maps: dict[str, dict[str, dict[str, float]]] = {}
    for (mode, kind, amp, cap), vals in sorted(gaps.items()):
        band = bands["price"] if kind == "cost" else bands["co2"]
        row = maps.setdefault(f"{mode}|{kind}", {}).setdefault(
            f"{amp:g}", {"inside_band": float(band[0] <= amp <= band[1])}
        )
        row[f"max_gap|{cap:g}"] = float(np.max(vals))
        row[f"median_gap|{cap:g}"] = float(np.median(vals))
        row[f"max_gain|{cap:g}"] = float(np.max(gains[(mode, kind, amp, cap)]))
    smallest: dict[str, dict[str, dict[str, float | None]]] = {}
    for thr in (0.01, 0.02):
        out: dict[str, dict[str, float | None]] = {}
        for (mode, kind, amp, cap), vals in sorted(gaps.items()):
            row2 = out.setdefault(f"{mode}|{kind}", {})
            cur = row2.get(f"{cap:g}")
            if max(vals) >= thr - 1e-9 and (cur is None or amp < cur):
                row2[f"{cap:g}"] = amp
            else:
                row2.setdefault(f"{cap:g}", None)
        smallest[f"{thr:g}"] = out
    top = sorted((c for c in cells if c["realistic"]), key=lambda c: -c["gap"])[:10]
    across: dict[str, float] = {}
    for rec in src["stage_a"]["grid"].values():
        for q in rec["qualities"].values():
            for name, f in q["full"].items():
                for i, h in enumerate(f["heat_kwh_per_day"]):
                    if h > 1e-6:
                        across[name] = min(across.get(name, 1.0), f["across_share"][i])
    write_artifact(
        args.name,
        {
            "_provenance": provenance(NO_HEAT_NETWORK),
            "name": args.name,
            "source_artifact": "gate_heat_pump",
            "source_provenance": src["provenance"],
            "amplitude_bands": bands,
            "maps_by_mode_and_kind": maps,
            "smallest_amplitude_by_mode_and_kind": smallest,
            "largest_realistic_gaps": top,
            "across_share_min_per_criterion_stage_a": across,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
