"""Cost-optimal flow of the medium of the heat recovery of one line (descriptive, NO reading).

Design: ``docs/NN_carrier_sweep.md``; settings: ``conf/dyssol_steady/carrier_sweep.yaml``; package
``twinlink_ml.dyssol_steady.carrier_sweep``.

Per outside temperature: cost per hour = fuel (gas price plus CO2 price) + electricity of the
circulation pump, over the medium flow on a fine grid, for one line with its own heat recovery
(rebuild of the stationary Dyssol units; no Dyssol run). The pump is a PLACEHOLDER (pressure drop
quadratic in the flow; YAML). Reported: the optimal flow, the cost at 80 % and 120 % of it, and
the same with the pressure drop times 0.5 and 2. The fresh gas keeps the sheet's humidity, capped
at saturation at the outside temperature; the optimum without the cap is reported where the cap
is active. Every scanned point is checked for over-saturation (dryer outlet and exhaust).

The check of the rebuild against the delivered sheet is not part of this script: it is Befund 130
(branch ``dyssol-sweep-case1``).

Outputs
-------
``artifacts/runs/<name>.json``: shares, percentages, ratios, flags, digests -- no input of the
line, no absolute flow, heat or cost.
``<out>/<name>_local.json`` (LOCAL ONLY): the inputs, the optima in kg/h and EUR/h, the cost
curves. ``<out>`` = ``$TWINLINK_CARRIER_OUT`` (default ``<root>/local/runs/carrier_sweep``). With
``--dry-run`` both files go to ``<out>`` and the tree need not be clean (for tests on synthetic
inputs; its numbers are never results).

Local folders: ``$TWINLINK_LOCAL_DIR`` (default ``<root>/local``) with
``dyssol_steady/carrier_sweep_local.yaml`` (inputs of the line; template next to the YAML);
``$TWINLINK_RUNS_DIR`` (default ``C:/twinlink_runs/<branch>``) for the copy of the materials
database with the added boiling point; ``$DYSSOL_HOME`` for the installed database.

Clean tree required; start with ``tools/run_here.sh scripts/run_carrier_sweep.py --name <name>``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import (  # noqa: E402
    NO_HEAT_NETWORK,
    RUNS_DIR,
    _json_default,
    git,
    provenance,
    require_clean_tree,
)

from twinlink_ml.dyssol_steady import carrier_sweep as cs  # noqa: E402
from twinlink_ml.dyssol_steady.materials import (  # noqa: E402
    DyssolProperties,
    patch_materials_database,
    read_properties,
    sha256_of,
)
from twinlink_ml.dyssol_steady.reference import LineInputs  # noqa: E402

REPO_ROOT = SCRIPTS.parent
LOCAL = Path(os.environ.get("TWINLINK_LOCAL_DIR", REPO_ROOT / "local"))
OUT = Path(os.environ.get("TWINLINK_CARRIER_OUT", REPO_ROOT / "local" / "runs" / "carrier_sweep"))
DYSSOL_HOME = Path(os.environ.get("DYSSOL_HOME", "C:/Program Files/Dyssol"))
CURVE_STRIDE = 50
"""Every n-th grid point of a cost curve is kept in the local output."""

_COMMITTED_OPTIMUM = (
    "pressure_factor",
    "flow_share_of_scan",
    "at_upper_bound",
    "at_zero",
    "saving_percent_of_no_flow_cost",
    "pump_share_of_cost_percent",
    "heat_recovered_share_of_scan_max",
    "co2_change_percent_of_no_flow",
    "oversaturated_at_optimum",
)
_COMMITTED_NEIGHBOUR = (
    "factor",
    "above_scanned_range",
    "increase_percent_of_cost",
    "increase_percent_of_saving",
)


def runs_dir() -> Path:
    default = Path("C:/twinlink_runs") / (git("rev-parse", "--abbrev-ref", "HEAD") or "detached")
    return Path(os.environ.get("TWINLINK_RUNS_DIR", default))


def load_properties() -> tuple[DyssolProperties, dict[str, str]]:
    """Properties of the materials database with the one added constant (docs/45, section 3)."""
    installed = DYSSOL_HOME / "Materials.dmdb"
    if not installed.exists():
        raise SystemExit(f"missing local file: {installed}")
    patched = runs_dir() / "materials" / f"Materials_{sha256_of(installed)[:12]}_boiling_point.dmdb"
    if not patched.exists():
        patch_materials_database(installed, patched)
    digests = {
        "installed_sha256": sha256_of(installed),
        "patched_sha256": sha256_of(patched),
        "patched_file": patched.name,
    }
    return read_properties(patched), digests


def _relative(result: dict[str, Any]) -> dict[str, Any]:
    out = {k: result[k] for k in _COMMITTED_OPTIMUM}
    out["neighbours"] = [{k: n[k] for k in _COMMITTED_NEIGHBOUR} for n in result["neighbours"]]
    return out


def scan(
    props: DyssolProperties, base: LineInputs, config: cs.SweepConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Committed (relative numbers) and local record of the scan, per outside temperature."""
    flows = config.flows()
    committed: list[dict[str, Any]] = []
    local: list[dict[str, Any]] = []
    for ambient in config.ambient_celsius:
        inputs, capped = cs.fresh_gas_at(base, ambient)
        curve = cs.cost_curve(props, inputs, flows, config.pump, config.prices)
        optima = [
            cs.optimum(curve, factor, config.neighbour_factors)
            for factor in config.pressure_factors
        ]
        uncapped = None
        if capped:
            free = cs.cost_curve(
                props,
                replace(inputs, gas_humidity_g_per_kg=base.gas_humidity_g_per_kg),
                flows,
                config.pump,
                config.prices,
            )
            uncapped = cs.optimum(free, 1.0, config.neighbour_factors)
        ends = [
            cs.saturation_ratios(cs.recompute(props, replace(inputs, medium_kg_per_h=float(m))))
            for m in (flows[0], flows[-1])
        ]
        head = {
            "ambient_celsius": ambient,
            "humidity_capped_at_saturation": capped,
            "oversaturated_share_of_scan": float(curve.oversaturated.mean()),
            "saturation_ratio_dryer_outlet": ends[0][0],
            "saturation_ratio_exhaust_at_no_flow": ends[0][1],
            "saturation_ratio_exhaust_at_upper_bound": ends[1][1],
            "heat_recovered_monotonic": bool(np.all(np.diff(curve.heat_recovered_kw) > 0.0)),
        }
        committed.append(
            {
                **head,
                "optima": [_relative(o) for o in optima],
                "optimum_without_humidity_cap": None if uncapped is None else _relative(uncapped),
            }
        )
        keep = sorted(set(range(0, len(flows), CURVE_STRIDE)) | {len(flows) - 1})
        local.append(
            {
                **head,
                "inputs": asdict(inputs),
                "optima": optima,
                "optimum_without_humidity_cap": uncapped,
                "curve": {
                    "flow_kg_per_h": flows[keep],
                    "fuel_kw": curve.fuel_kw[keep],
                    "heat_recovered_kw": curve.heat_recovered_kw[keep],
                    "pump_kw": np.asarray([config.pump.power_kw(m) for m in flows[keep]]),
                    "cost_eur_per_h": {
                        str(f): curve.costs_eur_per_h(f)[keep] for f in config.pressure_factors
                    },
                },
            }
        )
        best = next(o for o in optima if o["pressure_factor"] == 1.0)
        print(
            f"{ambient:g} degC: optimum at {100.0 * best['flow_share_of_scan']:.1f} % of the scan "
            f"(upper bound: {best['at_upper_bound']}), saving "
            f"{best['saving_percent_of_no_flow_cost']:.2f} % of the cost without flow, "
            f"over-saturated share of the scan {head['oversaturated_share_of_scan']:.3f}",
            flush=True,
        )
    return committed, local


def write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists -- artifacts are never overwritten; use a new name")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=_json_default), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size} bytes, sha256 {sha256_of(path)[:16]})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        require_clean_tree()
    config = cs.load_config()
    local_path = LOCAL / "dyssol_steady" / "carrier_sweep_local.yaml"
    if not local_path.exists():
        raise SystemExit(f"missing local file: {local_path}")
    local_file = yaml.safe_load(local_path.read_text(encoding="utf-8"))
    base = cs.line_inputs(local_file["line"])
    props, materials = load_properties()
    target = (OUT if args.dry_run else RUNS_DIR) / f"{args.name}.json"
    header = {
        "name": args.name,
        "dry_run": args.dry_run,
        "config_fingerprint": config.fingerprint,
        "line_inputs_sha256": sha256_of(local_path),
        "workbook_sha256": str(local_file["source"]["workbook_sha256"]),
        "materials": materials,
        "grid_points": len(config.flows()),
    }
    print(f"config {config.fingerprint[:12]}, {header['grid_points']} grid points", flush=True)
    committed, local = scan(props, base, config)
    block = provenance(heat_network=NO_HEAT_NETWORK, ignore_paths=[target])
    write_json(
        OUT / f"{args.name}_local.json",
        {
            "provenance": block,
            "header": header,
            "source": local_file["source"],
            "properties": props.as_dict(),
            "config": config.raw,
            "sheet_inputs": asdict(base),
            "scan": local,
        },
    )
    write_json(
        target,
        {"provenance": block, "header": header, "config": config.raw, "scan": committed},
    )


if __name__ == "__main__":
    main()
