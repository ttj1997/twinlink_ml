r"""Print the comparison Befund 99 (solid fraction 0.3) against the re-check (0.60 / 0.634 / 0.70).

Reads only: ``gate_storage_timeslice``, ``gate_storage_timeslice_posthoc_sensitivity`` and the
artifacts ``recheck_storage_timeslice_solid_fraction_{060,0634,070}`` (those present). Writes
nothing; the tables of ``docs/19a_storage_time_slice_solid_fraction_recheck.md`` are this
output.

Call::

    tools/run_here.sh scripts/report_storage_timeslice_solid_fraction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
from _common import RUNS_DIR

RECHECK = {"0.60": "060", "0.634": "0634", "0.70": "070"}
TOL_KWH = 0.5


def load(name: str) -> dict[str, Any] | None:
    path = RUNS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pct(x: float | None) -> str:
    return "--" if x is None else f"{100.0 * x:.2f} %"


def main() -> int:
    b99 = load("gate_storage_timeslice")
    b99_ph = load("gate_storage_timeslice_posthoc_sensitivity")
    assert b99 is not None and b99_ph is not None
    runs = {k: load(f"recheck_storage_timeslice_solid_fraction_{v}") for k, v in RECHECK.items()}
    runs = {k: v for k, v in runs.items() if v is not None}
    if not runs:
        print("no re-check artifact yet")
        return 1
    first = next(iter(runs.values()))
    cols = ["0.3", *runs]
    head = {"0.3": first["befund_99"]["headline"]}
    head.update({k: r["run"]["headline"] for k, r in runs.items()})
    gates = {"0.3": b99["gate"]}
    gates.update({k: r["run"]["gate"] for k, r in runs.items()})
    days = {"0.3": first["befund_99"]["days"]}
    days.update({k: r["run"]["days"] for k, r in runs.items()})

    for k, r in runs.items():
        prov = r["provenance"]
        print(
            f"x_s={k}: commit {prov['git/commit'][:7]} dirty={prov['git/dirty']} control={r['control_at_0.3']}"
        )
    print("\n## exhaust state of plant A")
    for d in ("product_1", "product_2"):
        for k in cols:
            s = days[k][d]
            print(
                f"{d} x_s={k}: Y={s['exhaust_humidity_reconstructed_mean_kg_per_kg']:.4f} "
                f"capped={s['exhaust_capped_at_saturation_fraction']:.2f} "
                f"exhaust A {s['exhaust_a_mean_kw_down_to_T_kalt']:.0f} kW, "
                f"exhaust C (2 lines) {s['exhaust_c_flat_mean_kw_volume_basis_two_lines']:.0f} kW, "
                f"admissible={s['admissible_by_energy_balance']}"
            )

    print("\n## gate")
    keys = (
        "reading",
        "n_cells",
        "n_cells_saving_at_least_5_percent",
        "n_cells_saving_and_gap",
        "n_scenarios_with_a_cell_saving_at_least_5_percent",
        "max_saving_fraction",
        "max_saving_fraction_label",
        "max_saving_kwh_per_day",
        "max_saving_kwh_per_day_label",
        "max_gap_fraction",
        "n_unrealistic",
    )
    for key in keys:
        print(f"{key}: " + " | ".join(f"{k}: {gates[k]['verdict'].get(key)}" for k in cols))
    groups = sorted(gates["0.3"]["per_group"])
    for g in groups:
        print(
            f"{g}: "
            + " | ".join(
                f"{k}: {gates[k]['per_group'][g]['reading']} "
                f"{gates[k]['per_group'][g]['n_cells_saving_at_least_5_percent']} "
                f"max {pct(gates[k]['per_group'][g]['max_saving_fraction'])}"
                for k in cols
                if g in gates[k]["per_group"]
            )
        )
    c = "closest_to_measured_days_site_A_flat_no_hall_no_water"
    print(
        "closest: "
        + " | ".join(
            f"{k}: {gates[k][c]['reading']} max {pct(gates[k][c]['max_saving_fraction'])}"
            for k in cols
        )
    )

    print("\n## main grid (headline)")
    for key in (
        "n_scenarios_with_cell_saving_at_least_5_percent",
        "share_scenarios_with_cell_saving_at_least_5_percent",
        "max_saving_fraction_product_1",
        "max_saving_fraction_product_2",
        "max_saving_fraction_mode_flat",
        "max_saving_fraction_closest_A_flat_0K_no_hall_no_water_loss0",
        "loss_effect_day_5MWh_kwh_per_day",
        "day_vs_week_shift_5MWh_max_abs_difference_kwh_per_day",
    ):
        print(f"{key}: " + " | ".join(f"{k}: {head[k][key]}" for k in cols))
    for key in head["0.3"]["count_by"]:
        print(
            f"count_by {key}: " + " | ".join(f"{k}: {head[k]['count_by'].get(key)}" for k in cols)
        )

    print("\n## whole main grid against Befund 99 (same label)")
    ref_grid = {r["label"]: r for r in b99["grid"]}
    for k, r in runs.items():
        d_direct, d_save, d_frac, changed = 0.0, 0.0, 0.0, {}
        for rec in r["run"]["grid"]:
            ref = ref_grid[rec["label"]]
            d_direct = max(d_direct, abs(rec["direct_kwh_per_day"] - ref["direct_kwh_per_day"]))
            ds = max(
                abs(a - b)
                for a, b in zip(rec["saving_kwh_per_day"], ref["saving_kwh_per_day"], strict=True)
            )
            d_frac = max(
                d_frac,
                max(
                    abs(a - b)
                    for a, b in zip(rec["saving_fraction"], ref["saving_fraction"], strict=True)
                ),
            )
            d_save = max(d_save, ds)
            if ds > TOL_KWH or abs(rec["direct_kwh_per_day"] - ref["direct_kwh_per_day"]) > TOL_KWH:
                key = f"{rec['site']}|{rec['day']}|{rec['ambient_shift_k']:+g}K"
                changed[key] = changed.get(key, 0) + 1
            if abs(rec["direct_kwh_per_day"] - ref["direct_kwh_per_day"]) >= d_direct:
                worst_direct = (rec["label"], rec["direct_kwh_per_day"], ref["direct_kwh_per_day"])
            if ds >= d_save:
                worst_save = (
                    rec["label"],
                    ds,
                    rec["saving_kwh_per_day"],
                    ref["saving_kwh_per_day"],
                )
        print(
            f"x_s={k}: max |dE_direct| {d_direct:.3g} kWh/d, max |d saving| {d_save:.1f} kWh/d "
            f"({100 * d_frac:.2f} points), scenarios changed > {TOL_KWH} kWh/d: {sum(changed.values())}"
        )
        print(f"  by site|day|weather: {dict(sorted(changed.items()))}")
        print(
            f"  largest E_direct change: {worst_direct[0]} {worst_direct[2]:.0f} -> {worst_direct[1]:.0f}"
        )
        print(
            f"  largest saving change: {worst_save[0]} "
            f"{[round(x) for x in worst_save[3]]} -> {[round(x) for x in worst_save[2]]}"
        )

    print("\n## table scenarios (E_direct; saving kWh/d at 0.5/1/2/5/10/20 MWh; C90; max rule gap)")
    for tag in head["0.3"]["table"]:
        for k in cols:
            t = head[k]["table"][tag]
            sv = t["saving_kwh_per_day"]
            print(
                f"{tag} x_s={k}: {t['direct_kwh_per_day']:.0f}; "
                + "/".join(
                    f"{sv[c]:.0f}" for c in ("500", "1000", "2000", "5000", "10000", "20000")
                )
                + f" ({pct(t['saving_fraction']['5000'])} at 5 MWh); C90 {t['capacity_at_90_percent_of_max_saving_kwh']}; "
                f"gap {t['max_rule_gap_fraction']:.1e}; level {t['ideal_level_celsius_5MWh']}"
            )

    print(
        "\n## decomposition (saving at 5 MWh, kWh/d; only entries that differ from Befund 99 by > 0.5)"
    )
    dec = {"0.3": b99["decomposition"]}
    dec.update({k: r["run"]["decomposition"] for k, r in runs.items()})
    for tag, ref in dec["0.3"].items():
        print(
            f"{tag}: full "
            + " | ".join(
                f"{k}: E_direct {dec[k][tag]['full']['direct_kwh_per_day']:.0f}, "
                f"saving {dec[k][tag]['full']['saving_5MWh_kwh_per_day']:.0f}, "
                f"multi-level {dec[k][tag]['ideal_single_level_5MWh_kwh_per_day'] - dec[k][tag]['multi_level_5MWh_kwh_per_day']:.0f}"
                for k in cols
            )
        )
        for part in ("leave_one_out", "pairs"):
            for name in ref[part]:
                vals = {k: dec[k][tag][part][name]["saving_5MWh_kwh_per_day"] for k in cols}
                if max(abs(x - vals["0.3"]) for x in vals.values()) > TOL_KWH:
                    print(
                        f"  {part} {name}: " + " | ".join(f"{k}: {x:.0f}" for k, x in vals.items())
                    )
    for k, r in runs.items():
        extra = [t for t in r["run"]["decomposition"] if t not in dec["0.3"]]
        for t in extra:
            d = r["run"]["decomposition"][t]
            print(f"x_s={k} {t}: {d['scenario']} saving {d['full']['saving_5MWh_kwh_per_day']:.0f}")

    print("\n## sensitivity (saving at 5 MWh kWh/d; variants that differ from Befund 99 by > 0.5)")
    sens = {"0.3": {**b99["sensitivity"], **b99_ph["sensitivity"]}}
    sens.update({k: r["run"]["sensitivity"] for k, r in runs.items()})
    for rid in ("R1", "R2", "R3", "R4"):
        base = {k: sens[k][rid] for k in cols}
        b = {
            k: v.get("base_saving_kwh_per_day", 0.0 if rid in ("R1", "R2") else None)
            for k, v in base.items()
        }
        print(
            f"{rid}: E_direct "
            + " | ".join(f"{k}: {base[k]['base_direct_kwh_per_day']:.0f}" for k in cols)
            + "; base saving "
            + " | ".join(f"{k}: {b[k]}" for k in cols)
        )
        ref = {r_["variant"]: r_ for r_ in base["0.3"]["ranking"]}
        for name, row in ref.items():
            vals = {"0.3": row["saving_kwh_per_day"]}
            for k in runs:
                hit = [x for x in base[k]["ranking"] if x["variant"] == name]
                vals[k] = hit[0]["saving_kwh_per_day"] if hit else float("nan")
            if any(abs(x - vals["0.3"]) > TOL_KWH for x in vals.values() if x == x):
                print(f"  {name}: " + " | ".join(f"{k}: {x:.0f}" for k, x in vals.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
