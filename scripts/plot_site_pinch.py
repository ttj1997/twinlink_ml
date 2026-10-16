"""Figures of the site pinch study (docs/20_site_pinch.md) from the artifact ``run_site_pinch``.

Reads only the artifact (no computation of its own) and writes two PNG files; refuses to overwrite.

* ``docs/figures/fig_site_pinch_composites.png``: time-averaged hot and cold composite curves
  (top row) and grand composite curve (bottom row) of the reference scenarios R1, R2, R5, R6;
* ``docs/figures/fig_site_pinch_network_levels.png``: share of the target saving reached by a
  network at each (supply, return) level (network + in-plant direct) for R1-R6.

Call::

    tools/run_here.sh scripts/plot_site_pinch.py --artifact run_site_pinch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "artifacts" / "runs"
FIGURES = REPO_ROOT / "docs" / "figures"
PANELS = ("R1", "R2", "R5", "R6")
TITLES = {
    "R1": "R1: A product 1, C (volume)",
    "R2": "R2: A product 2, C (volume)",
    "R3": "R3: R1, C (annual figure)",
    "R4": "R4: R1 without plant C",
    "R5": "R5: R1 + 2 lines (exhaust ~61 degC)",
    "R6": "R6: R1 + 2 lines (exhaust ~34 degC)",
}


def _out(name: str) -> Path:
    path = FIGURES / name
    if path.exists():
        raise SystemExit(f"{path} exists -- figures are never overwritten; use a new name")
    return path


def composites(refs: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, len(PANELS), figsize=(4.2 * len(PANELS), 7.5))
    for j, key in enumerate(PANELS):
        d = refs[key]["detail"]
        ax = axes[0, j]
        t = d["composite_temperature_celsius"]
        # standard plot: hot composite from 0, cold composite shifted right by the cold utility;
        # each curve only over its own temperature range
        shift = d["tam_cold_utility_kw"]
        for values, offset, color, label in (
            (d["composite_hot_below_kw"], 0.0, "#c0392b", "hot composite (sources)"),
            (d["composite_cold_below_kw"], shift, "#2471a3", "cold composite (sinks)"),
        ):
            top = max(values)
            pts = [(v + offset, tt) for v, tt in zip(values, t, strict=True) if 0.0 < v < top]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, label=label)
        ax.set_title(TITLES[key], fontsize=9)
        ax.set_xlabel("heat flow, kW (time average)")
        if j == 0:
            ax.set_ylabel("temperature, degC")
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
        ax2 = axes[1, j]
        ax2.plot(d["gcc_kw"], d["gcc_shifted_celsius"], color="#7d3c98")
        ax2.axhline(
            d["tam_pinch_hot_celsius"] - refs[key]["axes"]["minimum_pinch_k"] / 2,
            color="grey",
            lw=0.8,
            ls="--",
        )
        ax2.set_xlabel("net heat flow, kW (grand composite, time average)")
        if j == 0:
            ax2.set_ylabel("shifted temperature, degC")
        ax2.grid(alpha=0.3)
    fig.suptitle(
        "Site pinch, time-averaged curves (placeholder scenarios, not a plant statement)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def network_levels(refs: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    levels = list(refs["R1"]["network"])
    for key in ("R1", "R2", "R3", "R4", "R5", "R6"):
        values = [refs[key]["network"][lv]["fraction_of_target_saving"] for lv in levels]
        ax.plot(range(len(levels)), values, marker="o", label=TITLES[key])
    ax.axhline(1.0, color="grey", lw=0.8)
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, rotation=45, fontsize=8)
    ax.set_xlabel("network supply / return, degC (approach dT_min/2 per exchanger)")
    ax.set_ylabel("share of the target saving reached")
    ax.set_title(
        "One network level + direct recovery inside each plant (placeholders)", fontsize=10
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--artifact", default="run_site_pinch")
    args = parser.parse_args()
    data = json.loads((RUNS / f"{args.artifact}.json").read_text(encoding="utf-8"))
    refs = data["references"]
    composites(refs, _out("fig_site_pinch_composites.png"))
    network_levels(refs, _out("fig_site_pinch_network_levels.png"))
    print("figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
