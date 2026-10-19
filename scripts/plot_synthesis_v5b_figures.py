r"""Figures of the synthesis, version 5b (docs/08), generated only from committed artifacts.

Same data and same plotting as ``plot_synthesis_v5_figures.py`` (which stays unchanged, so that the
figures of version 5 keep regenerating byte for byte); changes requested by review 4 (finding 21):

* numbered file names and titles: Figure 1 (learning curves, excited test set), Figure 2 (learning
  curves, constant-run holdout), Figure 3 (inference cost);
* units in the panel titles: energy target dE in kW, product moisture in kg/kg, bed temperature
  RMSE in K (an error of a temperature is a difference);
* provenance in the PNG metadata (``Source``: commit, script, artifacts); the script requires a clean
  tree so that the commit names the code that drew the figure;
* existing files are never overwritten.

Call::

    tools/run_here.sh scripts/plot_synthesis_v5b_figures.py --output-dir docs/figures
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent / "verify"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import provenance, require_clean_tree
from plot_synthesis_v5_figures import (
    CHANNELS,
    COLORS,
    HORIZON,
    INK,
    LABELS,
    N_TRAIN,
    ORDER,
    REPO_ROOT,
    RUNS,
    TEST_SETS,
    load_primary,
    style,
    values,
)

CHANNEL_TITLES = {
    "energy_target": "energy target dE, RMSE [kW]",
    "product_moisture_kg_per_kg": "product moisture, RMSE [kg/kg]",
    "bed_temperature_celsius": "bed temperature, RMSE [K]",
}
FILES = {
    "excited": ("Figure 1", "fig1_v5b_learning_curves_excited.png"),
    "constant_holdout": ("Figure 2", "fig2_v5b_learning_curves_constant_holdout.png"),
}
COST_FILE = ("Figure 3", "fig3_v5b_inference_cost_idle.png")


def metadata(source: str) -> dict[str, str]:
    commit = provenance()["git/commit"]
    return {
        "Source": f"twinlink_ml commit {commit}; scripts/plot_synthesis_v5b_figures.py; {source}",
        "Software": f"matplotlib {matplotlib.__version__}",
    }


def save(fig: plt.Figure, output: Path, source: str) -> None:
    if output.exists():
        raise SystemExit(f"{output} exists -- figures are never overwritten; use a new name")
    fig.savefig(output, dpi=160, metadata=metadata(source))
    plt.close(fig)
    print(f"wrote {output}")


def learning_curves(records: list[dict], test_set: str, output: Path, number: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, channel in zip(axes, CHANNELS, strict=True):
        style(ax)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(N_TRAIN)
        ax.set_xticklabels([str(n) for n in N_TRAIN])
        for method in ORDER:
            series = [values(records, method, n, test_set, channel) for n in N_TRAIN]
            if any(len(v) == 0 for v in series):
                continue
            median = [statistics.median(v) for v in series]
            ax.plot(
                N_TRAIN,
                median,
                color=COLORS[method],
                marker="o",
                ms=3,
                lw=1.6,
                label=LABELS[method],
            )
            ax.fill_between(
                N_TRAIN,
                [min(v) for v in series],
                [max(v) for v in series],
                color=COLORS[method],
                alpha=0.12,
                lw=0,
            )
        ax.set_title(CHANNEL_TITLES[channel], fontsize=10)
        ax.set_xlabel("training runs", fontsize=9, color=INK)
    axes[0].set_ylabel(f"RMSE at horizon {HORIZON} (900 s)", fontsize=9, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.suptitle(
        f"{number}: arm N (preset version 2), {TEST_SETS[test_set]}: median and range over 5 seeds",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    save(fig, output, "rerun_full_N.jsonl (eval/primary, rmse_run_median_physical, h36)")


def inference_cost(timing: dict, output: Path, number: str) -> None:
    rows = {r["method"]: r for r in timing["results"]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    panels = (
        ("latency_batch1_us_median", "one window (batch of 1)"),
        ("throughput_us_per_window_median", "per window in a batch (4352 windows)"),
    )
    for ax, (key, title) in zip(axes, panels, strict=True):
        style(ax)
        ax.set_yscale("log")
        ax.bar(
            range(len(ORDER)),
            [rows[m][key] for m in ORDER],
            color=[COLORS[m] for m in ORDER],
            zorder=3,
        )
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABELS[m] for m in ORDER], rotation=40, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("microseconds", fontsize=9, color=INK)
    fig.suptitle(
        f"{number}: inference cost, idle machine, one thread "
        "(arm N pool, n_train = 64, excited test set)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, output, "joint_timing_idle.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs" / "figures")
    arguments = parser.parse_args()
    require_clean_tree()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_primary(RUNS / "rerun_full_N.jsonl")
    for test_set, (number, name) in FILES.items():
        learning_curves(records, test_set, arguments.output_dir / name, number)
    timing = json.loads((RUNS / "joint_timing_idle.json").read_text(encoding="utf-8"))
    inference_cost(timing, arguments.output_dir / COST_FILE[1], COST_FILE[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
