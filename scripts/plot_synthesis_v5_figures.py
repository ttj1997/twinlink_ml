r"""Figures of the synthesis, version 5 (docs/08), generated only from committed artifacts.

  1. Learning curves of the eight methods, arm N (preset version 2; Befund 94): error over the
     number of training runs, median and range over five seeds, one panel per target, one
     figure per test set (excited / constant-run holdout). Records with ``eval/primary``
     (own pool, planned window inputs), physical RMSE of the run median at h = 36.
  2. Inference cost on an idle machine (Befund 97, ``joint_timing_idle``): latency of one
     window and cost per window in a batch, eight methods, microseconds, logarithmic axis.

Call::

    tools/run_here.sh scripts/plot_synthesis_v5_figures.py --output-dir docs/figures
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "artifacts" / "runs"
METRIC = "rmse_run_median_physical"
HORIZON = 36
N_TRAIN = (8, 16, 32, 64)
CHANNELS = ("energy_target", "product_moisture_kg_per_kg", "bed_temperature_celsius")
CHANNEL_TITLES = {
    "energy_target": "energy target",
    "product_moisture_kg_per_kg": "product moisture [kg/kg]",
    "bed_temperature_celsius": "bed temperature [degC]",
}
TEST_SETS = {
    "excited": "excited test set",
    "constant_holdout": "constant-run holdout",
}
ORDER = ("persistence", "static", "arx", "gru", "lstm", "transformer", "xlstm", "mamba")
LABELS = {
    "persistence": "persistence",
    "static": "static regression",
    "arx": "ARX",
    "gru": "GRU",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "xlstm": "xLSTM",
    "mamba": "Mamba",
}
COLORS = {
    "persistence": "#2a78d6",
    "static": "#eb6834",
    "arx": "#1baf7a",
    "gru": "#eda100",
    "lstm": "#e87ba4",
    "transformer": "#008300",
    "xlstm": "#4a3aa7",
    "mamba": "#e34948",
}
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#898781"


def load_primary(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return [r for r in records if r["params"].get("eval/primary")]


def values(records: list[dict], method: str, n: int, test_set: str, channel: str) -> list[float]:
    key = f"{METRIC}/{channel}/h{HORIZON}"
    return [
        r["metrics"][key]
        for r in records
        if r["tags"]["model"] == method
        and int(r["tags"]["n_train"]) == n
        and r["tags"]["test_set"] == test_set
        and key in r["metrics"]
    ]


def style(ax: plt.Axes) -> None:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, which="both", color=GRID, linewidth=0.7, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK, labelsize=8)


def learning_curves(records: list[dict], test_set: str, output: Path) -> None:
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
    axes[0].set_ylabel(f"RMSE (physical units), h = {HORIZON}", fontsize=9, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.suptitle(
        f"Arm N (preset version 2), {TEST_SETS[test_set]}: median and range over 5 seeds",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def inference_cost(timing: dict, output: Path) -> None:
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
        "Inference cost, idle machine, one thread (arm N pool, n_train = 64, excited test set)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs" / "figures")
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_primary(RUNS / "rerun_full_N.jsonl")
    for test_set in TEST_SETS:
        learning_curves(
            records, test_set, arguments.output_dir / f"fig1_learning_curves_arm_n_{test_set}.png"
        )
    timing = json.loads((RUNS / "joint_timing_idle.json").read_text(encoding="utf-8"))
    inference_cost(timing, arguments.output_dir / "fig2_inference_cost_idle.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
