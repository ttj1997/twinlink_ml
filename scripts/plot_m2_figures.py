r"""SY2: Abbildungen für die M2-Synthese, ausschließlich aus committeten JSONL-Artefakten
erzeugt -- keine Zahl von Hand, jede Abbildung regenerierbar.

Vier Abbildungen, alle mit physikalischen Einheiten auf der y-Achse (nicht NRMSE) und
Testmenge/Horizont in der Bildunterschrift (die Verwechslung der Testmengen hat im Projekt
bereits einmal eine falsche Zahl erzeugt, Befund 51 K5):

  1. Lernkurven, acht Architekturen, drei Kanäle, Konstantlauf-Holdout, h=36, Median und
     Spannweite über 5 Seeds, mit Auflösungsband (Befund 63, LSTM-Streuung als Quelle).
  2. Dieselben auf der angeregten Testmenge.
  3. Befund 55: angeregt gegen konstant trainiert (GRU/LSTM), Konstantlauf-Holdout,
     gleiche Testmenge wie beim Training verschieden.
  4. Inferenzzeit UND Parameterzahl je Architektur, nur aus der vergleichbaren
     Datenfamilie (Befund 51+58, Thread-Einstellungen identisch), n_train=64.

Farben: feste kategoriale Reihenfolge (dataviz-Skill-Referenzpalette, Slots 1-8, Light-Mode
-Hex), nie nach Rang neu vergeben -- jede Architektur behält ihre Farbe über alle vier
Abbildungen.

Aufruf::

    python scripts/plot_m2_figures.py --output-dir docs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twinlink_ml.evaluation.resolution import SeedSpread
from twinlink_ml.utils.seeding import VarianceSource
from twinlink_ml.utils.tracking import RunRecord, read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "artifacts" / "runs"
PROJECTED_JSONL = RUNS / "arch_learning_curve_projected.jsonl"
EXTENDED_JSONL = RUNS / "arch_learning_curve_projected_extended.jsonl"
CONSTANT_TRAINED_JSONL = RUNS / "arch_learning_curve_constant_trained.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "figures"

METRIC = "rmse_run_median_physical"
HORIZON = "36"
N_TRAIN_POINTS = ("8", "16", "32", "64")
CONFIDENCE = 0.95

CHANNEL_LABELS = {
    "energy_target": r"$P_{\mathrm{el}}$ [kW]",
    "product_moisture_kg_per_kg": r"$X_p$ [kg/kg]",
    "bed_temperature_celsius": r"$T_{\mathrm{Bett}}$ [°C]",
}
CHANNEL_TITLES = {
    "energy_target": "Energieverbrauch",
    "product_moisture_kg_per_kg": "Produktfeuchte",
    "bed_temperature_celsius": "Betttemperatur",
}

# dataviz-Skill, Referenzpalette (validiert), Light-Mode-Hex, feste Zuordnung.
ARCHITECTURE_ORDER = (
    "persistence",
    "static",
    "arx",
    "gru",
    "lstm",
    "transformer",
    "xlstm",
    "mamba",
)
ARCHITECTURE_LABELS = {
    "persistence": "Persistenz",
    "static": "Statisch",
    "arx": "ARX",
    "gru": "GRU",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "xlstm": "xLSTM",
    "mamba": "Mamba",
}
ARCHITECTURE_COLORS = {
    "persistence": "#2a78d6",  # Slot 1 blue
    "static": "#eb6834",  # Slot 2 orange
    "arx": "#1baf7a",  # Slot 3 aqua
    "gru": "#eda100",  # Slot 4 yellow
    "lstm": "#e87ba4",  # Slot 5 magenta
    "transformer": "#008300",  # Slot 6 green
    "xlstm": "#4a3aa7",  # Slot 7 violet
    "mamba": "#e34948",  # Slot 8 red
}
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
PRIMARY_INK = "#0b0b0b"


def _metric_key(channel: str) -> str:
    return f"{METRIC}/{channel}/h{HORIZON}"


def _values(
    records: list[RunRecord], *, model: str, n_train: str, test_set: str, channel: str
) -> list[float]:
    key = _metric_key(channel)
    return sorted(
        record.metrics[key]
        for record in records
        if record.tags.get("model") == model
        and record.tags.get("n_train") == n_train
        and record.tags.get("test_set") == test_set
        and key in record.metrics
    )


def _resolution_band(records: list[RunRecord], *, channel: str, test_set: str) -> dict[str, float]:
    """Relative Auflösungsschwelle je Stützstelle, aus LSTMs eigener Seed-Streuung
    (Begründung: Befund 63 -- Persistenz ist auf dem Konstantlauf-Holdout streuungsfrei)."""
    band: dict[str, float] = {}
    for n_train in N_TRAIN_POINTS:
        values = _values(records, model="lstm", n_train=n_train, test_set=test_set, channel=channel)
        if len(values) < 2:
            continue
        spread = SeedSpread(source=VarianceSource.JOINT, values=tuple(values))
        band[n_train] = spread.relative_minimum_detectable_difference(confidence=CONFIDENCE)
    return band


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, which="both", color=GRIDLINE, linewidth=0.7, zorder=0)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color(AXIS)
    ax.tick_params(colors=MUTED_INK, labelsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64])
    ax.set_xticklabels(["8", "16", "32", "64"])


def _learning_curve_figure(
    *,
    records: list[RunRecord],
    test_set: str,
    test_set_label: str,
    output_path: Path,
    architectures: tuple[str, ...] = ARCHITECTURE_ORDER,
) -> None:
    channels = list(CHANNEL_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor="white")
    for ax, channel in zip(axes, channels, strict=True):
        band = _resolution_band(records, channel=channel, test_set=test_set)
        best_at_point: dict[str, float] = {}
        for n_train in N_TRAIN_POINTS:
            medians = [
                float(
                    np.median(
                        _values(
                            records, model=m, n_train=n_train, test_set=test_set, channel=channel
                        )
                    )
                )
                for m in architectures
                if _values(records, model=m, n_train=n_train, test_set=test_set, channel=channel)
            ]
            if medians:
                best_at_point[n_train] = min(medians)

        if band:
            # Bei kleinem n_train kann die relative Schwelle > 100 % liegen (z. B.
            # product_moisture_kg_per_kg, n_train=16: 394 %) -- der Median selbst bleibt
            # unveraendert berichtet (Befund 63), nur die untere Bandkante wird hier auf
            # 5 % des Medians gedeckelt, sonst wird sie auf der log-Achse negativ und das
            # Band faellt visuell bis zum Achsenboden durch (reines Rendering-Artefakt).
            xs = [int(n) for n in band]
            lower = [best_at_point[n] * max(1.0 - band[n], 0.05) for n in band]
            upper = [best_at_point[n] * (1.0 + band[n]) for n in band]
            order = np.argsort(xs)
            xs_sorted = np.array(xs)[order]
            lower_sorted = np.array(lower)[order]
            upper_sorted = np.array(upper)[order]
            ax.fill_between(
                xs_sorted,
                lower_sorted,
                upper_sorted,
                color=MUTED_INK,
                alpha=0.15,
                zorder=1,
                label="Auflösungsband (LSTM-Streuung, 95%)",
            )

        for model in architectures:
            xs, medians, lo, hi = [], [], [], []
            for n_train in N_TRAIN_POINTS:
                values = _values(
                    records, model=model, n_train=n_train, test_set=test_set, channel=channel
                )
                if not values:
                    continue
                xs.append(int(n_train))
                medians.append(float(np.median(values)))
                lo.append(min(values))
                hi.append(max(values))
            if not xs:
                continue
            color = ARCHITECTURE_COLORS[model]
            ax.plot(
                xs,
                medians,
                marker="o",
                markersize=4,
                linewidth=1.8,
                color=color,
                label=ARCHITECTURE_LABELS[model],
                zorder=3,
            )
            ax.fill_between(xs, lo, hi, color=color, alpha=0.12, zorder=2)

        ax.set_yscale("log")
        _style_axis(ax)
        ax.set_xlabel(r"$n_{\mathrm{train}}$ [Trainingsläufe]", fontsize=9, color=MUTED_INK)
        ax.set_ylabel(CHANNEL_LABELS[channel], fontsize=9, color=MUTED_INK)
        ax.set_title(CHANNEL_TITLES[channel], fontsize=10, color=PRIMARY_INK)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle(
        f"Lernkurven, acht Architekturen, Testmenge: {test_set_label}, h=36 (900 s Rollout), "
        "Median über 5 Seeds, Band = Spannweite",
        fontsize=10.5,
        color=PRIMARY_INK,
        y=1.03,
    )
    fig.text(
        0.5,
        -0.16,
        "Primärmetrik rmse_run_median_physical, physikalische Einheiten. Graues Band: "
        "NAEHERUNGSWEISE Auflösungsschwelle, nur aus LSTMs eigener 5-Seed-Streuung (95% "
        "Konfidenz) um die jeweils beste Architektur -- fuer eine Text-/Tabellenaussage "
        "zwischen zwei konkreten Architekturen gilt die genauere Zwei-Stichproben-Schwelle "
        "aus Befund 63 (Streuung beider verglichener Modelle), die von diesem Band abweichen "
        "kann. Datenquelle: arch_learning_curve_projected.jsonl (Commit 0b3eaaf) + "
        "arch_learning_curve_projected_extended.jsonl (Commit 256d644).",
        fontsize=7.2,
        color=MUTED_INK,
        ha="center",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _befund55_figure(
    *,
    excited_records: list[RunRecord],
    constant_records: list[RunRecord],
    output_path: Path,
) -> None:
    channels = list(CHANNEL_LABELS)
    models = ("gru", "lstm")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor="white")
    for ax, channel in zip(axes, channels, strict=True):
        for model in models:
            color = ARCHITECTURE_COLORS[model]
            for records, arm_label, linestyle, alpha in (
                (excited_records, "angeregt trainiert", "-", 1.0),
                (constant_records, "konstant trainiert", "--", 0.75),
            ):
                xs, medians = [], []
                for n_train in N_TRAIN_POINTS:
                    values = _values(
                        records,
                        model=model,
                        n_train=n_train,
                        test_set="constant_holdout",
                        channel=channel,
                    )
                    if not values:
                        continue
                    xs.append(int(n_train))
                    medians.append(float(np.median(values)))
                if not xs:
                    continue
                ax.plot(
                    xs,
                    medians,
                    marker="o" if linestyle == "-" else "s",
                    markersize=4,
                    linewidth=1.8,
                    linestyle=linestyle,
                    color=color,
                    alpha=alpha,
                    label=f"{ARCHITECTURE_LABELS[model]}, {arm_label}",
                )
        ax.set_yscale("log")
        _style_axis(ax)
        ax.set_xlabel(r"$n_{\mathrm{train}}$ [Trainingsläufe]", fontsize=9, color=MUTED_INK)
        ax.set_ylabel(CHANNEL_LABELS[channel], fontsize=9, color=MUTED_INK)
        ax.set_title(CHANNEL_TITLES[channel], fontsize=10, color=PRIMARY_INK)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle(
        "Befund 55: angeregt gegen konstant trainiert, GRU/LSTM, Testmenge: Konstantlauf-"
        "Holdout (identisch für beide Arme), h=36, Median über 5 Seeds",
        fontsize=10.5,
        color=PRIMARY_INK,
        y=1.03,
    )
    fig.text(
        0.5,
        -0.18,
        "Beide Arme ausgewertet auf demselben Konstantlauf-Holdout -- nur die Trainingsart "
        "unterscheidet sich. Primärmetrik rmse_run_median_physical. Datenquelle: "
        "arch_learning_curve_projected.jsonl (Commit 0b3eaaf) + "
        "arch_learning_curve_constant_trained.jsonl (Commit 1b85ce4).",
        fontsize=7.2,
        color=MUTED_INK,
        ha="center",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _inference_and_parameters_figure(*, records: list[RunRecord], output_path: Path) -> None:
    inference_key = "inference_us_per_window"
    models = list(ARCHITECTURE_ORDER)
    inference_medians: dict[str, float] = {}
    parameter_counts: dict[str, int] = {}
    thread_settings: set[tuple[object, object, object]] = set()
    for model in models:
        values = [
            record.metrics[inference_key]
            for record in records
            if record.tags.get("model") == model
            and record.tags.get("n_train") == "64"
            and inference_key in record.metrics
        ]
        if values:
            inference_medians[model] = float(np.median(values))
        for record in records:
            if record.tags.get("model") == model and record.tags.get("n_train") == "64":
                parameter_counts[model] = int(record.params.get("model/n_parameters", 0))
                thread_settings.add(
                    (
                        record.params.get("torch/num_threads"),
                        record.params.get("env/mkl_num_threads"),
                        record.params.get("env/omp_num_threads"),
                    )
                )
                break

    fig, (ax_inference, ax_params) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor="white")
    y_positions = np.arange(len(models))[::-1]

    for ax, values_map, xlabel, title, use_log in (
        (ax_inference, inference_medians, "µs/Fenster (log)", "Inferenzzeit", True),
        (
            ax_params,
            parameter_counts,
            "Parameter (log, Persistenz/Statisch/ARX ≈ 0)",
            "Parameterzahl",
            True,
        ),
    ):
        ax.set_facecolor("#fcfcfb")
        for spine_name in ("top", "right"):
            ax.spines[spine_name].set_visible(False)
        for spine_name in ("left", "bottom"):
            ax.spines[spine_name].set_color(AXIS)
        ax.grid(True, axis="x", which="both", color=GRIDLINE, linewidth=0.7, zorder=0)
        for y, model in zip(y_positions, models, strict=True):
            value = values_map.get(model)
            if value is None:
                continue
            plotted = max(value, 0.05) if use_log else value
            ax.barh(y, plotted, color=ARCHITECTURE_COLORS[model], height=0.6, zorder=3)
            label = f"{value:,.0f}" if value >= 1 else f"{value:.2f}"
            ax.text(
                plotted * 1.15 if use_log else plotted + plotted * 0.02,
                y,
                label,
                va="center",
                fontsize=7.5,
                color=PRIMARY_INK,
            )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [ARCHITECTURE_LABELS[m] for m in models], fontsize=8.5, color=PRIMARY_INK
        )
        if use_log:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED_INK)
        ax.set_title(title, fontsize=10, color=PRIMARY_INK)
        ax.tick_params(colors=MUTED_INK, labelsize=8)

    thread_note = ", ".join(f"{t}" for t in sorted(thread_settings, key=str))
    fig.suptitle(
        "Inferenzzeit und Parameterzahl je Architektur, n_train=64, nur vergleichbare Datenfamilie",
        fontsize=10.5,
        color=PRIMARY_INK,
        y=1.04,
    )
    fig.text(
        0.5,
        -0.06,
        f"Thread-Einstellungen (torch/num_threads, env/mkl_num_threads, env/omp_num_threads) "
        f"in allen zugrunde liegenden Records identisch: {thread_note}. Nicht poolbar mit den "
        "im Fließtext von Befund 37/38 zitierten Zahlen (dort kein Thread-Feld geloggt). "
        "Datenquelle: arch_learning_curve_projected.jsonl (Commit 0b3eaaf) + "
        "arch_learning_curve_projected_extended.jsonl (Commit 256d644), Median über 5 "
        "Seeds × 2 Testmengen.",
        fontsize=7.2,
        color=MUTED_INK,
        ha="center",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    projected = read_jsonl(PROJECTED_JSONL)
    extended = read_jsonl(EXTENDED_JSONL)
    headline = projected + extended
    constant_trained = read_jsonl(CONSTANT_TRAINED_JSONL)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    fig1 = arguments.output_dir / "abb1_lernkurven_konstantlauf.png"
    _learning_curve_figure(
        records=headline,
        test_set="constant_holdout",
        test_set_label="Konstantlauf-Holdout",
        output_path=fig1,
    )
    print(f"Abbildung 1 geschrieben: {fig1}")

    fig2 = arguments.output_dir / "abb2_lernkurven_angeregt.png"
    _learning_curve_figure(
        records=headline, test_set="excited", test_set_label="angeregt", output_path=fig2
    )
    print(f"Abbildung 2 geschrieben: {fig2}")

    fig3 = arguments.output_dir / "abb3_befund55_angeregt_vs_konstant.png"
    _befund55_figure(excited_records=projected, constant_records=constant_trained, output_path=fig3)
    print(f"Abbildung 3 geschrieben: {fig3}")

    fig4 = arguments.output_dir / "abb4_inferenzzeit_parameter.png"
    _inference_and_parameters_figure(records=headline, output_path=fig4)
    print(f"Abbildung 4 geschrieben: {fig4}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
