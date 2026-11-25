"""Learning curve of surrogates of a stationary Dyssol flowsheet: config, metrics, reading.

Question (decision of the project lead, 2026-11-19): how many Dyssol runs does a surrogate need
when no rebuild of the flowsheet exists? The reading is pre-registered in
``docs/45_dyssol_steady_state.md`` and implemented in :func:`reading`.

Per cell (model, number of runs ``N``, seed) the surrogate is fitted on ``N`` points of the
training pool and scored on the test set. Two numbers decide:

* ``exhaust_p95_k`` -- 95th percentile over the test set of the absolute error of the exhaust
  temperature after the heat recovery, the larger of the lines;
* ``fuel_p95_percent`` -- 95th percentile of the relative error of the total fuel consumption.

A model *meets the thresholds at N* if the median over the seeds of both numbers is at or below
its threshold. ``N*`` of a model is the smallest ``N`` of the grid from which on (every larger
``N`` of the grid included) it meets them.

A later study (``docs/50_dyssol_freshgas.md``, settings ``dyssol_freshgas.yaml``) uses the same
reading with seven varied inputs and adds: a pilot grid given by its axes with one automatic
extension (:func:`pilot_grid`, :func:`pilot_extension`), a second, named test set of points of a
partner sheet (:func:`sheet_errors`, :func:`sheet_reading`), the control tabulated over an input
(:func:`control_by_bins`) and the check of a prediction of ``N*`` (:func:`prediction_check`). The
keys these need are optional in the YAML; without them a config behaves as before.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from itertools import product as iter_product
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from twinlink_ml.evaluation.b1_analysis import holm
from twinlink_ml.evaluation.paired_resolution import paired_comparison

from .design import VARIED
from .models import MlpRegressor, MlpSettings, PolynomialRegression, Surrogate
from .reference import LineOutputs
from .sweep_sheet import RESULT_KEYS, compare, totals

__all__ = [
    "BASELINES",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_TEST_SET_NAME",
    "MODELS",
    "PILOT_AXES",
    "SHEET_TARGETS",
    "TARGETS_PER_LINE",
    "StudyConfig",
    "cell_metrics",
    "control_by_bins",
    "fit_and_score",
    "learning_table",
    "load_config",
    "make_model",
    "pilot_extension",
    "pilot_grid",
    "pilot_selection",
    "prediction_check",
    "rank_tests",
    "reading",
    "sheet_errors",
    "sheet_reading",
    "target_matrix",
    "target_names",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "conf" / "dyssol_steady" / "dyssol_steady.yaml"

TARGETS_PER_LINE: tuple[str, ...] = (
    "exhaust_celsius",
    "dryer_outlet_celsius",
    "preheated_celsius",
    "heat_recovered_kw",
    "fuel_kg_per_h",
)
"""Targets of one line, in the column order of every target array."""

DEFAULT_TEST_SET_NAME = "in-box Sobol test set, independent of the training pool"

PILOT_AXES: tuple[str, ...] = ("learning_rate", "width", "steps", "weight_decay")
"""Axes of a pilot grid given as a full factorial, in the order of the product (the first axis
varies slowest); ``width`` is the width of each of the hidden layers."""

SHEET_TARGETS: Mapping[str, str] = {
    "product_celsius": "dryer_outlet_celsius",
    "exhaust_celsius": "exhaust_celsius",
    "recovered_kw": "heat_recovered_kw",
    "medium_hot_celsius": "exhaust_celsius",
    "medium_cold_celsius": "preheated_celsius",
    "fuel_kg_per_h": "fuel_kg_per_h",
}
"""Target of the surrogate behind each quantity of the sweep sheet (as
:func:`twinlink_ml.dyssol_steady.sweep_sheet.sweep_table` takes them from the units' results)."""

BASELINES: tuple[str, ...] = ("linear", "quadratic")
MODELS: tuple[str, ...] = (*BASELINES, "mlp")
"""Models in the order of increasing complexity."""


@dataclass(frozen=True)
class StudyConfig:
    """Settings of the study (``conf/dyssol_steady/dyssol_steady.yaml``)."""

    version: int
    cases: Mapping[str, Mapping[str, Any]]
    pool_size: int
    pool_seed: int
    test_size: int
    test_seed: int
    budgets: tuple[int, ...]
    seeds: tuple[int, ...]
    hidden: tuple[int, ...]
    selected: Mapping[str, Mapping[str, Any] | None]
    pilot: Mapping[str, Any]
    threshold_exhaust_k: float
    threshold_fuel_percent: float
    control_temperature_k: float
    control_relative_percent: float
    alpha: float
    raw: Mapping[str, Any] = field(repr=False)
    varied: tuple[str, ...] = VARIED
    """Varied inputs per line (key ``varied``; default: the five of docs/45)."""
    box_file: str = "box_local.yaml"
    """Name of the local box file under ``<local>/dyssol_steady/`` (key ``box_file``)."""
    test_set_name: str = DEFAULT_TEST_SET_NAME
    sheet_test: Mapping[str, Any] | None = None
    """Second test set: points of a partner sheet (key ``sheet_test``)."""
    prediction: Mapping[str, Any] | None = None
    """Prediction of an ``N*`` fixed before the data (key ``prediction``)."""

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the parsed YAML (canonical JSON)."""
        text = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def mlp_settings(self, case: str) -> MlpSettings:
        """Selected optimiser settings of a case; fails while its pilot selection is missing."""
        selected = self.selected.get(case)
        if selected is None:
            raise ValueError(
                f"mlp.selected.{case} is empty: run the pilot and commit its selection"
            )
        return MlpSettings(
            hidden=tuple(int(h) for h in selected.get("hidden", self.hidden)),
            learning_rate=float(selected["learning_rate"]),
            steps=int(selected["steps"]),
            weight_decay=float(selected["weight_decay"]),
        )


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> StudyConfig:
    """Read and check the study YAML."""
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    budgets = tuple(int(n) for n in raw["budgets"])
    design = raw["design"]
    if list(budgets) != sorted(set(budgets)) or budgets[-1] > int(design["pool_size"]):
        raise ValueError("budgets must increase and fit into the pool")
    thresholds, control = raw["thresholds"], raw["control"]
    selected = {
        case: None
        if value is None
        else {k: [int(h) for h in v] if k == "hidden" else float(v) for k, v in value.items()}
        for case, value in raw["mlp"]["selected"].items()
    }
    if set(selected) != set(raw["cases"]):
        raise ValueError("mlp.selected needs one entry per case")
    return StudyConfig(
        version=int(raw["version"]),
        cases=raw["cases"],
        pool_size=int(design["pool_size"]),
        pool_seed=int(design["pool_seed"]),
        test_size=int(design["test_size"]),
        test_seed=int(design["test_seed"]),
        budgets=budgets,
        seeds=tuple(int(s) for s in raw["seeds"]),
        hidden=tuple(int(h) for h in raw["mlp"]["hidden"]),
        selected=selected,
        pilot=raw["pilot"],
        threshold_exhaust_k=float(thresholds["exhaust_temperature_k"]),
        threshold_fuel_percent=float(thresholds["total_fuel_percent"]),
        control_temperature_k=float(control["temperature_k"]),
        control_relative_percent=float(control["relative_percent"]),
        alpha=float(raw["statistics"]["alpha"]),
        raw=raw,
        varied=tuple(str(v) for v in raw.get("varied", VARIED)),
        box_file=str(raw.get("box_file", "box_local.yaml")),
        test_set_name=str(design.get("test_set_name", DEFAULT_TEST_SET_NAME)),
        sheet_test=raw.get("sheet_test"),
        prediction=raw.get("prediction"),
    )


def target_names(n_lines: int) -> tuple[str, ...]:
    """Column names ``line<k>/<target>`` of a target array."""
    return tuple(f"line{k}/{t}" for k in range(1, n_lines + 1) for t in TARGETS_PER_LINE)


def target_matrix(outputs: Sequence[Sequence[LineOutputs]]) -> np.ndarray:
    """Targets ``(n, lines x targets)`` from the results of ``n`` points."""
    return np.asarray(
        [[getattr(line, t) for line in point for t in TARGETS_PER_LINE] for point in outputs],
        dtype=np.float64,
    )


def _columns(names: Sequence[str], target: str) -> list[int]:
    return [i for i, n in enumerate(names) if n.endswith("/" + target)]


def cell_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: Sequence[str]) -> dict[str, float]:
    """Errors on a test set in physical units, plus the two numbers of the reading."""
    error = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    metrics: dict[str, float] = {}
    for i, name in enumerate(names):
        metrics[f"median_abs/{name}"] = float(np.median(error[:, i]))
        metrics[f"p95_abs/{name}"] = float(np.percentile(error[:, i], 95))
        metrics[f"max_abs/{name}"] = float(error[:, i].max())
    fuel = _columns(names, "fuel_kg_per_h")
    true_total = y_true[:, fuel].sum(axis=1)
    relative = 100.0 * np.abs(y_pred[:, fuel].sum(axis=1) - true_total) / np.abs(true_total)
    metrics["median_rel/total_fuel"] = float(np.median(relative))
    metrics["p95_rel/total_fuel"] = float(np.percentile(relative, 95))
    metrics["max_rel/total_fuel"] = float(relative.max())
    exhaust = [names[i] for i in _columns(names, "exhaust_celsius")]
    metrics["exhaust_p95_k"] = max(metrics[f"p95_abs/{n}"] for n in exhaust)
    metrics["fuel_p95_percent"] = metrics["p95_rel/total_fuel"]
    return metrics


def make_model(name: str, settings: MlpSettings | None, seed: int) -> Surrogate:
    """A fresh model instance (factory, so that no state lives across cells)."""
    if name == "linear":
        return PolynomialRegression(1)
    if name == "quadratic":
        return PolynomialRegression(2)
    if name == "mlp":
        if settings is None:
            raise ValueError("the MLP needs settings")
        return MlpRegressor(settings, seed)
    raise ValueError(f"unknown model {name!r}")


def fit_and_score(
    name: str,
    settings: MlpSettings | None,
    seed: int,
    train: tuple[np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray],
    names: Sequence[str],
) -> dict[str, float]:
    """Fit one model on scaled training inputs and score it on the test set."""
    model = make_model(name, settings, seed).fit(*train)
    return cell_metrics(test[1], model.predict(test[0]), names)


def _criterion(metrics: Mapping[str, float], config: StudyConfig) -> float:
    return 0.5 * (
        metrics["exhaust_p95_k"] / config.threshold_exhaust_k
        + metrics["fuel_p95_percent"] / config.threshold_fuel_percent
    )


def pilot_selection(cells: Sequence[Mapping[str, Any]], config: StudyConfig) -> dict[str, Any]:
    """Pilot setting with the lowest mean criterion (mean over pilot seeds and budgets).

    The criterion of a cell is the mean of both reading numbers, each divided by its threshold.
    Ties go to the setting listed first in the grid.
    """
    by_setting: dict[int, list[float]] = {}
    for cell in cells:
        by_setting.setdefault(int(cell["setting"]), []).append(_criterion(cell["metrics"], config))
    means = {k: float(np.mean(v)) for k, v in sorted(by_setting.items())}
    best = min(means, key=lambda k: (means[k], k))
    return {"criterion": means, "selected_setting": best}


def _setting(values: Mapping[str, Any]) -> dict[str, Any]:
    width = int(values["width"])
    return {
        "hidden": [width, width],
        "learning_rate": float(values["learning_rate"]),
        "steps": int(values["steps"]),
        "weight_decay": float(values["weight_decay"]),
    }


def _product(axes: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    if set(axes) != set(PILOT_AXES) or any(len(axes[a]) == 0 for a in PILOT_AXES):
        raise ValueError(f"pilot.axes must give at least one level for each of {PILOT_AXES}")
    return [
        _setting(dict(zip(PILOT_AXES, combo, strict=True)))
        for combo in iter_product(*(axes[a] for a in PILOT_AXES))
    ]


def pilot_grid(
    pilot: Mapping[str, Any], axes: Mapping[str, Sequence[Any]] | None = None
) -> list[dict[str, Any]]:
    """Settings of the pilot in their tie-break order.

    ``pilot.grid`` (a list of settings, docs/45) is returned as it is. ``pilot.axes`` is a full
    factorial over :data:`PILOT_AXES`. With ``axes`` (the axes after an extension) the settings
    of ``pilot.axes`` keep their positions and the new ones follow, so that an index names the
    same setting in both rounds.
    """
    if "axes" not in pilot:
        if axes is not None:
            raise ValueError("a pilot given as a list of settings has no extension")
        return [dict(setting) for setting in pilot["grid"]]
    first = _product(pilot["axes"])
    if axes is None:
        return first
    return first + [setting for setting in _product(axes) if setting not in first]


def pilot_extension(pilot: Mapping[str, Any], selected: Mapping[str, Any]) -> dict[str, Any]:
    """Axes after the one automatic extension, and the directions that were extended.

    Rule (``pilot.extension``, fixed before the pilot): on every axis on which the selected
    setting lies at the first or the last level, the level given for that side (``lower`` /
    ``upper``) is added. An axis without a level for that side is not extended (reported as
    ``at_edge_without_extension``). The extended grid is the full factorial of the new axes; it
    is not extended again.
    """
    axes = {a: list(pilot["axes"][a]) for a in PILOT_AXES}
    value = {
        "learning_rate": float(selected["learning_rate"]),
        "width": int(selected["hidden"][0]),
        "steps": int(selected["steps"]),
        "weight_decay": float(selected["weight_decay"]),
    }
    rule = pilot.get("extension") or {}
    extended: dict[str, str] = {}
    unextended: dict[str, str] = {}
    for axis in PILOT_AXES:
        levels = pilot["axes"][axis]
        sides = []
        if value[axis] == levels[0]:
            sides.append("lower")
        if value[axis] == levels[-1]:
            sides.append("upper")
        for side in sides:
            new = (rule.get(axis) or {}).get(side)
            if new is None:
                unextended[axis] = side if axis not in unextended else "both"
            elif side == "lower":
                axes[axis].insert(0, new)
                extended[axis] = side if axis not in extended else "both"
            else:
                axes[axis].append(new)
                extended[axis] = side if axis not in extended else "both"
    return {"axes": axes, "extended": extended, "at_edge_without_extension": unextended}


def sheet_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    names: Sequence[str],
    reported: Mapping[str, Sequence[float | None]] | None = None,
) -> dict[str, Any]:
    """Errors of a surrogate at the points of a sweep sheet (one line).

    The truth ``y_true`` is our own Dyssol run at each point. ``exhaust_abs_k`` and
    ``fuel_rel_percent`` are per point, in the order of the sheet; the two maxima are the numbers
    of the reading. With ``reported`` (the sheet's values per quantity) the prediction is also
    compared with the sheet as :func:`twinlink_ml.dyssol_steady.sweep_sheet.compare` does it
    (descriptive: counts within half a unit / one unit of the last reported decimal).
    """
    exhaust, fuel = _columns(names, "exhaust_celsius"), _columns(names, "fuel_kg_per_h")
    if len(exhaust) != 1 or len(fuel) != 1:
        raise ValueError("the sheet test set is defined for one line")
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    exhaust_error = np.abs(y_pred[:, exhaust[0]] - y_true[:, exhaust[0]])
    fuel_error = (
        100.0 * np.abs(y_pred[:, fuel[0]] - y_true[:, fuel[0]]) / np.abs(y_true[:, fuel[0]])
    )
    out: dict[str, Any] = {
        "exhaust_abs_k": exhaust_error.tolist(),
        "fuel_rel_percent": fuel_error.tolist(),
        "exhaust_max_k": float(exhaust_error.max()),
        "fuel_max_percent": float(fuel_error.max()),
    }
    if reported is not None:
        per_key = {
            key: compare(y_pred[:, _columns(names, SHEET_TARGETS[key])[0]].tolist(), reported[key])
            for key in RESULT_KEYS
        }
        out["against_sheet"] = {"per_quantity": per_key, "total": totals(per_key)}
    return out


def sheet_reading(
    cells: Sequence[Mapping[str, Any]], config: StudyConfig, model: str | None, n: int | None
) -> dict[str, Any]:
    """Reading on the sheet test set for the model of the verdict at its ``N*``.

    ``B0``  the median over the seeds of the LARGEST error over the points of the sheet is at or
            below the threshold for both numbers (exhaust temperature [K], fuel [%]).
    ``B1``  otherwise.
    Without a model (verdict ``D3``) there is nothing to read: ``not applicable``.
    """
    if model is None or n is None:
        return {"verdict": "not applicable", "model": None, "n": None}
    group = [c["metrics_sheet"] for c in cells if c["model"] == model and c["n"] == n]
    if len(group) != len(config.seeds):
        raise ValueError(f"{model} at N={n}: {len(group)} cells, expected one per seed")
    exhaust = [g["exhaust_max_k"] for g in group]
    fuel = [g["fuel_max_percent"] for g in group]
    met = bool(
        np.median(exhaust) <= config.threshold_exhaust_k
        and np.median(fuel) <= config.threshold_fuel_percent
    )
    return {
        "verdict": "B0" if met else "B1",
        "model": model,
        "n": n,
        "points": len(group[0]["exhaust_abs_k"]),
        "exhaust_max_k": {
            "median": float(np.median(exhaust)),
            "range": [min(exhaust), max(exhaust)],
        },
        "fuel_max_percent": {"median": float(np.median(fuel)), "range": [min(fuel), max(fuel)]},
        "seeds_meeting_both": sum(
            e <= config.threshold_exhaust_k and f <= config.threshold_fuel_percent
            for e, f in zip(exhaust, fuel, strict=True)
        ),
    }


def prediction_check(result: Mapping[str, Any], prediction: Mapping[str, Any]) -> dict[str, Any]:
    """Compare a prediction of ``N*`` (``{model, n_star}``, fixed before the data) with a result."""
    model, predicted = str(prediction["model"]), int(prediction["n_star"])
    found = result["n_star"][model]
    return {
        "model": model,
        "predicted_n_star": predicted,
        "found_n_star": found,
        "confirmed": found == predicted,
        "data_need_of_the_reading": result["data_need_runs"],
        "reached_by": result["reached_by"],
    }


def control_by_bins(
    values: Sequence[float],
    temperature_k: Sequence[float],
    relative_percent: Sequence[float],
    edges: Sequence[float],
) -> list[dict[str, Any]]:
    """Largest deviation of the control per bin of an input (``edges`` increasing).

    Bin ``k`` holds ``edges[k] <= value < edges[k + 1]``; the last bin includes its upper edge. A
    value outside all bins raises (the table must cover every point).
    """
    edges = [float(e) for e in edges]
    if len(edges) < 2 or any(a >= b for a, b in pairwise(edges)):
        raise ValueError("bin edges must increase")
    rows: list[dict[str, Any]] = [
        {
            "lower": lo,
            "upper": hi,
            "points": 0,
            "max_temperature_k": 0.0,
            "max_relative_percent": 0.0,
        }
        for lo, hi in pairwise(edges)
    ]
    for value, kelvin, percent in zip(values, temperature_k, relative_percent, strict=True):
        if not edges[0] <= value <= edges[-1]:
            raise ValueError(f"value {value} outside the bins")
        k = min(int(np.searchsorted(edges, value, side="right")) - 1, len(rows) - 1)
        rows[k]["points"] += 1
        rows[k]["max_temperature_k"] = max(rows[k]["max_temperature_k"], float(kelvin))
        rows[k]["max_relative_percent"] = max(rows[k]["max_relative_percent"], float(percent))
    return rows


def learning_table(
    cells: Sequence[Mapping[str, Any]], config: StudyConfig, model: str
) -> tuple[dict[int, dict[str, Any]], int | None]:
    """Rows of one model over ``N`` (feasibility, medians, ``met``) and its ``N*``.

    The rule of :func:`reading` for a single model; ``cells`` as there.
    """
    rows: dict[int, dict[str, Any]] = {}
    for n in config.budgets:
        group = [c["metrics"] for c in cells if c["model"] == model and c["n"] == n]
        if len(group) != len(config.seeds):
            raise ValueError(f"{model} at N={n}: {len(group)} cells, expected one per seed")
        exhaust = [m["exhaust_p95_k"] for m in group]
        fuel = [m["fuel_p95_percent"] for m in group]
        rows[n] = {
            "seeds_meeting_both": sum(
                e <= config.threshold_exhaust_k and f <= config.threshold_fuel_percent
                for e, f in zip(exhaust, fuel, strict=True)
            ),
            "exhaust_p95_k": {
                "median": float(np.median(exhaust)),
                "range": [min(exhaust), max(exhaust)],
            },
            "fuel_p95_percent": {
                "median": float(np.median(fuel)),
                "range": [min(fuel), max(fuel)],
            },
            "met": bool(
                np.median(exhaust) <= config.threshold_exhaust_k
                and np.median(fuel) <= config.threshold_fuel_percent
            ),
        }
    n_star = next(
        (n for n in config.budgets if all(rows[m]["met"] for m in config.budgets if m >= n)),
        None,
    )
    return rows, n_star


def reading(cells: Sequence[Mapping[str, Any]], config: StudyConfig) -> dict[str, Any]:
    """Pre-registered reading: feasibility per cell group first, then ``N*`` and the verdict.

    ``cells``: mappings with ``model``, ``n``, ``seed`` and ``metrics``.

    Verdicts
    --------
    ``D1``  the MLP has an ``N*`` and no baseline has one at or below it: an MLP is needed,
            the data need is ``N*`` of the MLP.
    ``D2``  a baseline has an ``N*`` at or below that of the MLP (or the MLP has none): a
            regression suffices; the data need is the smallest baseline ``N*`` (the simpler
            model at equal ``N*``).
    ``D3``  no model meets the thresholds up to the largest ``N``: the data need exceeds the grid.
    """
    table: dict[str, dict[int, dict[str, Any]]] = {}
    n_star: dict[str, int | None] = {}
    for model in MODELS:
        table[model], n_star[model] = learning_table(cells, config, model)
    reached = {m: n for m, n in n_star.items() if n is not None}
    baseline = min(
        (m for m in BASELINES if m in reached),
        key=lambda m: (reached[m], BASELINES.index(m)),
        default=None,
    )
    if not reached:
        verdict, need, by = "D3", None, None
    elif baseline is not None and ("mlp" not in reached or reached[baseline] <= reached["mlp"]):
        verdict, need, by = "D2", reached[baseline], baseline
    else:
        verdict, need, by = "D1", reached["mlp"], "mlp"
    return {
        "verdict": verdict,
        "data_need_runs": need,
        "reached_by": by,
        "n_star": n_star,
        "table": table,
        "thresholds": {
            "exhaust_p95_k": config.threshold_exhaust_k,
            "fuel_p95_percent": config.threshold_fuel_percent,
        },
    }


def rank_tests(cells: Sequence[Mapping[str, Any]], config: StudyConfig) -> list[dict[str, Any]]:
    """MLP against each baseline, paired over the seeds, Holm per family.

    A family is one baseline and one reading number over all ``N`` of the grid. The test is on
    the mean of the paired differences (baseline minus MLP; positive = MLP better).
    """
    out: list[dict[str, Any]] = []
    for baseline in BASELINES:
        for metric in ("exhaust_p95_k", "fuel_p95_percent"):
            rows = []
            for n in config.budgets:

                def values(model: str, n: int = n, metric: str = metric) -> dict[int, float]:
                    return {
                        int(c["seed"]): float(c["metrics"][metric])
                        for c in cells
                        if c["model"] == model and c["n"] == n
                    }

                test = paired_comparison(
                    values("mlp"),
                    values(baseline),
                    candidate_label="mlp",
                    reference_label=baseline,
                    confidence=1.0 - config.alpha,
                )
                rows.append((n, test))
            usable = [(n, t) for n, t in rows if not t.degenerate]
            rejected = holm([t.p_value for _, t in usable], config.alpha) if usable else []
            flags = {n: bool(r) for (n, _), r in zip(usable, rejected, strict=True)}
            for n, t in rows:
                out.append(
                    {
                        "family": f"mlp_vs_{baseline}/{metric}",
                        "n": n,
                        "mean_difference": t.mean_difference,
                        "threshold": t.threshold,
                        "p_value": t.p_value,
                        "degenerate": t.degenerate,
                        "resolved_holm": flags.get(n, False),
                        "tests_in_family": len(usable),
                    }
                )
    return out
