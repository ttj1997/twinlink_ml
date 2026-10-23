"""Global sensitivity analysis (Sobol indices) of the steady-state models P1 and P2.

Study document: ``docs/26_sensitivity_sobol.md``; inputs, ranges (placeholders with reasons),
designs and criteria: ``conf/sensitivity/sobol_steady.yaml``; models:
``twinlink_ml.sensitivity.models`` (both only read).

Pre-registered (fixed in this docstring and the YAML before the first run)
--------------------------------------------------------------------------
* Indices: first order S1 (Saltelli 2010) and total ST (Jansen) from SALib, 95 % bootstrap
  intervals (1000 resamples over blocks). "CI width" = the FULL width of the interval.
* **Dominant**: ST >= 0.1. **Converged**: CI width < 0.05 for S1 and for ST of every dominant
  input at the largest base sample. Convergence is read at N/4, N/2, N of one design (nested
  prefixes = doubling); for P1 and P2 designs E, O, R N = 2^12 (2^10 .. 2^12).
* **Invalid samples** are counted per reason and never dropped silently: indices of an output use
  only blocks whose D + 2 rows are all valid for that output; the valid share is reported with
  every index. P2: fixed operating point that exceeds the quality limit is NOT invalid; it is
  counted, attributed to the inputs (share of violations per fifth of each input's range, Sobol
  indices of the violation indicator and of the moisture, one-at-a-time range ends), and (d) is
  reported twice: ``savings_all`` (every valid sample) and ``savings_fixed_feasible`` (only
  samples whose fixed point holds the limit).
* **Ranking change** (designs J and R against E, epistemic inputs, per output): the first K places
  by ST, K = number of inputs dominant in E, at least 3; changed = the ordered first K places
  differ; affected inputs and whether a swap is resolved (CIs separated in both designs) are
  reported.
* **Range robustness** (design R): the three epistemic inputs with the largest ST in E over the
  decision outputs (P2: (b) energy at the optimum, (c) optimal inlet temperature; P1: fuel,
  recovered heat, dew-point distance) get their range doubled about its centre (log scale for
  log inputs), clipped to the hard bounds of the YAML.

Precision extension (declared 2026-10-22, before its run)
---------------------------------------------------------
``--designs E --base-log2 E=14`` (artifact ``sobol_steady_e14``) repeats design E with base sample
2^14 and the same seed: the 2^12 design of ``sobol_steady_v1`` is its prefix (the indices at
N = 4096 must reproduce v1). It is a precision extension of the same pre-registered test, not a
new test: same criteria, 2^12 and 2^14 reported side by side, and any input whose classification
"dominant" (ST >= 0.1) changes is named. If the criterion is not met at 2^14 either, the base
sample is not doubled again; the inputs and indices that miss it are reported.

Designs
-------
E  epistemic inputs varied, operative inputs nominal (main design of the data request)
O  operative inputs varied, epistemic nominal (operating envelope, not running operation)
J  all inputs (check only: does the epistemic ranking of E change? P2 at 2^10, prefix 2^9)
R  E with the top-3 epistemic ranges doubled

Variants: ``p1_unpressurised`` (main P1 result), ``p1_pressurised`` (additional), ``p2``.

Output: ``artifacts/runs/<name>.json`` (indices, counts, verdicts; written once, never
overwritten). Every model run is appended to ``artifacts/runs/_<name>_cache.jsonl`` (gitignored);
``--resume`` reuses it (same config fingerprint and the same tree of ``src/`` required). Run
from the worktree with ``tools/run_here.sh scripts/run_sensitivity.py --name sobol_steady_v1``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS / "verify"))

from _common import RUNS_DIR, git, provenance, require_clean_tree  # noqa: E402

from twinlink_ml.sensitivity import (  # noqa: E402
    InputSpec,
    SensitivityStudyConfig,
    SobolIndices,
    load_sensitivity_config,
    rank_change,
    saltelli_unit,
    sobol_indices,
    verdict,
)
from twinlink_ml.sensitivity.config import DEFAULT_CONFIG_PATH  # noqa: E402
from twinlink_ml.sensitivity.models import GreyBoxModel, LineModel, SteadyModel  # noqa: E402

VARIANTS: dict[str, tuple[str, str | None]] = {
    "p1_unpressurised": ("p1", "physical_unpressurised_95"),
    "p1_pressurised": ("p1", "physical_pressurised"),
    "p2": ("p2", None),
}
DECISION_OUTPUTS: dict[str, tuple[str, ...]] = {
    "p1": ("fuel_kg_per_h", "recovered_kw", "dew_point_distance_k"),
    "p2": ("energy_optimum_kw", "inlet_temperature_optimum_celsius"),
}
DESIGN_ORDER = ("E", "R", "O", "J")
CHUNK_ROWS = {"p1": 2048, "p2": 24}

_WORKER: dict[str, Any] = {}


# ------------------------------------------------------------------------------ evaluation
def build_model(config: SensitivityStudyConfig, variant: str) -> SteadyModel:
    model, mode = VARIANTS[variant]
    if model == "p1":
        assert mode is not None
        return LineModel(config.settings["p1"], config.inputs["p1"], mode)
    return GreyBoxModel(config.settings["p2"], config.inputs["p2"])


def _init_worker(config_path: str) -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    warnings.simplefilter("ignore")
    _WORKER["config"] = load_sensitivity_config(config_path)
    _WORKER["models"] = {}


def _clean(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def _evaluate_rows(
    variant: str, rows: list[tuple[int, dict[str, float]]]
) -> list[tuple[int, dict]]:
    models = _WORKER["models"]
    if variant not in models:
        models[variant] = build_model(_WORKER["config"], variant)
    model = models[variant]
    out = []
    for idx, values in rows:
        t0 = time.perf_counter()
        e = model.evaluate(values)
        out.append(
            (
                idx,
                {
                    "o": {k: _clean(v) for k, v in e.outputs.items()},
                    "v": e.valid,
                    "r": list(e.reasons),
                    "x": e.extra,
                    "t": time.perf_counter() - t0,
                },
            )
        )
    return out


class Runner:
    """Evaluates design rows in the pool and keeps every record in the resumable cache."""

    def __init__(self, pool: ProcessPoolExecutor, cache: Path, header: dict[str, Any]) -> None:
        self.pool = pool
        self.cache = cache
        self.records: dict[str, dict] = {}
        if cache.exists():
            with cache.open(encoding="utf-8") as handle:
                first = json.loads(handle.readline())
                if first.get("header") != header:
                    raise SystemExit(f"{cache} was written by another config/src: {first}")
                for line in handle:
                    entry = json.loads(line)
                    self.records[entry["k"]] = entry["r"]
            print(f"resume: {len(self.records)} cached model runs", flush=True)
        else:
            with cache.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"header": header}) + "\n")

    def run(self, variant: str, design: str, values: list[dict[str, float]]) -> list[dict]:
        keys = [f"{variant}|{design}|{i}" for i in range(len(values))]
        missing = [i for i, k in enumerate(keys) if k not in self.records]
        size = CHUNK_ROWS[VARIANTS[variant][0]]
        chunks = [missing[i : i + size] for i in range(0, len(missing), size)]
        pending = {
            self.pool.submit(_evaluate_rows, variant, [(i, values[i]) for i in chunk])
            for chunk in chunks
        }
        done_rows, step, t0 = 0, max(1, len(missing) // 20), time.time()
        next_print = step
        with self.cache.open("a", encoding="utf-8") as handle:
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    for i, record in future.result():
                        self.records[keys[i]] = record
                        handle.write(json.dumps({"k": keys[i], "r": record}) + "\n")
                        done_rows += 1
                handle.flush()
                if done_rows >= next_print:
                    rate = done_rows / max(time.time() - t0, 1e-9)
                    left = (len(missing) - done_rows) / max(rate, 1e-9)
                    print(
                        f"  {variant}/{design}: {done_rows}/{len(missing)} "
                        f"({rate:.1f} rows/s, ~{left / 60:.1f} min left)",
                        flush=True,
                    )
                    next_print += step
        return [self.records[k] for k in keys]


# ------------------------------------------------------------------------------ designs
def design_specs(
    config: SensitivityStudyConfig, variant: str, design: str, widened: Sequence[str] = ()
) -> tuple[InputSpec, ...]:
    model = VARIANTS[variant][0]
    model_inputs = _model_inputs(config, variant)
    if design == "E":
        return tuple(s for s in model_inputs if s.group == "epistemic")
    if design == "O":
        return tuple(s for s in model_inputs if s.group == "operative")
    if design == "J":
        return model_inputs
    if design == "R":
        return tuple(
            s.widened(2.0) if s.name in widened else s
            for s in model_inputs
            if s.group == "epistemic"
        )
    raise ValueError(f"unknown design {design!r} ({model})")


_NOMINAL_INPUTS: dict[str, tuple[InputSpec, ...]] = {}


def _model_inputs(config: SensitivityStudyConfig, variant: str) -> tuple[InputSpec, ...]:
    """The model's inputs with resolved nominal values (P1: the nominal RH)."""
    if variant not in _NOMINAL_INPUTS:
        model, mode = VARIANTS[variant]
        if model == "p1":
            assert mode is not None
            _NOMINAL_INPUTS[variant] = LineModel(
                config.settings["p1"], config.inputs["p1"], mode
            ).inputs
        else:
            _NOMINAL_INPUTS[variant] = config.inputs["p2"]
    return _NOMINAL_INPUTS[variant]


def design_values(
    specs: Sequence[InputSpec], base_log2: int, seed: int
) -> tuple[np.ndarray, list[dict[str, float]]]:
    unit = saltelli_unit(len(specs), base_log2, seed)
    cols = [s.value(unit[:, j]) for j, s in enumerate(specs)]
    values = [{s.name: float(cols[j][i]) for j, s in enumerate(specs)} for i in range(len(unit))]
    return unit, values


# ------------------------------------------------------------------------------ analysis
def _arrays(records: list[dict], output: str) -> tuple[np.ndarray, np.ndarray]:
    y = np.array([np.nan if r["o"][output] is None else r["o"][output] for r in records])
    valid = np.array([bool(r["v"][output]) for r in records])
    return y, valid


def analyse_design(
    config: SensitivityStudyConfig,
    model: SteadyModel,
    specs: Sequence[InputSpec],
    records: list[dict],
    base_log2: int,
) -> tuple[dict[str, Any], dict[str, SobolIndices]]:
    crit = config.criteria
    names = [s.name for s in specs]
    full: dict[str, SobolIndices] = {}
    per_output: dict[str, Any] = {}
    prefixes = [b for b in (base_log2 - 2, base_log2 - 1, base_log2) if b >= 2]
    for out in model.outputs:
        y, valid = _arrays(records, out.name)
        conv = {}
        for b in prefixes:
            ix = sobol_indices(
                y,
                valid,
                names,
                n_blocks=2**b,
                resamples=crit.bootstrap_resamples,
                confidence_level=crit.confidence_level,
                seed=crit.bootstrap_seed,
            )
            v = verdict(
                ix, dominant_total_index=crit.dominant_total_index, max_ci_width=crit.max_ci_width
            )
            conv[str(2**b)] = {**ix.to_dict(), "verdict": v}
            full_ix = ix
        full[out.name] = full_ix
        final = conv[str(2**base_log2)]["verdict"]
        converged_at = next(
            (int(n) for n, c in conv.items() if c["verdict"]["converged"] is True), None
        )
        per_output[out.name] = {
            "label": out.label,
            "unit": out.unit,
            "description": out.description,
            "by_base_sample": conv,
            "dominant": final["dominant"],
            "converged": final["converged"],
            "converged_from_base_sample": converged_at,
            "ranking_by_ST": full_ix.ranking(),
        }
    reasons: Counter[str] = Counter()
    any_invalid = 0
    for r in records:
        for reason in set(r["r"]):
            reasons[reason] += 1
        any_invalid += bool(r["r"])
    times = np.array([r["t"] for r in records])
    extra: dict[str, Any] = {}
    bindings = Counter(r["x"].get("binding") for r in records if "binding" in r["x"])
    if bindings:
        extra["binding_counts"] = dict(bindings)
    return {
        "inputs": [
            {
                "name": s.name,
                "group": s.group,
                "nominal": s.nominal,
                "low": s.low,
                "high": s.high,
                "scale": s.scale,
                "unit": s.unit,
            }
            for s in specs
        ],
        "base_sample": 2**base_log2,
        "model_runs": len(records),
        "invalid": {
            "rows_with_any_reason": any_invalid,
            "rows_with_any_reason_share": any_invalid / len(records),
            "rows_per_reason": dict(sorted(reasons.items())),
            "rows_invalid_per_output": {
                o.name: int(sum(not r["v"][o.name] for r in records)) for o in model.outputs
            },
        },
        "outputs": per_output,
        "eval_time_s": {"mean": float(times.mean()), "total": float(times.sum())},
        **extra,
    }, full


def violation_attribution(
    specs: Sequence[InputSpec], unit: np.ndarray, records: list[dict], bins: int = 5
) -> dict[str, Any]:
    """P2: where does the fixed operating point exceed the limit? (base rows A and B only)."""
    d = len(specs)
    rows = np.arange(len(records))
    base = (rows % (d + 2) == 0) | (rows % (d + 2) == d + 1)
    fixed_ok = np.array([r["v"]["fixed_violates_limit"] for r in records])
    viol = np.array([r["o"]["fixed_violates_limit"] == 1.0 for r in records])
    use = base & fixed_ok
    per_input = {}
    for j, s in enumerate(specs):
        edges = np.linspace(0.0, 1.0, bins + 1)
        shares = []
        for k in range(bins):
            sel = use & (unit[:, j] >= edges[k]) & (unit[:, j] <= edges[k + 1])
            shares.append(float(viol[sel].mean()) if sel.any() else None)
        per_input[s.name] = {
            "violation_share_per_fifth": shares,
            "fifth_edges": [float(x) for x in s.value(edges)],
        }
    return {
        "base_rows": int(base.sum()),
        "base_rows_fixed_valid": int(use.sum()),
        "violation_share": float(viol[use].mean()) if use.any() else None,
        "per_input": per_input,
    }


def one_at_a_time(model: SteadyModel, specs: Sequence[InputSpec]) -> dict[str, Any]:
    """Each input alone at the ends of its range, the others nominal (P2: fixed point check)."""
    out: dict[str, Any] = {}
    for s in specs:
        out[s.name] = {}
        for end, x in (("low", s.low), ("high", s.high)):
            e = model.evaluate({s.name: x})
            out[s.name][end] = {
                "value": x,
                "outputs": {k: _clean(v) for k, v in e.outputs.items()},
                "reasons": list(e.reasons),
            }
    return out


def question_table(
    config: SensitivityStudyConfig,
    specs: Sequence[InputSpec],
    indices: dict[str, SobolIndices],
    outputs: Sequence[str],
) -> dict[str, Any]:
    """Value of each epistemic input and each open question for the decision outputs.

    Per input: S1, ST and the standard deviation of the output that remains in expectation once
    the input is known exactly, sigma * sqrt(1 - S1) (independent inputs). Per question (inputs
    grouped): the sum of S1 (a lower bound of the closed first-order index of the group) and the
    sum of ST (an upper bound of the group's total index).
    """
    per_input: dict[str, Any] = {}
    per_question: dict[str, Any] = {}
    for out in outputs:
        ix = indices[out]
        sigma = ix.std
        for s in specs:
            i = ix.names.index(s.name)
            s1 = float(ix.s1[i])
            per_input.setdefault(s.name, {"question": s.question})[out] = {
                "S1": s1,
                "ST": float(ix.st[i]),
                "std_output": sigma,
                "std_after_learning": sigma * math.sqrt(max(0.0, 1.0 - max(s1, 0.0))),
            }
            q = per_question.setdefault(
                s.question, {"text": config.questions.get(s.question or "", ""), "inputs": []}
            )
            if s.name not in q["inputs"]:
                q["inputs"].append(s.name)
            agg = q.setdefault(out, {"sum_S1_lower_bound": 0.0, "sum_ST_upper_bound": 0.0})
            agg["sum_S1_lower_bound"] += max(s1, 0.0)
            agg["sum_ST_upper_bound"] += max(float(ix.st[i]), 0.0)
    return {"per_input": per_input, "per_question": per_question}


def top_epistemic(indices: dict[str, SobolIndices], outputs: Sequence[str], k: int) -> list[str]:
    """The ``k`` inputs with the largest ST over ``outputs`` (maximum over the outputs)."""
    best: dict[str, float] = {}
    for out in outputs:
        ix = indices[out]
        for i, n in enumerate(ix.names):
            if np.isfinite(ix.st[i]):
                best[n] = max(best.get(n, -np.inf), float(ix.st[i]))
    return sorted(best, key=lambda n: (-best[n], n))[:k]


def _finite(value: Any) -> Any:
    """NaN and infinities as ``None`` (strict JSON)."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer | np.bool_):
        return value.item()
    return value


# ------------------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--designs", default=",".join(DESIGN_ORDER))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--base-log2-cap",
        type=int,
        default=None,
        help="cap every base sample at 2**cap (smoke run only; needs --out-dir outside runs)",
    )
    parser.add_argument(
        "--base-log2",
        action="append",
        default=[],
        metavar="DESIGN=K",
        help="base sample 2**K for DESIGN instead of the YAML (precision extension, e.g. E=14; "
        "recorded in the artifact; same seed, so the YAML's design is a prefix of it)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(RUNS_DIR),
        help="output directory; anything but artifacts/runs is a smoke run (no clean-tree check)",
    )
    args = parser.parse_args()
    config = load_sensitivity_config(args.config)
    out_dir = Path(args.out_dir).resolve()
    smoke = out_dir != RUNS_DIR.resolve()
    if args.base_log2_cap is not None and not smoke:
        raise SystemExit("--base-log2-cap is for smoke runs only")
    override: dict[str, int] = {}
    for item in args.base_log2:
        design, _, k = item.partition("=")
        if design not in config.designs or not k.isdigit():
            raise SystemExit(
                f"--base-log2 expects DESIGN=K with a design of the YAML, got {item!r}"
            )
        if int(k) < config.designs[design].base_log2:
            raise SystemExit("--base-log2 may only enlarge a design (precision extension)")
        override[design] = int(k)
    variants = [v for v in args.variants.split(",") if v]
    designs = [d for d in DESIGN_ORDER if d in args.designs.split(",")]
    if "R" in designs and "E" not in designs:
        raise SystemExit("design R needs design E (its top-3 inputs)")
    output = out_dir / f"{args.name}.json"
    cache = out_dir / f"_{args.name}_cache.jsonl"
    if output.exists():
        raise SystemExit(f"{output} exists -- artifacts are never overwritten; use a new name")
    if cache.exists() and not args.resume:
        raise SystemExit(f"{cache} exists -- pass --resume to continue it, or use a new name")
    if smoke:
        print(f"smoke run into {out_dir}: no clean-tree requirement, not an artifact")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        require_clean_tree([output, cache])
    prov = provenance("not used", ignore_paths=[output, cache])
    header = {
        "config_fingerprint": config.fingerprint,
        "src_tree": git("rev-parse", "HEAD:src"),
        "cap": args.base_log2_cap,
        "base_log2_override": override,
    }

    def base_log2(design: str, model: str) -> int:
        b = override.get(design, config.designs[design].base_log2_for(model))
        return b if args.base_log2_cap is None else min(b, args.base_log2_cap)

    started = time.time()
    payload: dict[str, Any] = {
        "provenance": prov,
        "study": "global sensitivity analysis, steady-state models (docs/26_sensitivity_sobol.md)",
        "study_config": config.raw,
        "study_config_fingerprint": config.fingerprint,
        "smoke": smoke,
        "designs_run": designs,
        "base_log2_override": override,
        "workers": args.workers,
        "packages": {p: metadata.version(p) for p in ("SALib", "casadi", "numpy", "scipy")},
        "correlations": "not modelled: all inputs independent and uniform (see the YAML)",
        "variants": {},
    }
    warnings.simplefilter("ignore")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(args.config,)
    ) as pool:
        runner = Runner(pool, cache, header)
        for variant in variants:
            model_name, mode = VARIANTS[variant]
            model = build_model(config, variant)
            block: dict[str, Any] = {"model": model_name, "mode": mode, "designs": {}}
            if isinstance(model, GreyBoxModel):
                block["fixed_command"] = {
                    "inlet_temperature_celsius": model.fixed_command[0],
                    "gas_velocity_m_per_s": model.fixed_command[1],
                    "nominal_power_kw": model.fixed_nominal_power_kw,
                    "margin_fixed_kg_per_kg": model.margin_fixed,
                    "margin_optimum_kg_per_kg": model.margin_optimum,
                    "quality_limit_kg_per_kg": model.limit,
                }
                block["one_at_a_time"] = one_at_a_time(model, model.inputs)
            block["nominal"] = {k: _clean(v) for k, v in model.evaluate({}).outputs.items()}
            indices_by_design: dict[str, dict[str, SobolIndices]] = {}
            widened: list[str] = []
            for design in designs:
                t0 = time.time()
                spec = config.designs[design]
                specs = design_specs(config, variant, design, widened)
                b = base_log2(design, model_name)
                unit, values = design_values(specs, b, spec.seed)
                label = design if design != "R" else "R:" + "+".join(widened)
                print(f"{variant}/{design}: {len(values)} model runs (N = 2^{b})", flush=True)
                records = runner.run(variant, label, values)
                result, ix = analyse_design(config, model, specs, records, b)
                indices_by_design[design] = ix
                epistemic = [s.name for s in specs if s.group == "epistemic"]
                decision = DECISION_OUTPUTS[model_name]
                if design == "E":
                    widened = top_epistemic(ix, decision, config.criteria.robustness_top_k)
                    result["top_epistemic_for_robustness"] = widened
                    result["value_of_questions"] = question_table(
                        config,
                        specs,
                        ix,
                        [o.name for o in model.outputs],
                    )
                    if model_name == "p2":
                        result["fixed_point_violation"] = violation_attribution(
                            specs, unit, records
                        )
                if design in ("J", "R") and "E" in indices_by_design:
                    result["ranking_vs_E"] = {
                        out.name: rank_change(
                            indices_by_design["E"][out.name],
                            ix[out.name],
                            epistemic,
                            dominant_total_index=config.criteria.dominant_total_index,
                            top_k=config.criteria.robustness_top_k,
                        )
                        for out in model.outputs
                    }
                    if design == "J":
                        half = {}  # J also at half its base sample (2^9 for P2)
                        for out in model.outputs:
                            y, valid = _arrays(records, out.name)
                            ix_half = sobol_indices(
                                y,
                                valid,
                                [s.name for s in specs],
                                n_blocks=2 ** (b - 1),
                                resamples=config.criteria.bootstrap_resamples,
                                seed=config.criteria.bootstrap_seed,
                            )
                            half[out.name] = rank_change(
                                indices_by_design["E"][out.name],
                                ix_half,
                                epistemic,
                                dominant_total_index=config.criteria.dominant_total_index,
                                top_k=config.criteria.robustness_top_k,
                            )
                        result["ranking_vs_E_half_base_sample"] = half
                if design == "R":
                    result["widened"] = widened
                result["wall_time_s"] = time.time() - t0
                block["designs"][design] = result
                print(f"{variant}/{design} done in {(time.time() - t0) / 60:.1f} min", flush=True)
            payload["variants"][variant] = block
    payload["runtime_s"] = time.time() - started
    text = json.dumps(_finite(payload), indent=1, allow_nan=False, default=str)
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
