r"""Claims of ``docs/08_synthese_m2.md`` version 6 that version 5f did not have.

Imported by ``scripts/check_m2_synthesis_consistency.py --version 6`` (the default). No run, no
simulation: committed artifacts, the lab book and study documents are re-read.

Version 6 keeps the claims of version 5f wherever their sentence is still in the text (the
fragments are looked up after the renumbering of ``FRAGMENT_REWRITES``); a claim of 5f whose
sentence is gone is retired and listed, and the coverage rule of the main script then demands a
new claim for every number that is still written. This module registers those new claims, in two
kinds:

* **text claims** (``Sources.book`` / ``Sources.document``): a passage of the synthesis is named by
  its first and last words and by the findings it cites. Every numeric token of the passage (digits
  and number words, after the reference patterns are removed) must be written in the lab-book entry
  of one of those findings (the entry, its decision paragraphs and its dated addenda) or in a named
  study document. This is the rule of the synthesis itself: no number that is not in ``docs/04``,
  in a study document or in an artifact. It shows that the number is written in the cited source,
  not which quantity it belongs to;
* **artifact claims** (``report.claim``): the key numbers of the control studies are in addition
  computed from the committed analysis artifacts and compared within the rounding of the written
  number, as in version 5f.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
LAB_BOOK = REPO_ROOT / "docs" / "04_stand_ap1.md"

# (old, new): sections and appendices of version 5f that are renumbered in version 6.
FRAGMENT_REWRITES: list[tuple[str, str]] = []

# Reference patterns of version 6 in addition to those of version 5f (applied first).
EXTRA_PATTERNS: list[str] = [
    r"\b\d{4}-\d{2}-\d{2}\b",  # dates of decisions and addenda
    r"###\s+Rule\s+\d+:",
    r"###\s+\d+\.\d+\s",
    r"\b[Rr]ules?\s+\d+(?:\s*(?:,|and|to)\s*\d+)*",
    r"\b[Ss]ections?\s+\d+(?:\.\d+)?(?:\s*(?:,|and|to)\s*\d+(?:\.\d+)?)*",
    r"\b[Vv]ersions?\s+\d+[a-z]?\b",
    r"\bitems?\s+\d+(?:\s+and\s+\d+)?",
    r"\*\*E\d\.",
    r"\bE\d\b",
    r"\b2\^1[24]\b",  # base sample sizes of the sensitivity study, written as in the lab book
]

WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "twenty": 20,
    "hundred": 100,
}
WORD_PATTERN = re.compile(r"\b(" + "|".join(WORDS) + r")\b", re.IGNORECASE)
DIGIT_PATTERN = re.compile(r"\d[\d,\.]*\d|\d")
# The reference patterns of version 5f (kept in step with the main script).
BASE_PATTERNS = [
    r"Befunde?\s+\d+[a-z]?(?:\s*(?:,|and|/|to|-)\s*\d+[a-z]?)*(?:\s+(?:addendum\s+(?:[a-z]|\d)\b|[a-z]\b))?",
    r"\bFigures?\s+\d+(?:\s+(?:and|to)\s+\d+)?",
    r"docs/\d+",
    r"\b[A-Za-z]{1,3}-?\d+[a-z]?\b",
    r"R\^2",
    r"\(\d\)",
    r"\bAP\d\b",
]


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


def load(name: str) -> Any:
    path = RUNS_DIR / name
    if name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def tokens_of(passage: str) -> list[str]:
    """Numeric tokens of a passage, as the coverage rule of the main script sees them."""
    body = re.sub(r"`[^`]*`", " ", passage)
    for pattern in EXTRA_PATTERNS + BASE_PATTERNS:
        body = re.sub(pattern, " ", body)
    found = [m.group() for m in DIGIT_PATTERN.finditer(body)]
    found += [m.group().lower() for m in WORD_PATTERN.finditer(body)]
    return sorted(set(found))


def written_in(token: str, source: str) -> bool:
    """Tell whether the number is written in the source (digits with boundaries; a word also as digits)."""
    if token in WORDS:
        digits = str(WORDS[token])
        return bool(
            re.search(r"\b" + token + r"\b", source, re.IGNORECASE)
            or re.search(r"(?<![\d.,])" + digits + r"(?![\d]|[.,]\d)", source)
        )
    return bool(re.search(r"(?<![\d.,])" + re.escape(token) + r"(?![\d]|[.,]\d)", source))


class Sources:
    """Lab-book entries per finding and study documents, as normalised text."""

    def __init__(self, report: Any, text: str) -> None:
        self.report = report
        self.text = text
        self.entries: dict[int, str] = {}
        raw = LAB_BOOK.read_text(encoding="utf-8")
        heading = re.compile(r"^###\s+(.*)$", re.MULTILINE)
        marks = [(m.start(), m.group(1)) for m in heading.finditer(raw)]
        for index, (start, title) in enumerate(marks):
            stop = marks[index + 1][0] if index + 1 < len(marks) else len(raw)
            numbers: list[int] = []
            if title.startswith("Befund "):
                first = re.match(r"Befund\s+(\d+)", title)
                if first:
                    numbers = [int(first.group(1))]
            elif title.startswith("Addendum"):
                numbers = [int(n) for n in re.findall(r"Befund\s+(\d+)", title.split(":")[0])]
            for number in numbers:
                self.entries[number] = (
                    self.entries.get(number, "") + " " + normalise(raw[start:stop])
                )
        self.documents: dict[str, str] = {}

    def document_text(self, relative: str) -> str:
        if relative not in self.documents:
            self.documents[relative] = normalise((REPO_ROOT / relative).read_text(encoding="utf-8"))
        return self.documents[relative]

    def span(self, first: str, last: str) -> str:
        """The passage of the synthesis from ``first`` to ``last`` (both included); must be unique."""
        first, last = normalise(first), normalise(last)
        if self.text.count(first) != 1:
            raise SystemExit(f"anchor not unique or missing ({self.text.count(first)}): {first!r}")
        start = self.text.index(first)
        stop = self.text.find(last, start)
        if stop == -1:
            raise SystemExit(f"end anchor missing after {first!r}: {last!r}")
        return self.text[start : stop + len(last)]

    def passage(
        self,
        first: str,
        last: str,
        befunde: tuple[int, ...] = (),
        documents: tuple[str, ...] = (),
        also: dict[str, str] | None = None,
    ) -> None:
        """Text claim: every numeric token of the passage is written in one of the named sources.

        ``also``: a token mapped to the wording under which the source carries it (for example a
        value the synthesis rounds or writes as a word); the wording must be in a source.
        """
        try:
            fragment = self.span(first, last)
        except SystemExit as problem:
            self.report.results.append(
                {
                    "fragment": normalise(first),
                    "source": "passage of version 6",
                    "kind": "text",
                    "status": "DEVIATING",
                    "fragment_in_text": False,
                    "problem": str(problem),
                }
            )
            return
        self.report.fragments.append(fragment)
        missing = [n for n in befunde if n not in self.entries]
        if missing:
            raise SystemExit(f"no lab-book entry found for Befund {missing}")
        texts = [(f"docs/04 Befund {n}", self.entries[n]) for n in befunde]
        texts += [(d, self.document_text(d)) for d in documents]
        names = ", ".join(name for name, _ in texts)
        for token in tokens_of(fragment):
            wording = (also or {}).get(token, token)
            if wording == token:
                found = any(written_in(token, body) for _, body in texts)
            else:
                found = any(normalise(wording) in body for _, body in texts)
            self.report.results.append(
                {
                    "fragment": fragment[:160] + (" ..." if len(fragment) > 160 else ""),
                    "written": token,
                    "wording_in_source": wording,
                    "source": names + " (text)",
                    "kind": "text",
                    "status": "confirmed" if found else "DEVIATING",
                    "fragment_in_text": True,
                }
            )


def register(report: Any, text: str, raw: str) -> dict[str, Any]:
    """Register the claims of version 6; returns its exemptions and reference patterns."""
    sources = Sources(report, text)
    register_passages(sources)
    register_artifact_claims(report)
    labels = [raw.count(f"**{label}.**") for label in ("Rule", "Reason", "Finding", "Pitfall")]
    observations = raw.split("Four observations from the comparison of section 3:")[-1].split(
        "**Finding.**"
    )[0]
    counted = [
        (
            "Each rule has the same four parts",
            labels == [13, 13, 13, 13],
            "the four labels Rule, Reason, Finding, Pitfall occur once in each of the thirteen rules",
        ),
        (
            "Four observations from the comparison of section 3",
            len(re.findall(r"^- [*][*]", observations, re.MULTILINE)) == 4,
            "four list items follow the sentence",
        ),
        (
            "Four templates for a study",
            len(re.findall(r"^[*][*]E[0-9][.] ", raw, re.MULTILINE)) == 4,
            "appendix E has the four templates E1 to E4",
        ),
        (
            "Two words that change their meaning between the studies",
            "*margin m*" in raw and "*reading labels*" in raw,
            "the paragraph explains the two terms margin and reading labels",
        ),
        (
            "on two test sets that are always named",
            "*excited* test set" in raw and "*constant-run holdout*" in raw,
            "the two test sets of the rule: excited test set and constant-run holdout",
        ),
    ]
    exemptions = [
        {
            "exemption": literal,
            "in_text": normalise(literal) in text,
            "verified": bool(ok),
            "reason": reason,
        }
        for literal, ok, reason in counted
    ]
    return {"exemptions": exemptions, "patterns": list(EXTRA_PATTERNS)}


RULES = "CLAUDE.md"
D24 = "docs/24_dynamic_headroom.md"
D26 = "docs/26_sensitivity_sobol.md"
D27 = "docs/27_outlier_robustness.md"
D28 = "docs/28_sensor_value.md"
D29 = "docs/29_sensor_value_2.md"
D30 = "docs/30_surrogate_env.md"
D33 = "docs/33_observability.md"
D34 = "docs/34_plant_scale.md"
D37 = "docs/37_static_transfer.md"
D39 = "docs/39_hybrid_model.md"
D40 = "docs/40_plant_calibration.md"
D41 = "docs/41_hybrid_structural.md"
D42 = "docs/42_hybrid_structural_2.md"
D43 = "docs/43_load_following.md"
D44 = "docs/44_learned_setpoints.md"
D45 = "docs/45_dyssol_steady_state.md"
D46 = "docs/46_order_dependence.md"
D47 = "docs/47_module_api.md"
SECTION_11 = (100, 101, 102, 103, 106, 110, 114, 115, 118, 119, 121)


def register_passages(sources: Sources) -> None:
    """Text claims, in the order of the document."""
    p = sources.passage
    # ---- summary -----------------------------------------------------------------------------
    p(
        "Control with the known model. The energy lies",
        "direct optimisation (section 9).",
        (105, 126),
    )
    p(
        "Learned models in the control. Replacing the known model",
        "on the unsafe side (Befunde 120, 122, 124, 125).",
        (127,),
    )
    p(
        "Moisture information. Under a model mismatch",
        "decide the operating point (Befund 108).",
        (111, 112),
    )
    p(
        "Method, templates, and what is not shown.",
        "partial observability; batch operation.",
        (117,),
    )
    # ---- section 2: the rules ----------------------------------------------------------------
    p(
        "Reason. Windows of one run share its trajectory",
        "which is what a plant asks for.",
        (),
        (RULES,),
    )
    p(
        "Pitfall. A factor against persistence without its test set",
        "(section 5; Befund 127).",
        (127,),
    )
    p(
        "Finding. Befunde 67, 82, 83, 94 and 98 with their addenda.",
        '"not resolvably better" (Befund 122, section 7.3).',
        (109, 120, 122, 124, 125, 127),
    )
    p(
        "Rule. Question, design, metric, the classes of the reading",
        "has to be written down where the result is.",
        (105, 109, 110, 115, 116, 124, 125, 126, 127),
        (RULES,),
    )
    p(
        "Rule. When models or controllers are compared for use under a constraint",
        "by the upper end of the search.",
        (124, 125, 127),
        (RULES,),
    )
    p(
        "Rule. A model that is to stand inside an optimiser",
        "its members share (Befunde 113 and 122).",
        (113, 122, 125, 127),
    )
    p(
        "Rule. Where a physical model exists, the order of attempts is",
        "structural deviation is unknown (section 10).",
        (107, 120, 122, 123, 124, 125),
    )
    p(
        "Rule. A margin is reported with the family of cases",
        "a deployed controller would need.",
        (122, 126, 127),
    )
    p(
        "Rule. Before a learned controller or a reinforcement-learning agent",
        "with the true state known (section 12).",
        (105, 126, 127),
    )
    p(
        "Rule. New studies that use the steady-state optimiser set",
        "keeps most of the worse local optima.",
        (127, 129),
        (RULES,),
    )
    p(
        "Rule. A stage that runs for hours saves its results per unit",
        "wall-clock time was meant (Befund 113).",
        (105, 111, 113, 122, 127),
        (RULES,),
    )
    p(
        "Rule. Scripts are committed before they run, on a clean working tree",
        'without its label "placeholder".',
        (127, 128),
        (RULES,),
    )
    # ---- sections 3 and 4: additions ---------------------------------------------------------
    p(
        "Robustness of the ranking against outliers in the moisture signal (Befund 109).",
        "static regression, energy target and bed temperature.",
        (109,),
        (D27,),
    )
    p(
        "Time per decision of the controllers.",
        "it does not separate the methods.",
        (105, 127),
        (D24, D44),
    )
    # ---- section 5 ---------------------------------------------------------------------------
    p(
        "The data need depends on the metric (Befund 127).",
        "does not give the data need of a controller.",
        (127,),
    )
    p(
        "How many evaluations of the true model a set-point optimisation on a surrogate needs (Befund 107).",
        "must not be reported as measured values.",
        (107,),
    )
    p(
        "Pre-training on a physical model, static case (Befund 120).",
        "with 2 points (section 7.4).",
        (120,),
    )
    p(
        "Stationary Dyssol flowsheets as the data source (Befund 128).",
        "and is not used here.",
        (128,),
        (D45,),
    )
    p(
        "What was established about reducing the data need, and what was not.",
        "with few runs (Befund 122).",
        (94, 107, 120, 122, 127),
    )
    # ---- section 6 ---------------------------------------------------------------------------
    p(
        "Truth in every study of this section: the grey-box model of the laboratory line.",
        "is the subject of section 7.",
        (105,),
        (D24,),
    )
    p(
        "Continuous operation: the headroom of dynamic control (Befund 105).",
        "R3 and R2 were not recomputed.",
        (105,),
        (D24,),
    )
    p("Product change (Befund 105, no reading).", "per changeover and not extrapolated.", (105,))
    p(
        "Load following under an hourly price (Befund 126).",
        "C4 were not recomputed.",
        (126,),
        (D43,),
    )
    p(
        "The steady optimiser depends on the order of its queries (Befund 129).",
        "were not recomputed (`docs/12`, item 38).",
        (127, 129),
        (D46,),
    )
    p("What the studies of this section say together.", "and the fixed point differ.", (105, 126))
    # ---- section 7 ---------------------------------------------------------------------------
    p(
        "The controllers of section 6 carry the true model. This section replaces it",
        "All of it is simulation with placeholder parameters.",
        (113, 127),
    )
    p(
        "The one-step operating-point problem of Befund 96:",
        "than the optimiser on U0 with the margin m.",
        (113,),
        (D30,),
    )
    p(
        "This is the question the proposal names, the prediction of optimised transient manipulated",
        "every model is the tested, untuned configuration.",
        (127,),
        (D44,),
    )
    p(
        "Section 7.2 gives the physics its most favourable case.",
        "is known from data (section 10).",
        (120, 122, 124, 125),
        (D39, D41, D42),
    )
    p(
        "Study A (section 5) found pre-training on the simulator plus fine-tuning ahead of training from",
        "and they were not part of study A.",
        (120,),
        (D37,),
    )
    # ---- section 8 ---------------------------------------------------------------------------
    p(
        "Total Sobol indices of the steady model of the laboratory line",
        "which sits at its bound in most samples.",
        (108,),
        (D26,),
    )
    p(
        "The truth now differs from the model of the controller:",
        "and are not steps of one ladder.",
        (105, 111, 112),
        (D28, D29),
    )
    p(
        "Measure: the share A of the value of the inline sensor",
        "per 0.01 kg/kg for a single sensor.",
        (112, 116),
        (D29,),
    )
    p(
        "Signal-to-noise ratio of a moisture difference of 0.01 kg/kg",
        "section I, item 19).",
        (112, 116),
        (D33,),
    )
    # ---- section 9 ---------------------------------------------------------------------------
    p(
        "The proposal puts an emphasis on reinforcement learning in combination with time-based models. Version 5f",
        "measured again before a study of reinforcement learning.",
        (96, 105, 113, 115, 126),
    )
    # ---- section 10 --------------------------------------------------------------------------
    p(
        "A plant-near model of plant A: the structure does not carry from one day to the other",
        "placeholders) and is not quoted here.",
        (117,),
        (D34,),
    )
    p(
        "Identifiability before calibration (Befund 123).",
        "laboratory moisture values with their clock time.",
        (123,),
        (D40,),
    )
    p(
        "What this means for the simulator studies.",
        "with supply air of 83 to 85 degC.",
        (120, 124, 125),
    )
    # ---- section 11 --------------------------------------------------------------------------
    p(
        "The proposal wants the pipeline to be usable for other optimisation tasks",
        "the picture they form together.",
        (100,),
    )
    p(
        "| Befund 100 | Site pinch:",
        "are what is reliable. |",
        SECTION_11,
    )
    p(
        "The picture these studies form.",
        "with campaigns a network pays (Befunde 118, 119).",
        (*SECTION_11, 99),
    )
    # ---- sections 12 and 13 ------------------------------------------------------------------
    p(
        "The statements of sections 3 to 11 end where this list begins.",
        "framework of the consortium coordinator is open.",
        (105, 112, 116, 117, 123, 124, 125, 126, 127, 128, 129),
    )
    p(
        "Fewer seeds in three studies.",
        "were withdrawn by addenda and are not quoted.",
        (105, 109, 112, 113, 116, 122, 124, 125, 126, 127, 128),
    )
    # ---- appendices --------------------------------------------------------------------------
    p(
        "One package per study line, each importing the core",
        "the interface pydantic (extra `api`).",
        (),
        (D47,),
    )
    p(
        "Settings of the control studies (Befunde 105, 126, 127).",
        "and end at 0.12 kg/kg.",
        (105, 126, 127),
        (D24, D43, D44),
    )
    p(
        "Forecast model. A model that predicts the targets over a horizon",
        "for confirmation by the project lead.",
        (105, 122, 125, 127),
    )


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def register_artifact_claims(report: Any) -> None:
    """Key numbers of the control studies, computed from the analysis artifacts."""
    pct = 100.0

    # ---- section 13: the quality limit is a placeholder of its own (correction of a sentence of 5f) --
    limits_file = REPO_ROOT / "conf" / "headroom" / "dynamic_headroom.yaml"

    def find(node: Any, key: str) -> Any:
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for value in node.values():
                found = find(value, key)
                if found is not None:
                    return found
        return None

    limit = find(
        yaml.safe_load(limits_file.read_text(encoding="utf-8")), "product_moisture_limit_kg_per_kg"
    )
    report.claim(
        "a separate placeholder limit, 0.15 kg/kg in their main case",
        [("0.15", limit)],
        "conf/headroom/dynamic_headroom.yaml, product_moisture_limit_kg_per_kg",
    )
    wording = "kinetics at 0.20 (decoupled mechanics of Befund 64), quality limit 0.15 main case and 0.20 stress case"
    report.truth(
        "decoupled from the kinetics since Befund 64",
        wording in normalise(LAB_BOOK.read_text(encoding="utf-8")),
        "the lab book describes the set-up of Befund 80 as: " + wording,
        "docs/04_stand_ap1.md",
    )

    # ---- Befund 105: dynamic headroom ----------------------------------------------------------
    source = "dynamic_headroom_v2_analysis.json (Befund 105)"
    headroom = load("dynamic_headroom_v2_analysis.json")
    observed = headroom["observed"]
    pooled, margins = observed["pooled"], observed["margins"]
    report.claim(
        "| R0 -> R1, fixed point to quasi-stationary optimisation | 8.93 % | 0.69 .. 25.23 % | 0.00664 (R0: 0.04355) |",
        [
            ("8.93", pct * pooled["g10"]["median"]),
            ("0.69", pct * pooled["g10"]["min"]),
            ("25.23", pct * pooled["g10"]["max"]),
            ("0.00664", margins["R1"]),
            ("0.04355", margins["R0"]),
        ],
        source,
    )
    report.claim(
        "| R1 -> R3, MPC without preview | 0.55 % | 0.05 .. 1.50 % | 0.00270 |",
        [
            ("0.55", pct * pooled["g13"]["median"]),
            ("0.05", pct * pooled["g13"]["min"]),
            ("1.50", pct * pooled["g13"]["max"]),
            ("0.00270", margins["R3"]),
        ],
        source,
    )
    report.claim(
        "| R1 -> R2, perfect preview (upper bound) | 1.51 % | 1.02 .. 2.44 % | 0.00021 |",
        [
            ("1.51", pct * pooled["g12"]["median"]),
            ("1.02", pct * pooled["g12"]["min"]),
            ("2.44", pct * pooled["g12"]["max"]),
            ("0.00021", margins["R2"]),
        ],
        source,
    )
    report.claim(
        "its margin share is 0.92 % and its trajectory share -0.35 %",
        [
            ("0.92", pct * pooled["g13_margin_share"]["median"]),
            ("0.35", -pct * pooled["g13_trajectory_share"]["median"]),
        ],
        source,
    )
    report.claim(
        "of the 1.51 % of the perfect preview 1.49 % are margin",
        [
            ("1.51", pct * pooled["g12"]["median"]),
            ("1.49", pct * pooled["g12_margin_share"]["median"]),
        ],
        source,
    )
    report.truth(
        "The reading is L0",
        headroom["reading"]["class"] == "L0" and not headroom["reading"]["types_in_other_class"],
        "class L0 and no scenario type in another class",
        source,
    )
    tighten = headroom["product_change"]["by_direction"]["moisture_target_tighten"]
    report.claim(
        "R1 needs 37 to 55 min to reach the new specification and R3 0.9 to 1.8 min",
        [
            ("37", tighten["R1"]["duration_to_spec_s"]["min"] / 60.0),
            ("55", tighten["R1"]["duration_to_spec_s"]["max"] / 60.0),
            ("0.9", tighten["R3"]["duration_to_spec_s"]["min"] / 60.0),
            ("1.8", tighten["R3"]["duration_to_spec_s"]["max"] / 60.0),
        ],
        source,
    )
    report.claim(
        "the 8.93 % over the scenarios of Befund 105",
        [("8.93", pct * pooled["g10"]["median"])],
        source,
    )
    report.claim(
        "set-point optimisation saves 8.93 % of the energy in the median, an MPC adds 0.55 %, and the upper bound with perfect preview is 1.51 %",
        [
            ("8.93", pct * pooled["g10"]["median"]),
            ("0.55", pct * pooled["g13"]["median"]),
            ("1.51", pct * pooled["g12"]["median"]),
        ],
        source,
    )

    # ---- Befund 126: load following ------------------------------------------------------------
    source = "load_following_v1_analysis.json (Befund 126)"
    following = load("load_following_v1_analysis.json")
    reading, cost, lf_margins = (
        following["reading"],
        following["reading"]["J"],
        following["margins"],
    )
    parts = following["decomposition"]
    report.claim(
        "| C0 constant throughput | 0.00547 | 0.40633 |",
        [("0.00547", lf_margins["C0"]), ("0.40633", cost["C0"])],
        source,
    )
    rows = [
        (
            "| C1 price rule | 0.00742 | 0.39792 | 2.07 % | -0.48 % | 2.55 % |",
            "C1",
            "0.00742",
            "0.39792",
            "2.07",
            "0.48",
            "2.55",
            reading["gain_C1_vs_C0"],
            -1.0,
        ),
        (
            "| C2 schedule on the map | 0.00742 | 0.39795 | 2.06 % | -0.48 % | 2.54 % |",
            "C2",
            "0.00742",
            "0.39795",
            "2.06",
            "0.48",
            "2.54",
            reading["gain_C2_vs_C0"],
            -1.0,
        ),
        (
            "| C3 economic MPC | 0.00379 | 0.39463 | 2.88 % | +0.40 % | 2.48 % |",
            "C3",
            "0.00379",
            "0.39463",
            "2.88",
            "0.40",
            "2.48",
            reading["gain_C3_vs_C0"],
            1.0,
        ),
        (
            "| C4 perfect preview | 0.00023 | 0.39053 | 3.89 % | +1.24 % | 2.65 % |",
            "C4",
            "0.00023",
            "0.39053",
            "3.89",
            "1.24",
            "2.65",
            reading["headroom_preview"],
            1.0,
        ),
    ]
    for fragment, name, margin, price, gain, margin_share, schedule_share, gain_value, sign in rows:
        report.claim(
            fragment,
            [
                (margin, lf_margins[name]),
                (price, cost[name]),
                (gain, pct * gain_value),
                (margin_share, sign * pct * parts[name]["margin_share"]),
                (schedule_share, pct * parts[name]["schedule_share"]),
            ],
            source,
        )
    report.truth(
        "The reading is T1",
        reading["reading"] == "T1" and reading["valid"],
        "reading T1, valid",
        source,
    )
    report.claim(
        "the gap between C1 and C3 is 0.83 %", [("0.83", pct * reading["gap_C1_C3"])], source
    )
    report.claim(
        "of perfect information, is 1.05 %", [("1.05", pct * reading["gap_C3_C4"])], source
    )
    report.claim(
        "(1135.3 against 1132.9 kWh over the 40 days)",
        [
            ("1135.3", following["controllers"]["C1"]["energy_kwh_total"]),
            ("1132.9", following["controllers"]["C0"]["energy_kwh_total"]),
            ("40", following["n_days"]),
        ],
        source,
    )
    report.claim(
        "C1 and C2 are 0.04 % and 0.08 % more expensive",
        [
            ("0.04", -pct * following["gas"]["gain_vs_C0"]["C1"]),
            ("0.08", -pct * following["gas"]["gain_vs_C0"]["C2"]),
        ],
        source,
    )
    report.claim(
        "below 2 % in February (1.14 %), in November (1.61 %) and in the low tercile of the price spread (1.27 %)",
        [
            ("1.14", pct * following["per_month"]["2"]["gain_C1_vs_C0"]),
            ("1.61", pct * following["per_month"]["11"]["gain_C1_vs_C0"]),
            ("1.27", pct * following["per_tercile"]["low"]["gain_C1_vs_C0"]),
        ],
        source,
    )
    control = following["control_constant_price"]["gain_vs_C0"]
    report.claim(
        "C3 gains 0.49 % and C4 1.81 % without a price signal",
        [("0.49", pct * control["C3"]), ("1.81", pct * control["C4"])],
        source,
    )
    box = following["box"]
    report.claim(
        "the perfect preview gains 8.96 % and the schedule 1.98 % at a margin of 0.03047 kg/kg",
        [
            ("8.96", pct * box["gain_vs_C0"]["C4"]),
            ("1.98", pct * box["gain_vs_C0"]["C2"]),
            ("0.03047", box["margins"]["C2"]),
        ],
        source,
    )
    report.claim(
        "lowers the cost per kg of product by 2.07 % and an MPC by 2.88 %",
        [("2.07", pct * reading["gain_C1_vs_C0"]), ("2.88", pct * reading["gain_C3_vs_C0"])],
        source,
    )

    # ---- Befund 129: order dependence, and the addenda to Befunde 105 and 126 -----------------
    source = "order_dependence_v1_analysis.json (Befund 129)"
    order = load("order_dependence_v1_analysis.json")
    searched, old = order["variants"]["searched"], order["old"]
    report.claim(
        "R1 holds the limit with the margin 0.005859 instead of 0.006641, and the median saving against R0 is 9.10 % instead of 8.93 %",
        [
            ("0.005859", searched["margins"]["R1"]),
            ("0.006641", order["old_margins"]["R1"]),
            ("9.10", pct * searched["new"]["K1"]),
            ("8.93", pct * old["K1"]),
        ],
        source,
    )
    report.claim(
        "searched margin 0.007812 instead of 0.007422 its gain in the main band is 1.98 % instead of 2.06 %, and that of C1 is 2.08 % instead of 2.07 %",
        [
            ("0.007812", searched["margins"]["C2"]),
            ("0.007422", order["old_margins"]["C2"]),
            ("1.98", pct * searched["new"]["K3"]),
            ("2.06", pct * old["K3"]),
            ("2.08", pct * searched["new"]["K2"]),
            ("2.07", pct * old["K2"]),
        ],
        source,
    )
    report.claim(
        "(0.397944 against 0.397950 EUR per kg in the sum)",
        [("0.397944", old["K4"]), ("0.397950", order["old_cost_per_kg"]["C2"])],
        source,
    )
    queries = order["queries"]["pooled"]
    above_tolerance = queries["by_threshold"]["1e-06"]
    report.claim(
        "Measured on 94211 queries: the warm start returns a worse local optimum than a cold multi-start with eight starts in 1.82 % of the queries, by up to 2.58 % of the power; a single cold start is worse in 1.58 %",
        [
            ("94211", queries["n_queries"]),
            ("1.82", pct * above_tolerance["share_warm_natural_worse"]),
            ("2.58", pct * above_tolerance["worse_warm_natural_relative_power"]["max"]),
            ("1.58", pct * above_tolerance["share_cold_single_start_worse"]),
        ],
        source,
    )
    report.claim(
        "the largest change is 3.78 %", [("3.78", pct * order["reading"]["largest_change"])], source
    )
    report.truth(
        "reading O0 (no key number of Befunde 105, 126 and 127",
        order["reading"]["reading"] == "O0" and order["reading"]["valid"],
        "reading O0, valid",
        source,
    )
    report.claim(
        "(Befund 105; with the cold multi-start of the steady optimiser 9.10 % instead of 8.93 %, dated addendum)",
        [("9.10", pct * searched["new"]["K1"]), ("8.93", pct * old["K1"])],
        source,
    )

    # ---- Befund 127: learned set points --------------------------------------------------------
    source = "learned_setpoints_v1_analysis.json (Befund 127)"
    learned = load("learned_setpoints_v1_analysis.json")
    tables, cells, shares = learned["tables"], learned["cells"], learned["decomposition"]

    def cell_median(arm: str, level: str, budget: str, field: str, table: dict[str, Any]) -> float:
        seeds = tables[f"{arm}|{level}"][budget]["admissible_seeds"]
        return median([table[f"{arm}|{level}|{budget}|{seed}"][field] for seed in seeds])

    table_rows = [
        (
            "| M-quad, no feedback | 4 of 5 | +22.36 % | 0.0934 | +33.90 % | +0.46 % |",
            "M-quad",
            "none",
            "64",
            "4",
            "22.36",
            "0.0934",
            "33.90",
            "0.46",
            1.0,
        ),
        (
            "| M-quad, inline feedback | 5 of 5 | +8.51 % | 0.0203 | +3.34 % | +1.94 % |",
            "M-quad",
            "fb",
            "64",
            "5",
            "8.51",
            "0.0203",
            "3.34",
            "1.94",
            1.0,
        ),
        (
            "| M-gru, no feedback | 5 of 5 | +7.07 % | 0.0357 | +7.83 % | -1.17 % |",
            "M-gru",
            "none",
            "64",
            "5",
            "7.07",
            "0.0357",
            "7.83",
            "1.17",
            -1.0,
        ),
        (
            "| M-gru, inline feedback | 5 of 5 | +3.87 % | 0.0221 | +3.82 % | +0.05 % |",
            "M-gru",
            "fb",
            "64",
            "5",
            "3.87",
            "0.0221",
            "3.82",
            "0.05",
            1.0,
        ),
        (
            "| M-gru, laboratory sample (descriptive) | 5 of 5 | +7.62 % | 0.0348 | +7.52 % | +0.31 % |",
            "M-gru",
            "lab",
            "64",
            "5",
            "7.62",
            "0.0348",
            "7.52",
            "0.31",
            1.0,
        ),
        (
            "| D-gru | 5 of 5 | +2.98 % | 0.0088 | +0.34 % | +2.54 % |",
            "D-gru",
            "none",
            "64",
            "5",
            "2.98",
            "0.0088",
            "0.34",
            "2.54",
            1.0,
        ),
        (
            "| P-lin | 5 of 5 | +14.15 % | 0.0344 | +7.40 % | +6.74 % |",
            "P-lin",
            "none",
            "40",
            "5",
            "14.15",
            "0.0344",
            "7.40",
            "6.74",
            1.0,
        ),
        (
            "| P-gru | 5 of 5 | +10.47 % | 0.0338 | +7.22 % | +3.31 % |",
            "P-gru",
            "none",
            "40",
            "5",
            "10.47",
            "0.0338",
            "7.22",
            "3.31",
            1.0,
        ),
    ]
    for (
        fragment,
        arm,
        level,
        budget,
        admissible,
        gap,
        margin,
        margin_share,
        set_point_share,
        sign,
    ) in table_rows:
        entry = tables[f"{arm}|{level}"][budget]
        report.claim(
            fragment,
            [
                (admissible, len(entry["admissible_seeds"])),
                ("5", len(entry["seeds"])),
                (gap, pct * entry["gap"]["median"]),
                (margin, entry["margin"]["median"]),
                (margin_share, pct * cell_median(arm, level, budget, "margin_share", shares)),
                (
                    set_point_share,
                    sign * pct * cell_median(arm, level, budget, "set_point_share", shares),
                ),
            ],
            source,
        )
    mlp = tables["P-mlp|none"]["40"]
    report.claim(
        "| P-mlp | 1 of 5 | not admissible for the reading |",
        [("1", len(mlp["admissible_seeds"])), ("5", len(mlp["seeds"]))],
        source,
    )
    learned_cells = [key for key in cells if key.split("|")[0] not in ("Cfix", "C3-15", "M-tab")]
    report.claim(
        "27 of the 170 learned cells are not admissible",
        [
            ("27", sum(not cells[key]["admissible"] for key in learned_cells)),
            ("170", len(learned_cells)),
        ],
        source,
    )
    report.truth(
        "The reading is U3 in every family",
        all(row["reading"] == "U3" and row["valid"] for row in learned["readings"].values()),
        "reading U3, valid, in every family of the artifact",
        source,
    )
    gap_at = lambda arm, level, budget: pct * tables[f"{arm}|{level}"][budget]["gap"]["median"]  # noqa: E731
    report.claim(
        "Gap of M-gru at 8, 16, 32 and 64 runs: +11.28, +7.44, +7.04 and +7.07 % without feedback and +6.06, +3.98, +3.23 and +3.87 % with it",
        [
            ("11.28", gap_at("M-gru", "none", "8")),
            ("7.44", gap_at("M-gru", "none", "16")),
            ("7.04", gap_at("M-gru", "none", "32")),
            ("7.07", gap_at("M-gru", "none", "64")),
            ("6.06", gap_at("M-gru", "fb", "8")),
            ("3.98", gap_at("M-gru", "fb", "16")),
            ("3.23", gap_at("M-gru", "fb", "32")),
            ("3.87", gap_at("M-gru", "fb", "64")),
        ],
        source,
    )
    report.claim(
        "D-gru: +5.44 % at 16 and +2.98 % at 64 runs. P-gru: +21.18, +12.05 and +10.47 % at 10, 20 and 40 solved days",
        [
            ("5.44", gap_at("D-gru", "none", "16")),
            ("2.98", gap_at("D-gru", "none", "64")),
            ("21.18", gap_at("P-gru", "none", "10")),
            ("12.05", gap_at("P-gru", "none", "20")),
            ("10.47", gap_at("P-gru", "none", "40")),
        ],
        source,
    )
    twin, constant = learned["J"]["twin"], learned["J"]["C0"]
    report.claim(
        "The twin gains 2.07 % against constant throughput",
        [("2.07", pct * (constant - twin) / constant)],
        source,
    )
    report.claim(
        "D-gru at 64 runs, is 0.85 % above C0",
        [("0.85", -pct * cell_median("D-gru", "none", "64", "saving_vs_c0", cells))],
        source,
    )
    report.claim(
        "the GRU in the loop is 2.98 % more expensive per kg than the same control law with the known model, the learned steady-state map 3.87 % with an inline moisture feedback and 7.07 % without",
        [
            ("2.98", gap_at("D-gru", "none", "64")),
            ("3.87", gap_at("M-gru", "fb", "64")),
            ("7.07", gap_at("M-gru", "none", "64")),
        ],
        source,
    )
    report.truth(
        "**The reading is V2**",
        load("sensor_value_v1_analysis.json")["reading"] == "V2",
        "reading V2",
        "sensor_value_v1_analysis.json (Befund 111)",
    )
