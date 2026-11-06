"""Ranking of the six heat-recovery cases of the partner workbook over the operating range.

Study ``docs/NN_case_ranking.md`` (number assigned at merge). The six cases of the workbook
``TwinLink Excel.xlsx`` (docs/11 section 13) are evaluated with the rebuild of
:mod:`twinlink_ml.flowsheet` over a box of eight inputs: per dryer the supply-air temperature, the
spray load (kg spray per kg dry air) and the solid fraction; shared the ambient temperature and the
relative humidity of the outside air. Everything else stays at the workbook values of docs/11
section 13.1 (:mod:`.workbook`); the workbook is never read.

* :mod:`.space` -- inputs, box, named regions, scrambled Sobol designs,
* :mod:`.workbook` -- the fixed workbook values and the mapping of a point to the flowsheet inputs,
* :mod:`.evaluator` -- ``CaseEvaluator`` (ABC), the three modes of docs/11 section 13.6, the
  evaluation with the workbook's circuit flows and with circuit flows optimised per case,
* :mod:`.ranking` -- best case, ties (exact / practical), regret, ranks,
* :mod:`.rule` -- explanatory quantities and the shallow decision tree of Q2,
* :mod:`.config` -- the study settings of ``conf/case_ranking/case_ranking.yaml``.
"""

from .config import CaseRankingConfig, load_config
from .evaluator import MODES, CaseEvaluator, FlowsheetCaseEvaluator, Mode, PointResult
from .space import INPUT_NAMES, Box, Interval, OperatingPoint
from .workbook import WORKBOOK_POINT

__all__ = [
    "INPUT_NAMES",
    "MODES",
    "WORKBOOK_POINT",
    "Box",
    "CaseEvaluator",
    "CaseRankingConfig",
    "FlowsheetCaseEvaluator",
    "Interval",
    "Mode",
    "OperatingPoint",
    "PointResult",
    "load_config",
]
