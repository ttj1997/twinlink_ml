"""Explanatory quantities and the shallow decision tree of Q2.

The rule is a description of where case 1 is best, not physics. Explanatory quantities: the eight
inputs and derived ones -- the exhaust temperatures and their difference, the heat-capacity flows
of the exhausts, the demand of each sink (supply air from ambient to its set point) and the share
of sink 1 in the total demand, and the heat each exhaust could give above ambient plus 20 K (two
exchangers of 10 K, docs/11 section 13.7). The circuit capacity flows are fixed in the variant
"fixed" and are therefore no explanatory quantity there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .space import INPUT_NAMES, ambient_humidity_kg_per_kg
from .workbook import WORKBOOK, WorkbookConstants

__all__ = ["FEATURE_NAMES", "RuleFit", "features", "fit_rule"]

FEATURE_NAMES: tuple[str, ...] = (
    *INPUT_NAMES,
    "humidity_g_per_kg",
    "exhaust_1_celsius",
    "exhaust_2_celsius",
    "exhaust_1_minus_2_k",
    "exhaust_capacity_1_kw_per_k",
    "exhaust_capacity_2_kw_per_k",
    "demand_1_kw",
    "demand_2_kw",
    "demand_share_1",
    "exhaust_available_1_kw",
    "exhaust_available_2_kw",
)

APPROACH_TOTAL_K = 20.0
"""Two exchangers of 10 K in series between exhaust and supply air (docs/11 section 13.7)."""


def features(
    points: np.ndarray,
    exhaust_celsius: np.ndarray,
    exhaust_humidity: np.ndarray,
    constants: WorkbookConstants = WORKBOOK,
) -> np.ndarray:
    """Matrix of :data:`FEATURE_NAMES`; ``points`` (n, 8), exhausts (n, 2) degC and kg/kg."""
    x = np.asarray(points, dtype=np.float64)
    t_exh = np.asarray(exhaust_celsius, dtype=np.float64)
    y_exh = np.asarray(exhaust_humidity, dtype=np.float64)
    m = constants.material
    cp_g, cp_v = m.cp_gas_kj_per_kg_k, m.cp_vapour_kj_per_kg_k
    air = np.array(constants.air_kg_per_h) / 3600.0
    t_amb = x[:, 0]
    y_in = np.array([ambient_humidity_kg_per_kg(t, r) for t, r in x[:, :2]])
    cap_exh = air[None, :] * (cp_g + y_exh * cp_v)
    cp_in = cp_g + y_in * cp_v
    demand = np.stack(
        [air[0] * cp_in * (x[:, 2] - t_amb), air[1] * cp_in * (x[:, 3] - t_amb)], axis=1
    )
    available = cap_exh * np.maximum(t_exh - t_amb[:, None] - APPROACH_TOTAL_K, 0.0)
    total = demand.sum(axis=1)
    return np.column_stack(
        [
            x,
            y_in * 1e3,
            t_exh,
            t_exh[:, 0] - t_exh[:, 1],
            cap_exh,
            demand,
            demand[:, 0] / total,
            available,
        ]
    )


@dataclass(frozen=True)
class RuleFit:
    """The fitted tree as text and its hit rates."""

    text: str
    train_accuracy: float
    test_accuracy: float
    test_balanced_accuracy: float
    test_accuracy_outside_practical_ties: float
    n_train: int
    n_test: int
    importances: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "train_accuracy": self.train_accuracy,
            "test_accuracy": self.test_accuracy,
            "test_balanced_accuracy": self.test_balanced_accuracy,
            "test_accuracy_outside_practical_ties": self.test_accuracy_outside_practical_ties,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "importances": self.importances,
        }


def fit_rule(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    clear_test: np.ndarray,
    max_depth: int,
    min_samples_leaf: int,
) -> RuleFit:
    """Depth-limited decision tree; ``clear_test`` marks test points outside the practical tie."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.tree import DecisionTreeClassifier, export_text

    tree = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0
    )
    tree.fit(x_train, y_train)
    pred = tree.predict(x_test)
    clear = np.asarray(clear_test, dtype=bool)
    single_class = len(np.unique(y_test)) < 2
    return RuleFit(
        text=export_text(tree, feature_names=list(FEATURE_NAMES), decimals=3),
        train_accuracy=float((tree.predict(x_train) == y_train).mean()),
        test_accuracy=float((pred == y_test).mean()),
        test_balanced_accuracy=float("nan")
        if single_class
        else float(balanced_accuracy_score(y_test, pred)),
        test_accuracy_outside_practical_ties=float((pred[clear] == y_test[clear]).mean())
        if clear.any()
        else float("nan"),
        n_train=len(y_train),
        n_test=len(y_test),
        importances={
            n: float(v)
            for n, v in zip(FEATURE_NAMES, tree.feature_importances_, strict=True)
            if v > 0.0
        },
    )
