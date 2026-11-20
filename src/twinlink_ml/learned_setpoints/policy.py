"""Family P: policies that predict the commands directly, the guard and the range check.

Arms (decision 11), all trained by imitation on the labels of :mod:`.labels`:

P-lin
    ridge regression on the hand-made inputs (:data:`~.labels.HAND_MADE_FEATURES`); baseline.
P-mlp
    multilayer perceptron (two hidden layers, tanh) on all inputs.
P-gru
    GRU over the decisions of the day on all inputs, the hidden state carried from the start of
    the day.

Inputs and outputs are standardised with the statistics of the training rows. P-mlp and P-gru
stop on the validation DAYS of their cell (day-level split, drawn with the seed); P-lin has
nothing to stop and uses every day of its cell. Inference runs in NumPy from the weights.

:class:`TargetGuard` is the same for every learned arm of the study: it keeps the commands in the
action box and the spray rate in the band, keeps the daily target reachable (to within
``_TARGET_SLACK_KG``) and sprays the remainder in the last hour as C2 does; every intervention is
counted.

Range check (addition C; descriptive): per input the minimum and maximum over the labels of the
cell; a decision counts as outside if at least one input is outside. The check per input does not
capture unfamiliar combinations of inputs and tends to underestimate the distribution shift.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.load_following.config import LoadFollowingConfig
from twinlink_ml.load_following.controllers import LoadController
from twinlink_ml.load_following.market import DayScenario

from .config import PolicySettings
from .gru_mpc import gru_step
from .labels import FEATURE_NAMES, HAND_MADE_FEATURES, LabelSet, decision_features

__all__ = [
    "POLICY_ARMS",
    "GruPolicy",
    "LookupPolicy",
    "MlpPolicy",
    "Policy",
    "PolicyController",
    "RidgePolicy",
    "TargetGuard",
    "fit_policy",
    "load_policy",
    "save_policy",
    "validation_days",
]

POLICY_ARMS: tuple[str, ...] = ("P-lin", "P-mlp", "P-gru")
_RANGE_TOLERANCE = 1e-9
_TARGET_SLACK_KG = 0.02
"""Slack of the target guard before the last hour (kg of suspension per day). It lets the schedule
of C2 pass untouched (its levels lie on a grid and the truth limits the rate of the spray rate);
the last hour sprays the remainder."""


def _statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return mean, np.where(std < 1e-6, 1.0, std)


class Policy(ABC):
    """Maps the inputs of a decision to the three commands."""

    arm: str = "abstract"

    def __init__(self, feature_index: np.ndarray, low: np.ndarray, high: np.ndarray) -> None:
        self.feature_index = np.asarray(feature_index, dtype=np.int64)
        self.input_low = np.asarray(low, dtype=np.float64)
        self.input_high = np.asarray(high, dtype=np.float64)

    def reset(self) -> None:  # noqa: B027 - optional hook, only policies with a state override it
        """Start of a day."""

    @abstractmethod
    def act(self, features: np.ndarray) -> np.ndarray:
        """Commands ``(3,)`` for the full input vector ``features`` (:data:`FEATURE_NAMES`)."""

    def outside(self, features: np.ndarray) -> bool:
        """Whether at least one input of this policy lies outside the range of its labels."""
        x = np.asarray(features)[self.feature_index]
        return bool(
            np.any(x < self.input_low - _RANGE_TOLERANCE)
            or np.any(x > self.input_high + _RANGE_TOLERANCE)
        )

    @abstractmethod
    def arrays(self) -> dict[str, np.ndarray]:
        """Weights and statistics for :func:`save_policy`."""


class RidgePolicy(Policy):
    """P-lin."""

    arm = "P-lin"

    def __init__(
        self,
        feature_index: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        in_stats: tuple[np.ndarray, np.ndarray],
        out_stats: tuple[np.ndarray, np.ndarray],
        weights: np.ndarray,
    ) -> None:
        super().__init__(feature_index, low, high)
        self._in, self._out, self._weights = in_stats, out_stats, weights

    def act(self, features: np.ndarray) -> np.ndarray:
        x = (np.asarray(features)[self.feature_index] - self._in[0]) / self._in[1]
        return np.asarray(self._out[0] + self._out[1] * (x @ self._weights))

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "in_mean": self._in[0], "in_scale": self._in[1],
            "out_mean": self._out[0], "out_scale": self._out[1], "weights": self._weights,
        }  # fmt: skip


class MlpPolicy(Policy):
    """P-mlp."""

    arm = "P-mlp"

    def __init__(
        self,
        feature_index: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        in_stats: tuple[np.ndarray, np.ndarray],
        out_stats: tuple[np.ndarray, np.ndarray],
        layers: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        super().__init__(feature_index, low, high)
        self._in, self._out, self._layers = in_stats, out_stats, list(layers)

    def act(self, features: np.ndarray) -> np.ndarray:
        x = (np.asarray(features)[self.feature_index] - self._in[0]) / self._in[1]
        for weight, bias in self._layers[:-1]:
            x = np.tanh(weight @ x + bias)
        weight, bias = self._layers[-1]
        return np.asarray(self._out[0] + self._out[1] * (weight @ x + bias))

    def arrays(self) -> dict[str, np.ndarray]:
        out = {
            "in_mean": self._in[0], "in_scale": self._in[1],
            "out_mean": self._out[0], "out_scale": self._out[1],
        }  # fmt: skip
        for i, (weight, bias) in enumerate(self._layers):
            out[f"w{i}"], out[f"b{i}"] = weight, bias
        return out


class GruPolicy(Policy):
    """P-gru."""

    arm = "P-gru"

    def __init__(
        self,
        feature_index: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        in_stats: tuple[np.ndarray, np.ndarray],
        out_stats: tuple[np.ndarray, np.ndarray],
        weights: dict[str, np.ndarray],
    ) -> None:
        super().__init__(feature_index, low, high)
        self._in, self._out, self._w = in_stats, out_stats, weights
        self._hidden = np.zeros((1, weights["w_hh"].shape[1]))

    def reset(self) -> None:
        """Zero hidden state at the start of the day."""
        self._hidden = np.zeros_like(self._hidden)

    def act(self, features: np.ndarray) -> np.ndarray:
        x = (np.asarray(features)[self.feature_index] - self._in[0]) / self._in[1]
        w = self._w
        self._hidden = gru_step(
            x[None, :], self._hidden, w["w_ih"], w["w_hh"], w["b_ih"], w["b_hh"]
        )
        y = self._hidden[0] @ w["w_out"].T + w["b_out"]
        return np.asarray(self._out[0] + self._out[1] * y)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "in_mean": self._in[0], "in_scale": self._in[1],
            "out_mean": self._out[0], "out_scale": self._out[1], **self._w,
        }  # fmt: skip


class LookupPolicy(Policy):
    """Returns the label whose inputs are nearest (test of the plumbing, not an arm)."""

    arm = "lookup"

    def __init__(self, labels: LabelSet) -> None:
        index = np.arange(labels.features.shape[1])
        super().__init__(index, labels.features.min(axis=0), labels.features.max(axis=0))
        self._features, self._commands = labels.features, labels.commands
        self.largest_distance = 0.0

    def act(self, features: np.ndarray) -> np.ndarray:
        distance = np.abs(self._features - np.asarray(features)[None, :]).max(axis=1)
        row = int(np.argmin(distance))
        self.largest_distance = max(self.largest_distance, float(distance[row]))
        return np.asarray(self._commands[row].copy())

    def arrays(self) -> dict[str, np.ndarray]:
        raise NotImplementedError("the lookup policy is not saved")


def save_policy(policy: Policy, path: Path) -> None:
    """Write a policy as ``.npz`` (never over an existing file)."""
    if path.exists():
        raise FileExistsError(f"{path} exists -- policies are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        arm=np.array(policy.arm),
        feature_index=policy.feature_index,
        input_low=policy.input_low,
        input_high=policy.input_high,
        **policy.arrays(),  # type: ignore[arg-type]
    )


def load_policy(path: Path) -> Policy:
    """Read a policy written by :func:`save_policy`."""
    with np.load(path) as data:
        a = {key: np.array(data[key]) for key in data.files}
    arm = str(a["arm"])
    head = (a["feature_index"], a["input_low"], a["input_high"])
    stats = ((a["in_mean"], a["in_scale"]), (a["out_mean"], a["out_scale"]))
    if arm == "P-lin":
        return RidgePolicy(*head, *stats, a["weights"])
    if arm == "P-mlp":
        n_layers = sum(1 for key in a if key.startswith("w") and key[1:].isdigit())
        return MlpPolicy(*head, *stats, [(a[f"w{i}"], a[f"b{i}"]) for i in range(n_layers)])
    if arm == "P-gru":
        names = ("w_ih", "w_hh", "b_ih", "b_hh", "w_out", "b_out")
        return GruPolicy(*head, *stats, {name: a[name] for name in names})
    raise ValueError(f"unknown policy arm {arm!r}")


def validation_days(days: Sequence[int], fraction: float, seed: int) -> list[int]:
    """Validation days of a cell: ``round(fraction * n)``, at least one, drawn with ``seed``."""
    count = max(1, round(fraction * len(days)))
    chosen = np.random.default_rng([seed, len(days), 7]).choice(
        len(days), size=count, replace=False
    )
    return sorted(int(days[i]) for i in chosen)


def _train_network(
    network: Any,
    forward: Any,
    train: tuple[Any, Any],
    validation: tuple[Any, Any],
    settings: PolicySettings,
    batch_size: int,
    generator: Any,
) -> dict[str, Any]:
    """Adam with early stopping on the validation loss; the best weights are restored."""
    import torch

    optimiser = torch.optim.Adam(network.parameters(), lr=settings.learning_rate)
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    n = train[0].shape[0]
    epoch = 0
    for epoch in range(1, settings.max_epochs + 1):
        network.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            rows = order[start : start + batch_size]
            optimiser.zero_grad()
            loss = torch.mean((forward(train[0][rows]) - train[1][rows]) ** 2)
            loss.backward()  # type: ignore[no-untyped-call]
            optimiser.step()
        network.eval()
        with torch.no_grad():
            value = float(torch.mean((forward(validation[0]) - validation[1]) ** 2))
        if value < best - 1e-9:
            best, best_epoch, stale = value, epoch, 0
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
        else:
            stale += 1
            if stale >= settings.patience:
                break
    if best_state is not None:
        network.load_state_dict(best_state)
    return {"epochs": epoch, "best_epoch": best_epoch, "validation_loss": best}


def fit_policy(
    arm: str, labels: LabelSet, days: Sequence[int], settings: PolicySettings, seed: int
) -> tuple[Policy, dict[str, Any]]:
    """Fit the policy ``arm`` on the labels of ``days``; returns it with its training record."""
    cell = labels.subset(days)
    if cell.features.shape[0] == 0:
        raise ValueError("no labels for these days")
    names = HAND_MADE_FEATURES if arm == "P-lin" else FEATURE_NAMES
    index = np.array([FEATURE_NAMES.index(name) for name in names])
    low, high = cell.features[:, index].min(axis=0), cell.features[:, index].max(axis=0)
    info: dict[str, Any] = {"arm": arm, "days": [int(d) for d in days], "rows": len(cell.day)}
    if arm == "P-lin":
        x, y = cell.features[:, index], cell.commands
        in_stats, out_stats = _statistics(x), _statistics(y)
        xs, ys = (x - in_stats[0]) / in_stats[1], (y - out_stats[0]) / out_stats[1]
        gram = xs.T @ xs / len(xs) + settings.ridge * np.eye(xs.shape[1])
        weights = np.linalg.solve(gram, xs.T @ ys / len(xs))
        info["train_loss"] = float(np.mean((xs @ weights - ys) ** 2))
        return RidgePolicy(index, low, high, in_stats, out_stats, weights), info

    import torch

    torch.set_num_threads(1)
    held_out = validation_days(days, settings.validation_fraction, seed)
    fit_days = [d for d in days if d not in held_out]
    if not fit_days:
        raise ValueError("a cell needs at least two days (one for validation)")
    part, val = labels.subset(fit_days), labels.subset(held_out)
    in_stats, out_stats = _statistics(part.features[:, index]), _statistics(part.commands)
    info.update({"validation_days": held_out, "fit_rows": len(part.day)})
    generator = torch.Generator().manual_seed(
        int(np.random.SeedSequence([seed, len(days), 11]).generate_state(1)[0])
    )
    torch.manual_seed(int(np.random.SeedSequence([seed, len(days), 13]).generate_state(1)[0]))

    def tensors(rows: LabelSet) -> tuple[Any, Any]:
        x = (rows.features[:, index] - in_stats[0]) / in_stats[1]
        y = (rows.commands - out_stats[0]) / out_stats[1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    if arm == "P-mlp":
        hidden = settings.hidden_size
        network = torch.nn.Sequential(
            torch.nn.Linear(len(index), hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 3),
        )  # fmt: skip
        info.update(
            _train_network(network, network, tensors(part), tensors(val), settings, 256, generator)
        )
        layers = [
            (
                m.weight.detach().numpy().astype(np.float64),
                m.bias.detach().numpy().astype(np.float64),
            )
            for m in network
            if isinstance(m, torch.nn.Linear)
        ]
        return MlpPolicy(index, low, high, in_stats, out_stats, layers), info
    if arm == "P-gru":
        core = torch.nn.GRU(len(index), settings.gru_hidden_size, batch_first=True)
        head = torch.nn.Linear(settings.gru_hidden_size, 3)
        module = torch.nn.ModuleDict({"core": core, "head": head})

        def stacked(rows: LabelSet) -> tuple[Any, Any]:
            x, y = tensors(rows)
            order = rows.sequences()
            return torch.stack([x[i] for i in order]), torch.stack([y[i] for i in order])

        def forward(x: Any) -> Any:
            return head(core(x)[0])

        train, held = stacked(part), stacked(val)
        info.update(_train_network(module, forward, train, held, settings, 16, generator))
        state = {k: v.detach().numpy().astype(np.float64) for k, v in module.state_dict().items()}
        gru_weights = {
            "w_ih": state["core.weight_ih_l0"], "w_hh": state["core.weight_hh_l0"],
            "b_ih": state["core.bias_ih_l0"], "b_hh": state["core.bias_hh_l0"],
            "w_out": state["head.weight"], "b_out": state["head.bias"],
        }  # fmt: skip
        return GruPolicy(index, low, high, in_stats, out_stats, gru_weights), info
    raise ValueError(f"unknown policy arm {arm!r}")


class PolicyController(LoadController):
    """A policy in the loop: builds the inputs of the moment and applies the policy."""

    def __init__(
        self, policy: Policy, config: LoadFollowingConfig, margin: float, reference_tariff: float
    ) -> None:
        super().__init__()
        self._policy = policy
        self._config = config
        self._margin = margin
        self._reference = reference_tariff
        self._day: DayScenario | None = None
        self.info.update({"decisions": 0, "outside_label_range": 0})

    def reset(self, day: DayScenario, state: np.ndarray, command: np.ndarray) -> None:  # noqa: ARG002 - shared interface
        """Start a day."""
        self._day = day
        self._policy.reset()

    def command(
        self,
        k: int,
        state: np.ndarray,  # noqa: ARG002 - shared interface
        last_command: np.ndarray,
        disturbance: np.ndarray,
        sprayed_kg: float,
    ) -> np.ndarray:
        """Command of the policy (before the guard)."""
        assert self._day is not None
        features = decision_features(
            self._config,
            self._day,
            k,
            disturbance,
            sprayed_kg,
            last_command,
            self._margin,
            self._reference,
        )
        self.info["decisions"] += 1
        self.info["outside_label_range"] += int(self._policy.outside(features))
        return self._policy.act(features)


class TargetGuard(LoadController):
    """Keeps box, band and daily target for any inner controller; counts its interventions."""

    def __init__(
        self, inner: LoadController, config: LoadFollowingConfig, band: tuple[float, float]
    ) -> None:
        super().__init__()
        self.inner = inner
        self._config = config
        self._band = band
        actions = config.process.actions
        self._low = np.array(
            [actions.inlet_temperature_celsius.minimum, actions.gas_velocity_m_per_s.minimum]
        )
        self._high = np.array(
            [actions.inlet_temperature_celsius.maximum, actions.gas_velocity_m_per_s.maximum]
        )
        self._day: DayScenario | None = None
        self._last_level = config.nominal_spray_rate
        self._counts = {"guard_box": 0, "guard_band": 0, "guard_target": 0, "guard_last_hour": 0}

    def reset(self, day: DayScenario, state: np.ndarray, command: np.ndarray) -> None:
        """Start a day."""
        self._day = day
        self.inner.reset(day, state, command)

    def _sync(self) -> None:
        self.info = {**self.inner.info, **self._counts}

    def command(
        self,
        k: int,
        state: np.ndarray,
        last_command: np.ndarray,
        disturbance: np.ndarray,
        sprayed_kg: float,
    ) -> np.ndarray:
        """Command of the inner controller, corrected where it leaves box, band or target."""
        assert self._day is not None
        out = np.array(
            self.inner.command(k, state, last_command, disturbance, sprayed_kg), dtype=np.float64
        )
        low, high = self._band
        tol = 1e-9
        clipped = np.clip(out[:2], self._low, self._high)
        if np.any(np.abs(clipped - out[:2]) > tol):
            self._counts["guard_box"] += 1
            out[:2] = clipped
        spray = float(out[2])
        per_hour = self._config.intervals_per_hour
        hours = self._day.hours
        remaining = self._config.nominal_spray_rate * hours - sprayed_kg
        if k // per_hour == hours - 1:
            if k % per_hour == 0:
                self._last_level = float(np.clip(remaining / 1, low, high))
            if abs(spray - self._last_level) > tol:
                self._counts["guard_last_hour"] += 1
                spray = self._last_level
        else:
            if spray < low - tol or spray > high + tol:
                self._counts["guard_band"] += 1
                spray = float(np.clip(spray, low, high))
            # the target stays reachable: neither so little now that the later intervals at the
            # upper end of the band cannot make up for it, nor so much that they cannot give way
            later_h = (hours * per_hour - k - 1) / per_hour
            least = float(
                np.clip((remaining - _TARGET_SLACK_KG - high * later_h) * per_hour, low, high)
            )
            most = float(
                np.clip((remaining + _TARGET_SLACK_KG - low * later_h) * per_hour, low, high)
            )
            if spray < least - tol or spray > most + tol:
                self._counts["guard_target"] += 1
                spray = float(np.clip(spray, least, most))
        out[2] = spray
        self._sync()
        return out

    def pre_roll_command(self, disturbance: np.ndarray, last_command: np.ndarray) -> np.ndarray:
        """Command of the inner controller in the pre-roll (box only)."""
        out = np.array(self.inner.pre_roll_command(disturbance, last_command), dtype=np.float64)  # type: ignore[attr-defined]
        out[:2] = np.clip(out[:2], self._low, self._high)
        return out

    @property
    def observes(self) -> bool:
        """Whether the inner controller wants the blocks of every interval."""
        return hasattr(self.inner, "observe")

    def observe(self, exogenous: np.ndarray, targets: np.ndarray) -> None:
        """Hand the blocks of the last interval to the inner controller."""
        self.inner.observe(exogenous, targets)  # type: ignore[attr-defined]
        self._sync()
