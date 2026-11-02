"""Abstraction ``ProcessSimulator`` and trajectory container.

This interface is the actual yield of step 1: it is at the same time the
**interface specification towards TUHH-A and the software integration partner**. Anyone who can
deliver a dataset
that can be represented as a :class:`Trajectory` can be plugged in without changing the
training code.

Planned implementations:

* :class:`~twinlink_ml.simulation.greybox.GreyBoxFluidBedGranulator` -- now
* ``DyssolSimulator`` -- once the Python library of the
  software integration partner is available (M7)
* ``ReplaySimulator`` -- replay of measured application-partner/lab data as trajectories
* ``SurrogateSimulator`` -- learned model as environment for TD3/SAC

The rollout is implemented as a template method in the base class: a new
data source only needs to provide ``step`` and the name lists (open-closed).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .backend import Array, ArrayBackend, FloatArray

IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class StepResult:
    """Result of one sampling step."""

    state: Array
    """Zustandsvektor, Form ``(batch, n_state)``."""

    outputs: Array
    """Ausgangs-/Zielgrößen, Form ``(batch, n_output)``."""

    diagnostics: dict[str, Array]
    """Nicht-Ziel-Größen für Plausibilisierung und Constraint-Prüfung."""


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One simulation run (or a batch of them).

    The axis convention is ``(batch, time, channel)`` -- identical to the one the
    windowing in ``data/`` expects. ``run_id`` is the unit on which the splits
    are formed (**never** at window level, see work plan step 5).
    """

    run_id: tuple[str, ...]
    time_s: FloatArray
    actions: FloatArray
    disturbances: FloatArray
    states: FloatArray
    outputs: FloatArray
    measurements: FloatArray
    diagnostics: dict[str, FloatArray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    requested_actions: FloatArray | None = None
    """Aktion VOR einer Sicherheitsprojektion (z. B. Fluidisierungsfenster), Form wie
    ``actions``. ``actions`` bleibt die REALISIERTE Aktion -- sie speist Modellmerkmale und
    Massenbilanz und muss zum integrierten Zustand passen. Nur populiert, wenn
    ``projection_active`` -- auch dann, wenn die Projektion an keinem Punkt gegriffen hat
    (dann bitgleich zu ``actions``). ``None`` heißt: kein Projektionsmechanismus konfiguriert,
    nicht nur "hat nicht gegriffen" -- siehe ``projection_active``."""
    projection_active: bool = False
    """Ob ein Projektionsmechanismus fuer diesen Rollout konfiguriert war -- eine Eigenschaft
    der Simulation (des Prozesssimulators/seiner Config), nicht eines einzelnen Laufs: pro
    Rollout entweder aktiv oder nicht, deshalb ein Skalar und kein Array, obwohl ``actions``/
    ``states``/etc. Batch-Felder sind. Ob die Projektion an einem konkreten Lauf tatsächlich
    gegriffen hat, steht in ``n_projected_steps`` -- "aktiv" und "hat eingegriffen" sind zwei
    verschiedene Fakten, siehe Feld-Docstrings."""
    n_projected_steps: IntArray | None = None
    """Anzahl Schritte je Lauf, an denen die Projektion tatsaechlich eingegriffen hat, Form
    ``(n_batch,)``, dtype int. Anders als ``projection_active`` IST dies laufspezifisch (ein
    Lauf kann durchgehend im zulaessigen Fenster bleiben, ein anderer nicht). ``None`` nur
    wenn ``projection_active`` False ist; sonst immer ein Array (ggf. voller Nullen)."""

    def __post_init__(self) -> None:
        n_batch, n_time = self.actions.shape[0], self.actions.shape[1]
        if len(self.run_id) != n_batch:
            raise ValueError(
                f"run_id hat {len(self.run_id)} Einträge, Batch-Dimension ist {n_batch}"
            )
        for name in ("disturbances", "states", "outputs", "measurements"):
            array = getattr(self, name)
            if array.shape[:2] != (n_batch, n_time):
                raise ValueError(
                    f"{name} hat Form {array.shape}, erwartet ({n_batch}, {n_time}, ...)"
                )
        if self.time_s.shape[-1] != n_time:
            raise ValueError("time_s passt nicht zur Zeitdimension")
        if self.projection_active:
            if self.requested_actions is None or self.n_projected_steps is None:
                raise ValueError(
                    "projection_active=True verlangt requested_actions und "
                    "n_projected_steps -- 'aktiv' ohne Wirkungsnachweis waere nicht "
                    "auditierbar"
                )
            if self.requested_actions.shape != self.actions.shape:
                raise ValueError(
                    f"requested_actions hat Form {self.requested_actions.shape}, "
                    f"erwartet {self.actions.shape} wie actions"
                )
            if self.n_projected_steps.shape != (n_batch,):
                raise ValueError(
                    f"n_projected_steps hat Form {self.n_projected_steps.shape}, "
                    f"erwartet ({n_batch},)"
                )
        elif self.requested_actions is not None or self.n_projected_steps is not None:
            raise ValueError(
                "requested_actions/n_projected_steps gesetzt, aber projection_active=False -- "
                "widersprüchlicher Zustand (Struktur ohne Bedeutung), siehe Feld-Docstrings"
            )

    @property
    def n_runs(self) -> int:
        return int(self.actions.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.actions.shape[1])

    def select(self, index: int) -> Trajectory:
        """Extract a single run; the batch dimension is kept."""
        sl = slice(index, index + 1)
        return Trajectory(
            run_id=(self.run_id[index],),
            time_s=self.time_s,
            actions=self.actions[sl],
            disturbances=self.disturbances[sl],
            states=self.states[sl],
            outputs=self.outputs[sl],
            measurements=self.measurements[sl],
            diagnostics={k: v[sl] for k, v in self.diagnostics.items()},
            metadata=dict(self.metadata),
            requested_actions=(
                None if self.requested_actions is None else self.requested_actions[sl]
            ),
            projection_active=self.projection_active,
            n_projected_steps=(
                None if self.n_projected_steps is None else self.n_projected_steps[sl]
            ),
        )

    def concatenate(self, other: Trajectory) -> Trajectory:
        """Merge runs. Precondition: same time grid and ``projection_active``.

        Differing ``projection_active`` is not merged silently -- that would mix runs with
        and without safety projection indistinguishably, a data integrity error of the same
        kind as the already checked time-grid precondition.
        """
        if not np.allclose(self.time_s, other.time_s):
            raise ValueError("Trajektorien mit unterschiedlichem Zeitraster")
        if self.projection_active != other.projection_active:
            raise ValueError(
                "Trajektorien mit unterschiedlichem projection_active koennen nicht "
                "zusammengefuehrt werden -- das wuerde projizierte und unprojizierte "
                "Laeufe ununterscheidbar vermischen"
            )
        shared = set(self.diagnostics) & set(other.diagnostics)
        requested_actions = None
        n_projected_steps = None
        if self.projection_active:
            assert self.requested_actions is not None
            assert other.requested_actions is not None
            assert self.n_projected_steps is not None
            assert other.n_projected_steps is not None
            requested_actions = np.concatenate(
                [self.requested_actions, other.requested_actions], axis=0
            )
            n_projected_steps = np.concatenate(
                [self.n_projected_steps, other.n_projected_steps], axis=0
            )
        return Trajectory(
            run_id=self.run_id + other.run_id,
            time_s=self.time_s,
            actions=np.concatenate([self.actions, other.actions], axis=0),
            disturbances=np.concatenate([self.disturbances, other.disturbances], axis=0),
            states=np.concatenate([self.states, other.states], axis=0),
            outputs=np.concatenate([self.outputs, other.outputs], axis=0),
            measurements=np.concatenate([self.measurements, other.measurements], axis=0),
            diagnostics={
                k: np.concatenate([self.diagnostics[k], other.diagnostics[k]], axis=0)
                for k in sorted(shared)
            },
            metadata={**self.metadata, **other.metadata},
            requested_actions=requested_actions,
            projection_active=self.projection_active,
            n_projected_steps=n_projected_steps,
        )


class ProcessSimulator(ABC):
    """Source of trajectories for training and evaluation."""

    @property
    @abstractmethod
    def backend(self) -> ArrayBackend: ...

    @property
    @abstractmethod
    def action_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def disturbance_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def state_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def output_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def sample_time_s(self) -> float: ...

    @abstractmethod
    def initial_state(self, n_batch: int) -> Array:
        """State vector of shape ``(batch, n_state)`` at the initial operating point."""

    @abstractmethod
    def step(self, state: Array, action: Array, disturbance: Array) -> StepResult:
        """One sampling step from ``t`` to ``t + sample_time_s``."""

    @abstractmethod
    def measure(self, outputs: Array, rng: np.random.Generator) -> Array:
        """Sensor model: noisy, delayed observation of the output variables."""

    def time_constants(self, state: Array, action: Array, disturbance: Array) -> dict[str, float]:
        """Time constants at the linearised operating point, if the model provides them.

        Optional capability instead of mandatory method: the grey-box model can
        differentiate its Jacobian numerically, a future Dyssol adapter or a
        learned surrogate possibly not. The default implementation raises, so that
        `simulator.time_constants(...)` stays type-checked under mypy strict -- without this
        declaration every caller would need a `# type: ignore[attr-defined]`, which at the
        next refactoring becomes wrong unnoticed as easily as right.

        :raises NotImplementedError: if the simulator does not support time constants.
            Callers that must handle this check beforehand with
            ``hasattr(simulator, "time_constants")`` -- that stays valid because the method
            exists on every instance, it is just sometimes not implemented.
        """
        raise NotImplementedError(f"{type(self).__name__} liefert keine Zeitkonstanten")

    def reset_sensor(self) -> None:  # noqa: B027 - deliberately not abstract
        """Discard the internal state of the sensor model.

        Necessary because the sensor model contains a first-order lag element and
        thus carries **state across the time steps**. If the same
        simulator instance is rolled out twice, the filter otherwise starts with the final state
        of the previous run: a coupling between runs that does not exist in the process
        and that would be visible as an error nowhere. The default implementation does
        nothing -- a simulator without sensor dynamics has nothing to reset.
        """

    @property
    def projection_active(self) -> bool:
        """Whether :meth:`project_action` actually clips anything for this simulator.

        Generic switch state for :meth:`rollout`: determines whether the bookkeeping
        (``requested_actions``/``n_projected_steps`` on the resulting
        :class:`Trajectory`) is kept at all. Default False -- a simulator without
        safety projection (Dyssol adapter, replay, surrogate) need not override anything.
        """
        return False

    def project_action(self, state: Array, action: Array, disturbance: Array) -> Array:
        """Material-specific safety projection, BEFORE the step, on the true state.

        Generic call point in :meth:`rollout`, directly before ``step()`` -- requirement from the
        task: margins such as fluidization limits are state-dependent and must be computed from
        the TRUE, noise-free state (never from a measurement), immediately
        before the integration step whose action they restrict. The generic
        base implementation does nothing -- simulators with such a projection (see
        :meth:`~twinlink_ml.simulation.greybox.GreyBoxFluidBedGranulator.project_action`)
        override both this method and :attr:`projection_active`.
        """
        del state, disturbance  # unused in the generic no-op base implementation
        return action

    def remeasure(self, outputs: FloatArray, *, seed: int) -> FloatArray:
        """Apply the sensor model again to already computed output variables.

        Why this is a method of its own and not a second ``rollout``: the
        process model is deterministic, the seed affects **exclusively** the
        sensor model. Varying a noise seed or a noise level therefore requires
        no new integration. For step 5 this is the difference between
        one simulation per dataset realisation and one per combination of
        dataset, noise seed and noise level.

        The loop over time is deliberately the same as in :meth:`rollout` --
        a vectorised variant would consume the random generator in a different order
        and would not be bit-identical to the rollout. A test guarantees the
        equality.
        """
        self.reset_sensor()
        backend = self.backend
        rng = np.random.default_rng(seed)
        values = np.asarray(outputs, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("outputs muss die Form (batch, time, channel) haben")
        return np.stack(
            [
                backend.to_numpy(self.measure(backend.asarray(values[:, k, :]), rng))
                for k in range(values.shape[1])
            ],
            axis=1,
        )

    def rollout(
        self,
        actions: FloatArray,
        disturbances: FloatArray,
        *,
        run_id: tuple[str, ...] | None = None,
        seed: int = 0,
        initial_state: Array | None = None,
    ) -> Trajectory:
        """Trajectory for given excitation and disturbance signals.

        Template method: the loop is fixed here, the physics in ``step``.

        :param actions: shape ``(batch, time, n_action)`` in engineering units
        :param disturbances: shape ``(batch, time, n_disturbance)``
        :param seed: seed **exclusively** for the sensor model. The process model
            itself is deterministic -- reproducibility thus depends only on
            (config, excitation, seed).
        """
        self._validate_inputs(actions, disturbances)
        n_batch, n_time, _ = actions.shape
        backend = self.backend
        rng = np.random.default_rng(seed)

        state = self.initial_state(n_batch) if initial_state is None else initial_state
        self.reset_sensor()
        states: list[FloatArray] = []
        outputs: list[FloatArray] = []
        measurements: list[FloatArray] = []
        diagnostics: dict[str, list[FloatArray]] = {}

        # Generic projection switch: bookkeeping (requested_actions/
        # n_projected_steps) is kept only if configured at all -- otherwise the
        # rollout stays as light as before and the Trajectory bit-identical to the
        # pre-projection state (see the Trajectory field docstrings and the regression test).
        projection_active = self.projection_active
        realised_actions_steps: list[FloatArray] = []
        requested_actions_steps: list[FloatArray] = []
        n_projected_steps = np.zeros(n_batch, dtype=np.int64) if projection_active else None

        for k in range(n_time):
            requested_action = backend.asarray(actions[:, k, :])
            disturbance = backend.asarray(disturbances[:, k, :])
            # `state` is here the internal, TRUE simulator state of this iteration --
            # unchanged by self.measure() further below. project_action() must therefore
            # never rely on a measurement, see its contract.
            action = self.project_action(state, requested_action, disturbance)
            action_np = backend.to_numpy(action)
            realised_actions_steps.append(action_np)
            if projection_active:
                requested_np = backend.to_numpy(requested_action)
                requested_actions_steps.append(requested_np)
                assert n_projected_steps is not None
                n_projected_steps = n_projected_steps + np.any(action_np != requested_np, axis=-1)
            result = self.step(state, action, disturbance)
            state = result.state
            states.append(backend.to_numpy(state))
            outputs.append(backend.to_numpy(result.outputs))
            measurements.append(backend.to_numpy(self.measure(result.outputs, rng)))
            for key, value in result.diagnostics.items():
                diagnostics.setdefault(key, []).append(backend.to_numpy(value))

        requested_actions = np.stack(requested_actions_steps, axis=1) if projection_active else None

        return Trajectory(
            run_id=run_id if run_id is not None else tuple(f"run_{i:05d}" for i in range(n_batch)),
            time_s=np.arange(1, n_time + 1, dtype=np.float64) * self.sample_time_s,
            # REALISED action (after project_action()) -- feeds model features and
            # mass balance and must match the integrated state. With
            # projection_active=False bit-identical to the raw `actions` argument, because
            # project_action() is then the identity for every step.
            actions=np.stack(realised_actions_steps, axis=1),
            disturbances=np.asarray(disturbances, dtype=np.float64),
            states=np.stack(states, axis=1),
            outputs=np.stack(outputs, axis=1),
            measurements=np.stack(measurements, axis=1),
            diagnostics={k: np.stack(v, axis=1) for k, v in diagnostics.items()},
            metadata=self.run_metadata(seed=seed),
            requested_actions=requested_actions,
            projection_active=projection_active,
            n_projected_steps=n_projected_steps,
        )

    def run_metadata(self, *, seed: int) -> dict[str, Any]:
        """Written into every trajectory and later into the experiment tracking."""
        return {
            "simulator": type(self).__name__,
            "backend": self.backend.name,
            "sample_time_s": self.sample_time_s,
            "seed": seed,
            "action_names": self.action_names,
            "disturbance_names": self.disturbance_names,
            "state_names": self.state_names,
            "output_names": self.output_names,
        }

    def _validate_inputs(self, actions: FloatArray, disturbances: FloatArray) -> None:
        if actions.ndim != 3 or disturbances.ndim != 3:
            raise ValueError(
                "actions und disturbances müssen die Form (batch, time, channel) haben"
            )
        if actions.shape[:2] != disturbances.shape[:2]:
            raise ValueError("actions und disturbances haben unterschiedliche Batch-/Zeitform")
        if actions.shape[2] != len(self.action_names):
            raise ValueError(
                f"{actions.shape[2]} Stellgrößen übergeben, erwartet {len(self.action_names)}: "
                f"{self.action_names}"
            )
        if disturbances.shape[2] != len(self.disturbance_names):
            raise ValueError(
                f"{disturbances.shape[2]} Störgrößen übergeben, "
                f"erwartet {len(self.disturbance_names)}: {self.disturbance_names}"
            )
