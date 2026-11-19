r"""Steady responses (true and learned), their tabulation and the steady-state map on a table.

Family M of the study replaces the :class:`~twinlink_ml.load_following.steady.SteadyMap` inside
the unchanged classes :class:`~twinlink_ml.load_following.controllers.Schedule` and
:class:`~twinlink_ml.load_following.controllers.SetPointLaw` by a map computed from a learned
model. Three steps:

1. A :class:`SteadyResponse` maps the seven exogenous channels of the Befund 94 data (inlet
   temperature, gas velocity, spray rate, nozzle pressure, ambient temperature, relative humidity,
   solid fraction) to the steady power (kW), product moisture (kg/kg) and bed temperature (degC):

   * :class:`TruthResponse` -- the grey-box model itself (solid mass at its set point as in the
     steady optimiser of the headroom study; water mass and bed temperature by nested bisection on
     the model's own derivative). It is the control ``M-tab`` and the truth of the map error.
   * :class:`QuadraticResponse` -- the static quadratic regression of the Befund 94 pipeline.
   * :class:`GruFixedPointResponse` -- the composite GRU of the Befund 94 pipeline through the
     fixed point of its h36 map, the mapping and criterion of Befund 113
     (:mod:`twinlink_ml.surrogate_env.fixed_point`), unchanged, at any spray rate.

2. :func:`tabulate` evaluates a response on a regular grid (:class:`SteadyTable`): inlet
   temperature and gas velocity over the action box, spray rate over the band, ambient temperature
   and relative humidity over the disturbance ranges; nozzle pressure and solid fraction at the
   line values.

3. :class:`TableMap` has the interface of ``SteadyMap`` (``point``, ``curve``, ``failures``). A
   query interpolates the table linearly to the ambient state and spray rate (the moisture in its
   logarithm), refines the slice in the same way to a fine grid in (inlet temperature, gas
   velocity) and takes the cheapest point whose
   predicted moisture is at most the bound and whose predicted bed temperature is at most the
   limit; along the inlet temperature or the gas velocity the point is moved onto the moisture
   bound by linear interpolation where that lowers the power. A point is admissible only where
   every table node it is interpolated from is valid.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from twinlink_ml.headroom.symbolic import steady_particle_count
from twinlink_ml.load_following.steady import MapPoint
from twinlink_ml.simulation import GreyBoxFluidBedGranulator, ProcessConfig
from twinlink_ml.surrogate_env.fixed_point import FIXED_POINT_STATUSES, solve_fixed_points
from twinlink_ml.surrogate_env.model import EXOGENOUS_NAMES, FixedInputs, GruFixedPointModel

from .config import LearnedSetpointsConfig

__all__ = [
    "RESPONSE_NAMES",
    "GruFixedPointResponse",
    "QuadraticResponse",
    "SteadyResponse",
    "SteadyTable",
    "TableMap",
    "TruthResponse",
    "table_axes",
    "tabulate",
]

RESPONSE_NAMES: tuple[str, ...] = ("power_kw", "moisture_kg_per_kg", "bed_temperature_celsius")
_N_EXOGENOUS = len(EXOGENOUS_NAMES)
_STENCIL_WIDTH = 2  # nodes per axis of an interpolation stencil: 2 = linear
_MOISTURE_FLOOR = 1.0e-3  # kg/kg; predictions below it are far inside the admissible range


class SteadyResponse(ABC):
    """Steady power, moisture and bed temperature at constant exogenous inputs."""

    @abstractmethod
    def predict(self, exogenous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``exogenous (n, 7)`` physical -> ``values (n, 3)``, ``valid (n,)``.

        ``values`` in the order of :data:`RESPONSE_NAMES`; rows that are not valid hold ``nan``.
        """

    def status_counts(self) -> dict[str, int]:
        """Counts of what was not valid (empty for responses that cannot fail in kinds)."""
        return {}


class TruthResponse(SteadyResponse):
    """Steady state of the grey-box model at fixed commands, see the module docstring."""

    def __init__(self, process: ProcessConfig, *, iterations: int = 60) -> None:
        self._process = process
        self._sim = GreyBoxFluidBedGranulator(process)
        self._solid_mass = process.geometry.bed_mass_solid_setpoint_kg
        self._iterations = iterations
        self._counts: dict[float, float] = {}

    def _particle_count(self, solid_fraction: np.ndarray) -> np.ndarray:
        out = np.empty(len(solid_fraction))
        for value in np.unique(solid_fraction):
            key = float(value)
            if key not in self._counts:
                self._counts[key] = steady_particle_count(self._process, key)
            out[solid_fraction == value] = self._counts[key]
        return out

    def _derivative(
        self, exogenous: np.ndarray, count: np.ndarray, water: np.ndarray, temperature: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        state = np.stack(
            [np.full(len(water), self._solid_mass), water, count, temperature], axis=-1
        )
        derivative, diagnostics = self._sim.evaluate_derivative(
            state, exogenous[:, 0:4], exogenous[:, 4:7]
        )
        return np.asarray(derivative), diagnostics

    def _water_root(
        self, exogenous: np.ndarray, count: np.ndarray, temperature: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Water mass that balances the water at a bed temperature; whether it was bracketed."""
        low = np.full(len(temperature), 1e-6 * self._solid_mass)
        high = np.full(len(temperature), 50.0 * self._solid_mass)
        f_low = self._derivative(exogenous, count, low, temperature)[0][:, 1]
        f_high = self._derivative(exogenous, count, high, temperature)[0][:, 1]
        ok = (f_low > 0.0) & (f_high < 0.0)
        for _ in range(self._iterations):
            middle = 0.5 * (low + high)
            positive = self._derivative(exogenous, count, middle, temperature)[0][:, 1] > 0.0
            low = np.where(positive, middle, low)
            high = np.where(positive, high, middle)
        return 0.5 * (low + high), ok

    def predict(self, exogenous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        exogenous = np.atleast_2d(np.asarray(exogenous, dtype=np.float64))
        n = exogenous.shape[0]
        count = self._particle_count(exogenous[:, 6])

        def heat_balance(temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            water, ok = self._water_root(exogenous, count, temperature)
            return self._derivative(exogenous, count, water, temperature)[0][:, 3], water, ok

        low = np.minimum(exogenous[:, 4], exogenous[:, 0]) - 5.0
        high = exogenous[:, 0] + 5.0
        valid = (heat_balance(low)[0] > 0.0) & (heat_balance(high)[0] < 0.0)
        for _ in range(self._iterations):
            middle = 0.5 * (low + high)
            warming = heat_balance(middle)[0] > 0.0
            low = np.where(warming, middle, low)
            high = np.where(warming, high, middle)
        temperature = 0.5 * (low + high)
        water, ok = self._water_root(exogenous, count, temperature)
        valid &= ok
        _, diagnostics = self._derivative(exogenous, count, water, temperature)
        values = np.stack(
            [
                np.asarray(diagnostics["power_total_kw"]).reshape(n),
                water / self._solid_mass,
                temperature,
            ],
            axis=-1,
        )
        values[~valid] = np.nan
        return values, valid


class QuadraticResponse(SteadyResponse):
    """Static quadratic regression of the Befund 94 pipeline (``StaticRegressionModel``)."""

    def __init__(self, model: Any, input_scaler: Any, target_scaler: Any) -> None:
        self._model = model
        self._exogenous_scaler = input_scaler.select(range(_N_EXOGENOUS))
        self._target_scaler = target_scaler

    def predict(self, exogenous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from twinlink_ml.data.windows import WindowBatch

        exogenous = np.atleast_2d(np.asarray(exogenous, dtype=np.float64))
        n = exogenous.shape[0]
        batch = WindowBatch(
            history_inputs=np.zeros((n, 1, _N_EXOGENOUS + len(RESPONSE_NAMES))),
            future_exogenous=self._exogenous_scaler.transform(exogenous)[:, None, :],
            future_targets=np.zeros((n, 1, len(RESPONSE_NAMES))),
            run_index=np.zeros(n),
            window_start=np.zeros(n),
        )
        values = self._target_scaler.inverse(np.asarray(self._model.predict(batch))[:, 0, :])
        return np.asarray(values, dtype=np.float64), np.ones(n, dtype=bool)


class GruFixedPointResponse(SteadyResponse):
    """Composite GRU of the Befund 94 pipeline through the fixed point of its h36 map."""

    def __init__(self, model: Any, metadata: dict[str, Any], *, chunk_points: int) -> None:
        # the fixed inputs of the Befund 113 wrapper are not used: the horizon map is called with
        # the full exogenous vector, so the spray rate is free
        self._gru = GruFixedPointModel(model, metadata, FixedInputs(np.nan, np.nan, np.nan))
        self._mean = np.asarray(metadata["target_scaler"].mean, dtype=np.float64)
        self._scale = np.asarray(metadata["target_scaler"].scale, dtype=np.float64)
        self._chunk = chunk_points
        self._status: dict[str, int] = {}

    def status_counts(self) -> dict[str, int]:
        """Head statuses of the fixed-point search over all points predicted so far."""
        return dict(self._status)

    def predict(self, exogenous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        exogenous = np.atleast_2d(np.asarray(exogenous, dtype=np.float64))
        n = exogenous.shape[0]
        values = np.full((n, len(RESPONSE_NAMES)), np.nan)
        valid = np.zeros(n, dtype=bool)
        for start in range(0, n, self._chunk):
            rows = exogenous[start : start + self._chunk]
            n_rows = rows.shape[0]

            def residual(
                candidates: np.ndarray, rows: np.ndarray = rows, n_rows: int = n_rows
            ) -> np.ndarray:
                _, n_cand, n_heads = candidates.shape
                physical = self._mean + self._scale * candidates.reshape(-1, n_heads)
                final = self._gru.horizon_map(np.repeat(rows, n_cand, axis=0), physical)[:, -1, :]
                standardised = (final - self._mean) / self._scale
                shaped = standardised.reshape(n_rows, n_cand, n_heads)
                return np.asarray(shaped - candidates)

            result = solve_fixed_points(residual, n_rows, len(RESPONSE_NAMES), self._gru.criterion)
            ok = result.ok.all(axis=1)
            values[start : start + n_rows] = self._mean + self._scale * result.root
            valid[start : start + n_rows] = ok
            for k, target in enumerate(RESPONSE_NAMES):
                for name in FIXED_POINT_STATUSES:
                    key = f"{target}/{name}"
                    self._status[key] = self._status.get(key, 0) + int(
                        np.sum(result.status[:, k] == name)
                    )
        values[~valid] = np.nan
        return values, valid


@dataclass(frozen=True, eq=False)
class SteadyTable:
    """A steady response on a regular grid.

    ``axes``: inlet temperature (degC), gas velocity (m/s), spray rate (kg/h), ambient temperature
    (degC), relative humidity (-). ``values`` has the shape of the grid plus ``(3,)``
    (:data:`RESPONSE_NAMES`), with 0 where ``valid`` is false.
    """

    axes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    values: np.ndarray
    valid: np.ndarray
    nozzle_pressure_bar: float
    solid_fraction: float
    source: str

    @property
    def digest(self) -> str:
        """Hash of grid, values and validity."""
        h = hashlib.sha256()
        for axis in self.axes:
            h.update(np.ascontiguousarray(axis, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.values, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(self.valid, dtype=np.bool_).tobytes())
        h.update(repr((self.nozzle_pressure_bar, self.solid_fraction)).encode("utf-8"))
        return h.hexdigest()[:16]

    def save(self, path: Path) -> None:
        """Write the table as ``.npz`` (never over an existing file)."""
        if path.exists():
            raise FileExistsError(f"{path} exists -- tables are never overwritten")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            axis_0=self.axes[0],
            axis_1=self.axes[1],
            axis_2=self.axes[2],
            axis_3=self.axes[3],
            axis_4=self.axes[4],
            values=self.values,
            valid=self.valid,
            fixed=np.array([self.nozzle_pressure_bar, self.solid_fraction]),
            source=np.array(self.source),
        )

    @classmethod
    def load(cls, path: Path) -> SteadyTable:
        """Read a table written by :meth:`save`."""
        with np.load(path) as data:
            a0, a1, a2, a3, a4 = (np.array(data[f"axis_{i}"]) for i in range(5))
            return cls(
                axes=(a0, a1, a2, a3, a4),
                values=np.array(data["values"]),
                valid=np.array(data["valid"]),
                nozzle_pressure_bar=float(data["fixed"][0]),
                solid_fraction=float(data["fixed"][1]),
                source=str(data["source"]),
            )


def table_axes(
    config: LearnedSetpointsConfig, band: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid axes of the study: action box, band of the spray rate, disturbance ranges."""
    process, table = config.load_following.process, config.table
    inlet, gas = process.actions.inlet_temperature_celsius, process.actions.gas_velocity_m_per_s
    ambient = process.disturbances.ambient_temperature_celsius
    humidity = process.disturbances.ambient_relative_humidity
    return (
        np.linspace(inlet.minimum, inlet.maximum, table.inlet_temperature_points),
        np.linspace(gas.minimum, gas.maximum, table.gas_velocity_points),
        np.linspace(band[0], band[1], table.spray_rate_points),
        np.linspace(ambient.minimum, ambient.maximum, table.ambient_temperature_points),
        np.linspace(humidity.minimum, humidity.maximum, table.relative_humidity_points),
    )


def tabulate(
    response: SteadyResponse,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    nozzle_pressure_bar: float,
    solid_fraction: float,
    source: str,
) -> SteadyTable:
    """Evaluate ``response`` on the grid ``axes``."""
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    shape = grid.shape[:-1]
    flat = grid.reshape(-1, 5)
    exogenous = np.empty((flat.shape[0], _N_EXOGENOUS))
    exogenous[:, 0:3] = flat[:, 0:3]
    exogenous[:, 3] = nozzle_pressure_bar
    exogenous[:, 4:6] = flat[:, 3:5]
    exogenous[:, 6] = solid_fraction
    values, valid = response.predict(exogenous)
    return SteadyTable(
        axes=axes,
        values=np.where(valid[:, None], values, 0.0).reshape(*shape, len(RESPONSE_NAMES)),
        valid=valid.reshape(shape),
        nozzle_pressure_bar=float(nozzle_pressure_bar),
        solid_fraction=float(solid_fraction),
        source=source,
    )


def _stencil(axis: np.ndarray, value: float) -> tuple[int, np.ndarray, bool]:
    """Interpolation stencil on a uniform axis: first node, Lagrange weights, clipped or not.

    ``_STENCIL_WIDTH`` nodes (2 = linear; a wider stencil oscillates where the steady moisture
    rises steeply, scratch check before the runs); the value is clipped to the axis. The weights
    sum to one and reproduce the node values at the nodes.
    """
    clipped = bool(value < axis[0] - 1e-9 or value > axis[-1] + 1e-9)
    value = float(np.clip(value, axis[0], axis[-1]))
    width = _STENCIL_WIDTH if len(axis) >= _STENCIL_WIDTH else 2
    step = axis[1] - axis[0]
    left = int(np.clip(np.floor((value - axis[0]) / step), 0, len(axis) - 2))
    first = int(np.clip(left - (width // 2 - 1), 0, len(axis) - width))
    nodes = axis[first : first + width]
    weights = np.ones(width)
    for a in range(width):
        for b in range(width):
            if a != b:
                weights[a] *= (value - nodes[b]) / (nodes[a] - nodes[b])
    return first, weights, clipped


def _refinement_matrix(axis: np.ndarray, fine: np.ndarray) -> np.ndarray:
    """Matrix of the interpolation from ``axis`` to ``fine`` (stencils of :func:`_stencil`)."""
    matrix = np.zeros((len(fine), len(axis)))
    for row, value in enumerate(fine):
        first, weights, _ = _stencil(axis, float(value))
        matrix[row, first : first + len(weights)] = weights
    return matrix


def _support_matrix(axis: np.ndarray, fine: np.ndarray) -> np.ndarray:
    """0/1 matrix: which nodes of ``axis`` a fine point is interpolated from."""
    return np.asarray(_refinement_matrix(axis, fine) != 0.0, dtype=np.float64)


class TableMap:
    """Steady-state map on a :class:`SteadyTable` with the interface of ``SteadyMap``."""

    def __init__(self, table: SteadyTable, config: LearnedSetpointsConfig) -> None:
        self.table = table
        settings = config.table
        inlet, gas = table.axes[0], table.axes[1]
        n_inlet = round((inlet[-1] - inlet[0]) / settings.fine_inlet_temperature_step_k) + 1
        n_gas = round((gas[-1] - gas[0]) / settings.fine_gas_velocity_step_m_per_s) + 1
        self._inlet = np.linspace(inlet[0], inlet[-1], n_inlet)
        self._gas = np.linspace(gas[0], gas[-1], n_gas)
        self._refine_inlet = _refinement_matrix(inlet, self._inlet)
        self._refine_gas = _refinement_matrix(gas, self._gas).T
        self._support_inlet = _support_matrix(inlet, self._inlet)
        self._support_gas = _support_matrix(gas, self._gas).T
        self._bed_limit = config.load_following.headroom.line.bed_temperature_limit_celsius
        # the moisture is interpolated in its logarithm: towards low inlet temperatures the steady
        # moisture rises by orders of magnitude, which a polynomial stencil cannot follow
        self._values = table.values.copy()
        self._values[..., 1] = np.log(np.maximum(self._values[..., 1], _MOISTURE_FLOOR))
        self._valid = table.valid.astype(np.float64)
        self.failures = 0
        self.clipped = 0
        self.queries = 0

    def _slice(
        self, spray: float, ambient: float, humidity: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Table at (spray, ambient, humidity): values ``(nT, nu, 3)``, invalid nodes used."""
        axes = self.table.axes
        s, ws, cs = _stencil(axes[2], spray)
        a, wa, ca = _stencil(axes[3], ambient)
        r, wr, cr = _stencil(axes[4], humidity)
        self.clipped += int(cs or ca or cr)
        block = self._values[:, :, s : s + len(ws), a : a + len(wa), r : r + len(wr), :]
        values = np.einsum("ijsarc,s,a,r->ijc", block, ws, wa, wr)
        used = np.einsum("s,a,r->sar", ws != 0.0, wa != 0.0, wr != 0.0)
        invalid = np.einsum(
            "ijsar,sar->ij",
            1.0 - self._valid[:, :, s : s + len(ws), a : a + len(wa), r : r + len(wr)],
            used.astype(np.float64),
        )
        return values, invalid

    def response(
        self, inlet: float, gas: float, spray: float, ambient: float, humidity: float
    ) -> tuple[np.ndarray, bool]:
        """Predicted steady response ``(3,)`` at one operating point and whether it is valid."""
        values, invalid = self._slice(spray, ambient, humidity)
        i, wi, _ = _stencil(self.table.axes[0], inlet)
        j, wj, _ = _stencil(self.table.axes[1], gas)
        out = np.einsum("ijc,i,j->c", values[i : i + len(wi), j : j + len(wj)], wi, wj)
        out[1] = np.exp(out[1])
        used = np.outer(wi != 0.0, wj != 0.0)
        ok = bool(np.sum(invalid[i : i + len(wi), j : j + len(wj)] * used) == 0.0)
        return out, ok

    def point(self, disturbance: np.ndarray, spray: float, bound: float) -> MapPoint | None:
        """Cheapest admissible point at ``disturbance`` and ``spray``; ``None`` if there is none."""
        self.queries += 1
        values, invalid = self._slice(float(spray), float(disturbance[0]), float(disturbance[1]))
        fine = np.matmul(
            np.matmul(self._refine_inlet, np.moveaxis(values, -1, 0)), self._refine_gas
        )
        fine_valid = (self._support_inlet @ invalid @ self._support_gas) == 0.0
        power, moisture, bed = fine[0], np.exp(fine[1]), fine[2]
        feasible = fine_valid & (moisture <= bound) & (bed <= self._bed_limit)
        if not feasible.any():
            self.failures += 1
            return None
        i, j = np.unravel_index(int(np.argmin(np.where(feasible, power, np.inf))), power.shape)
        best = (float(power[i, j]), float(self._inlet[i]), float(self._gas[j]), float(bed[i, j]))
        # A neighbour that is cheaper and fails only on the moisture: move onto the bound along
        # that axis (linear between the two fine points); the cheaper of the two directions.
        for di, dj in ((1, 0), (0, 1)):
            a, b = i - di, j - dj
            if a < 0 or b < 0 or not fine_valid[a, b] or bed[a, b] > self._bed_limit:
                continue
            step = moisture[a, b] - moisture[i, j]
            if power[a, b] < power[i, j] and moisture[a, b] > bound and step > 0.0:
                share = (moisture[a, b] - bound) / step
                candidate = (
                    float(power[a, b] + share * (power[i, j] - power[a, b])),
                    float(self._inlet[a] + share * (self._inlet[i] - self._inlet[a])),
                    float(self._gas[b] + share * (self._gas[j] - self._gas[b])),
                    float(bed[a, b] + share * (bed[i, j] - bed[a, b])),
                )
                best = min(best, candidate)
        return MapPoint(
            command=np.array([best[1], best[2]]),
            state=np.array([np.nan, np.nan, best[3]]),
            total_kw=best[0],
            heater_kw=float("nan"),
        )

    def curve(
        self, disturbance: np.ndarray, levels: np.ndarray, bound: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Total power (kW) per spray level, ``nan`` where infeasible; no heater share."""
        total = np.full(len(levels), np.nan)
        for j, level in enumerate(levels):
            point = self.point(disturbance, float(level), bound)
            if point is not None:
                total[j] = point.total_kw
        return total, np.full(len(levels), np.nan)
