r"""Response tables of one fluidised-bed line at one ambient point (stage M3', scheme C).

The steady state of a line depends only on its own set points and the ambient state, not on the
other line (docs/09 1.1). For the nested search over all allocations (``joint_search``) the line
solves are therefore tabulated once per scenario on a grid of the inlet temperature and the gas
velocity (nozzle pressure fixed at its lower bound, D6) and interpolated bilinearly. Every table
value
that matters for a result is re-evaluated with true solves (the polishing of ``joint_search``);
the table
only steers the search.

Tabulated per node: the bed temperature, the outlet humidity, the dry-air flow, the electric
power of
the line (fan, pump, atomiser), the particle moisture and the minimum fluidisation and terminal
velocity. The constraint violation of the objective is rebuilt from them with the formula of
``operating_point.check_constraints`` (the solver-validity term is the moisture above the critical
moisture). A node whose solve failed is NaN; every cell that touches a NaN node is NaN, and the
objective gives the failure value there (never a sentinel inside a table or an interpolation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from twinlink_ml.simulation.config import ProcessConfig
from twinlink_ml.simulation.operating_point import BED_TEMPERATURE_LIMIT_C
from twinlink_ml.simulation.oracle_search import evaluate_point

from .site_network import LineState, line_state

__all__ = ["FIELDS", "LineTable", "build_line_table"]

FIELDS = (
    "moisture",
    "bed_temperature",
    "outlet_humidity",
    "dry_air_flow",
    "electric_kw",
    "u_mf",
    "u_terminal",
)
_IDX = {name: i for i, name in enumerate(FIELDS)}
_SOLVER_VALIDITY_TOLERANCE = 1.0e-9


@dataclass(frozen=True, slots=True)
class LineTable:
    """Bilinear table of one line at one ambient point.

    ``values`` has the shape ``(n_t, n_u, n_fields)``.
    """

    name: str
    ambient_temperature_celsius: float
    ambient_relative_humidity: float
    inlet_humidity_kg_per_kg: float
    duct_loss_k: float
    critical_moisture_kg_per_kg: float
    t_in_grid: np.ndarray
    u_grid: np.ndarray
    values: np.ndarray

    def interpolate(self, t_in: float, u_gas: float) -> np.ndarray:
        """Bilinear interpolation of all fields; NaN if the cell touches a failed node."""
        tg, ug = self.t_in_grid, self.u_grid
        i = min(max(int(np.searchsorted(tg, t_in, side="right")) - 1, 0), tg.size - 2)
        j = min(max(int(np.searchsorted(ug, u_gas, side="right")) - 1, 0), ug.size - 2)
        wt = (t_in - tg[i]) / (tg[i + 1] - tg[i])
        wu = (u_gas - ug[j]) / (ug[j + 1] - ug[j])
        v = self.values
        return (
            (1.0 - wt) * (1.0 - wu) * v[i, j]
            + (1.0 - wt) * wu * v[i, j + 1]
            + wt * (1.0 - wu) * v[i + 1, j]
            + wt * wu * v[i + 1, j + 1]
        )

    def state(
        self, t_in: float, u_gas: float, threshold_kg_per_kg: float
    ) -> tuple[LineState, float] | None:
        """``(line state, normalised constraint violation)`` or ``None`` on a failed cell."""
        f = self.interpolate(t_in, u_gas)
        if np.isnan(f).any():
            return None
        moisture = float(f[_IDX["moisture"]])
        bed = float(f[_IDX["bed_temperature"]])
        u_mf = float(f[_IDX["u_mf"]])
        u_t = float(f[_IDX["u_terminal"]])
        window = max(u_t - u_mf, 1e-3)
        violation = (
            max(0.0, moisture - threshold_kg_per_kg) / max(threshold_kg_per_kg, 1e-6)
            + max(0.0, u_mf - u_gas) / window
            + max(0.0, u_gas - u_t) / window
            + max(0.0, bed - BED_TEMPERATURE_LIMIT_C) / 10.0
            + (
                0.0
                if moisture <= self.critical_moisture_kg_per_kg + _SOLVER_VALIDITY_TOLERANCE
                else 1.0
            )
        )
        state = LineState(
            name=self.name,
            ambient_temperature_celsius=self.ambient_temperature_celsius,
            inlet_temperature_celsius=t_in,
            inlet_humidity_kg_per_kg=self.inlet_humidity_kg_per_kg,
            outlet_temperature_celsius=bed - self.duct_loss_k,
            outlet_humidity_kg_per_kg=float(f[_IDX["outlet_humidity"]]),
            dry_air_mass_flow_kg_per_s=float(f[_IDX["dry_air_flow"]]),
            electric_kw=float(f[_IDX["electric_kw"]]),
        )
        return state, float(violation)

    def to_dict(self) -> dict[str, Any]:
        def clean(array: np.ndarray) -> list[Any]:
            return [None if math.isnan(x) else float(x) for x in array.ravel()]

        return {
            "name": self.name,
            "ambient_temperature_celsius": self.ambient_temperature_celsius,
            "ambient_relative_humidity": self.ambient_relative_humidity,
            "inlet_humidity_kg_per_kg": self.inlet_humidity_kg_per_kg,
            "duct_loss_k": self.duct_loss_k,
            "critical_moisture_kg_per_kg": self.critical_moisture_kg_per_kg,
            "t_in_grid": [float(x) for x in self.t_in_grid],
            "u_grid": [float(x) for x in self.u_grid],
            "fields": list(FIELDS),
            "shape": list(self.values.shape),
            "values": clean(self.values),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineTable:
        values = np.array(
            [math.nan if x is None else x for x in data["values"]], dtype=np.float64
        ).reshape(data["shape"])
        if tuple(data["fields"]) != FIELDS:
            raise ValueError("table fields differ from this code version")
        return cls(
            name=data["name"],
            ambient_temperature_celsius=data["ambient_temperature_celsius"],
            ambient_relative_humidity=data["ambient_relative_humidity"],
            inlet_humidity_kg_per_kg=data["inlet_humidity_kg_per_kg"],
            duct_loss_k=data["duct_loss_k"],
            critical_moisture_kg_per_kg=data["critical_moisture_kg_per_kg"],
            t_in_grid=np.array(data["t_in_grid"]),
            u_grid=np.array(data["u_grid"]),
            values=values,
        )


def build_line_table(
    name: str,
    config: ProcessConfig,
    *,
    t_amb: float,
    phi_amb: float,
    t_in_grid: np.ndarray,
    u_grid: np.ndarray,
    p_nozzle: float,
    max_iterations: int = 30,
    tolerance: float = 1.0e-7,
) -> LineTable:
    """Solve every node (search precision of Befund 52); a failed solve is NaN."""
    values = np.full((t_in_grid.size, u_grid.size, len(FIELDS)), np.nan)
    inlet_humidity = math.nan
    for i, t_in in enumerate(t_in_grid):
        for j, u_gas in enumerate(u_grid):
            try:
                solution = evaluate_point(
                    float(t_in),
                    float(u_gas),
                    p_nozzle,
                    t_amb,
                    phi_amb,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    base_config=config,
                )
            except (RuntimeError, ValueError):
                continue
            state = line_state(
                name, solution, config, t_amb=t_amb, phi_amb=phi_amb, t_in=float(t_in)
            )
            inlet_humidity = state.inlet_humidity_kg_per_kg
            d = solution.diagnostics
            values[i, j] = [
                solution.particle_moisture_kg_per_kg,
                solution.bed_temperature_celsius,
                solution.outlet_humidity_kg_per_kg,
                solution.dry_air_mass_flow_kg_per_s,
                state.electric_kw,
                d["minimum_fluidization_velocity_m_per_s"],
                d["terminal_velocity_m_per_s"],
            ]
    if math.isnan(inlet_humidity):
        raise RuntimeError("every node of the table failed")
    return LineTable(
        name=name,
        ambient_temperature_celsius=t_amb,
        ambient_relative_humidity=phi_amb,
        inlet_humidity_kg_per_kg=inlet_humidity,
        duct_loss_k=config.exhaust.outlet_duct_heat_loss_k,
        critical_moisture_kg_per_kg=config.material.critical_moisture_kg_per_kg,
        t_in_grid=np.asarray(t_in_grid, dtype=np.float64),
        u_grid=np.asarray(u_grid, dtype=np.float64),
        values=values,
    )
