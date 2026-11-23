"""Stationary Dyssol flowsheet as a function: topology of the file, script, run, results.

A flowsheet of the family handled here has, per line, one fresh-gas feed, one suspension feed,
one heater, one dryer, and one heat recovery whose hot side is that dryer's exhaust and whose
cold side preheats the fresh gas of one line (its own or another one). The topology is read from
the ports of the ``.dflw`` file (:func:`discover_topology`); unit and stream names are taken from
the file and never written into the code.

Two sources of the same function (:class:`SteadyStateSource`):

* :class:`DyssolFlowsheet` -- runs ``DyssolC`` once per point with a generated script. The
  script keeps everything of the source file and sets, at the first and the last time point,
  mass flow, temperature, phase and compound fractions of both feeds and the unit parameters.
* :class:`ReferenceFlowsheet` -- the Python rebuild (:mod:`.reference`).
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import tempfile
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from twinlink_ml.flowsheet.validators import saturation_humidity_kg_per_kg
from twinlink_ml.partner_v4_1.comparison import DyssolFile, load_dyssol_steady_state

from .materials import COMPOUND_NAMES, DyssolProperties
from .reference import KELVIN, LineInputs, LineOutputs, evaluate

__all__ = [
    "DyssolError",
    "DyssolFlowsheet",
    "DyssolInstallation",
    "Evaluation",
    "LineUnits",
    "ReferenceFlowsheet",
    "SteadyStateSource",
    "Topology",
    "discover_topology",
    "extract_outputs",
    "nominal_inputs",
    "peak_working_set",
]

_DRYER_PORTS = {"GasIn", "Feed", "GasOut", "Product"}
_HEATER_PORTS = {"InPort", "OutPort"}
_RECOVERY_PORTS = {
    "Heat recovery inlet",
    "Preheater inlet",
    "Heat recovery outlet",
    "Preheater outlet",
}
_INLET_PORTS = {"InletMaterial"}
_FEED = "InputMaterial"
_P_EQUILIBRIUM = "Equilibrium moisture content"
_P_EFFICIENCY, _P_CALORIFIC, _P_OUTLET = "Efficiency", "Calorific Value", "Outlet Temperature"
_P_MEDIUM = "Medium mass flow"
_S_DUTY, _S_FUEL, _S_HEAT = "Heat duty [kW]", "Fuel consumption [kg/h]", "Heat recovered [kW]"
_STATES = {"solid": "SOLID", "liquid": "LIQUID", "gas": "GAS"}


class DyssolError(RuntimeError):
    """A Dyssol run reported an error or wrote no result."""


@dataclass(frozen=True)
class LineUnits:
    """Unit and stream names of one line, as in the file."""

    gas_inlet: str
    suspension: str
    heater: str
    dryer: str
    recovery: str
    """Heat recovery whose hot side is this line's dryer exhaust."""
    gas_in: str
    feed: str
    heater_in: str
    dryer_out: str
    exhaust: str
    product: str


@dataclass(frozen=True)
class Topology:
    """Lines of a flowsheet and which line's exhaust preheats which fresh gas."""

    lines: tuple[LineUnits, ...]
    preheated_by: tuple[int, ...]


def _one(candidates: list[str], what: str) -> str:
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one {what}, found {candidates}")
    return candidates[0]


def discover_topology(file: DyssolFile) -> Topology:
    """Read the lines of a flowsheet from the ports of its units."""
    stream_name = {s.key: s.name for s in file.streams.values()}
    kinds: dict[str, list[str]] = {"dryer": [], "heater": [], "recovery": [], "inlet": []}
    for name, unit in file.units.items():
        port_names = set(unit.ports)
        for kind, expected in (
            ("dryer", _DRYER_PORTS),
            ("heater", _HEATER_PORTS),
            ("recovery", _RECOVERY_PORTS),
            ("inlet", _INLET_PORTS),
        ):
            if port_names == expected:
                kinds[kind].append(name)

    def with_port(kind: str, port: str, key: str) -> str:
        found = [n for n in kinds[kind] if file.units[n].ports[port] == key]
        return _one(found, f"{kind} with port {port!r} on stream {stream_name.get(key, key)!r}")

    lines: list[LineUnits] = []
    preheater_of: list[str] = []
    for dryer in sorted(kinds["dryer"]):
        ports = file.units[dryer].ports
        heater = with_port("heater", "OutPort", ports["GasIn"])
        heater_in = file.units[heater].ports["InPort"]
        preheater = with_port("recovery", "Preheater outlet", heater_in)
        gas_in = file.units[preheater].ports["Preheater inlet"]
        recovery = with_port("recovery", "Heat recovery inlet", ports["GasOut"])
        lines.append(
            LineUnits(
                gas_inlet=with_port("inlet", "InletMaterial", gas_in),
                suspension=with_port("inlet", "InletMaterial", ports["Feed"]),
                heater=heater,
                dryer=dryer,
                recovery=recovery,
                gas_in=stream_name[gas_in],
                feed=stream_name[ports["Feed"]],
                heater_in=stream_name[heater_in],
                dryer_out=stream_name[ports["GasOut"]],
                exhaust=stream_name[file.units[recovery].ports["Heat recovery outlet"]],
                product=stream_name[ports["Product"]],
            )
        )
        preheater_of.append(preheater)
    if not lines:
        raise ValueError("no dryer of the stationary unit family in the flowsheet")
    recoveries = [ln.recovery for ln in lines]
    if sorted(recoveries) != sorted(kinds["recovery"]) or len(set(recoveries)) != len(lines):
        raise ValueError("flowsheet is not of the family 'one heat recovery per exhaust'")
    return Topology(tuple(lines), tuple(recoveries.index(p) for p in preheater_of))


def _phase(fractions: Mapping[str, float], state: str) -> str:
    return _one([n for n in fractions if n.lower().startswith(state)], f"{state} phase")


def _compound_index(file: DyssolFile, names: Mapping[str, str], role: str) -> int:
    wanted = COMPOUND_NAMES[role]
    found = [i for i, key in enumerate(file.compounds) if names.get(key) == wanted]
    return int(_one([str(i) for i in found], f"compound {wanted!r}"))


def nominal_inputs(
    file: DyssolFile, topology: Topology, names: Mapping[str, str]
) -> tuple[LineInputs, ...]:
    """Inputs stored in the file (feeds at the last time point, unit parameters)."""
    vapour = _compound_index(file, names, "vapour")
    out = []
    for ln in topology.lines:
        gas = file.streams[ln.gas_in]
        feed = file.streams[ln.feed]
        x_vapour = gas.compound_fractions[_phase(gas.phase_fractions, "gas")][vapour]
        heater = file.units[ln.heater].parameters
        out.append(
            LineInputs(
                dry_gas_kg_per_h=gas.mass_flow_kg_per_s * (1.0 - x_vapour) * 3600.0,
                gas_humidity_g_per_kg=x_vapour / (1.0 - x_vapour) * 1000.0,
                gas_celsius=gas.temperature_celsius,
                dryer_inlet_celsius=heater[_P_OUTLET][0],
                spray_kg_per_h=feed.mass_flow_kg_per_s * 3600.0,
                solid_fraction=feed.phase_fractions[_phase(feed.phase_fractions, "solid")],
                spray_celsius=feed.temperature_celsius,
                medium_kg_per_h=file.units[ln.recovery].parameters[_P_MEDIUM][0],
                equilibrium_moisture_g_per_kg=file.units[ln.dryer].parameters[_P_EQUILIBRIUM][0],
                heater_efficiency_percent=heater[_P_EFFICIENCY][0],
                calorific_value_kj_per_kg=heater[_P_CALORIFIC][0],
            )
        )
    return tuple(out)


def extract_outputs(
    file: DyssolFile, topology: Topology, names: Mapping[str, str]
) -> tuple[LineOutputs, ...]:
    """Results of every line at the last time point of a result file."""
    vapour = _compound_index(file, names, "vapour")
    out = []
    for ln in topology.lines:
        dryer_out = file.streams[ln.dryer_out]
        x_vapour = dryer_out.compound_fractions[_phase(dryer_out.phase_fractions, "gas")][vapour]
        humidity = x_vapour / (1.0 - x_vapour)
        saturation = saturation_humidity_kg_per_kg(dryer_out.temperature_celsius)
        heater = file.units[ln.heater].state
        out.append(
            LineOutputs(
                dryer_outlet_celsius=dryer_out.temperature_celsius,
                dryer_outlet_humidity_g_per_kg=humidity * 1000.0,
                exhaust_celsius=file.streams[ln.exhaust].temperature_celsius,
                preheated_celsius=file.streams[ln.heater_in].temperature_celsius,
                heat_recovered_kw=file.units[ln.recovery].state[_S_HEAT],
                heater_duty_kw=heater[_S_DUTY],
                fuel_kg_per_h=heater[_S_FUEL],
                product_kg_per_h=file.streams[ln.product].mass_flow_kg_per_s * 3600.0,
                saturation_ratio=humidity / saturation if saturation > 0.0 else float("inf"),
            )
        )
    return tuple(out)


@dataclass(frozen=True)
class Evaluation:
    """Results of one point and what the run cost."""

    outputs: tuple[LineOutputs, ...]
    wall_seconds: float
    peak_memory_bytes: int | None


class SteadyStateSource(ABC):
    """A stationary flowsheet as a function of the inputs of its lines."""

    @abstractmethod
    def evaluate(self, lines: Sequence[LineInputs], tag: str) -> Evaluation:
        """Results for one point; ``tag`` names the point in logs and scratch files."""


@dataclass(frozen=True)
class ReferenceFlowsheet(SteadyStateSource):
    """The Python rebuild with the properties of the materials database."""

    properties: DyssolProperties
    preheated_by: tuple[int, ...]

    def evaluate(self, lines: Sequence[LineInputs], tag: str = "") -> Evaluation:  # noqa: ARG002
        """Evaluate the closed formulas."""
        t0 = time.perf_counter()
        outputs = evaluate(self.properties, lines, self.preheated_by)
        return Evaluation(outputs, time.perf_counter() - t0, None)


@dataclass(frozen=True)
class DyssolInstallation:
    """Where the command-line Dyssol, the compiled units and the materials database are."""

    executable: Path
    units_dir: Path
    materials_db: Path


def peak_working_set(handle: int | None = None) -> int | None:
    """Peak working set [bytes] of a child process, or of this process (Windows only)."""
    if sys.platform != "win32":
        return None
    if handle is None:
        kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined,unused-ignore]
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        handle = int(kernel32.GetCurrentProcess())

    class Counters(ctypes.Structure):
        _fields_ = [  # noqa: RUF012
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    psapi = ctypes.WinDLL("psapi")  # type: ignore[attr-defined,unused-ignore]
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    ok = psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle), ctypes.byref(counters), counters.cb)
    return int(counters.PeakWorkingSetSize) if ok else None


class DyssolFlowsheet(SteadyStateSource):
    """One ``.dflw`` file run with ``DyssolC`` at given inputs."""

    def __init__(
        self,
        source: Path,
        installation: DyssolInstallation,
        work_dir: Path,
        names: Mapping[str, str],
    ) -> None:
        self.source = Path(source)
        self.installation = installation
        self.work_dir = Path(work_dir)
        self.names = dict(names)
        self.file = load_dyssol_steady_state(self.source)
        self.topology = discover_topology(self.file)
        self.nominal = nominal_inputs(self.file, self.topology, self.names)
        any_stream = next(iter(self.file.streams.values()))
        self.end_time = any_stream.time_s
        self._phases = tuple(any_stream.phase_fractions)
        self._index = {r: _compound_index(self.file, self.names, r) for r in COMPOUND_NAMES}
        self._local_copy: Path | None = None

    def local_source(self) -> Path:
        """The source file in a local temporary directory (copied once per instance).

        ``DyssolC`` cannot open a flowsheet from a network (UNC) path, so every job opens this
        copy instead of the delivered file. The directory is removed with the instance.
        """
        if self._local_copy is None or not self._local_copy.exists():
            directory = Path(tempfile.mkdtemp(prefix="twinlink_dyssol_src_"))
            weakref.finalize(self, shutil.rmtree, directory, True)
            copy = directory / self.source.name
            shutil.copyfile(self.source, copy)
            self._local_copy = copy
        return self._local_copy

    def _fractions(self, role_values: Mapping[str, float]) -> str:
        values = [0.0] * len(self.file.compounds)
        for role, value in role_values.items():
            values[self._index[role]] = value
        return " ".join(f"{v:.17g}" for v in values)

    def _phase_fractions(self, by_state: Mapping[str, float]) -> str:
        values = []
        for name in self._phases:
            state = _one([s for s in _STATES if name.lower().startswith(s)], f"state of {name!r}")
            values.append(by_state.get(state, 0.0))
        return " ".join(f"{v:.17g}" for v in values)

    def _feed(self, unit: str, mass: float, kelvin: float, phases: str, parts: list[str]) -> str:
        t0, t1 = 0.0, self.end_time
        overall = f'HOLDUP_OVERALL "{unit}" "{_FEED}"'
        rows = [
            f"{overall} MASS_FLOW {t0:g} {mass:.17g} {t1:g} {mass:.17g}",
            f"{overall} TEMPERATURE {t0:g} {kelvin:.17g} {t1:g} {kelvin:.17g}",
            f'HOLDUP_PHASES "{unit}" "{_FEED}" {t0:g} {phases} {t1:g} {phases}',
        ]
        rows += [f'HOLDUP_COMPOUNDS "{unit}" "{_FEED}" {p}' for p in parts]
        return "\n".join(rows)

    def script(self, lines: Sequence[LineInputs], result: Path, source: Path | None = None) -> str:
        """Script of one job: the source file with the feeds and parameters of ``lines``.

        ``source`` is the file ``DyssolC`` opens (default: the delivered file; ``evaluate``
        passes the local copy).
        """
        if len(lines) != len(self.topology.lines):
            raise ValueError("one LineInputs per line of the flowsheet")
        inst = self.installation
        t0, t1 = 0.0, self.end_time
        rows = [
            "JOB",
            f'SOURCE_FILE "{self.source if source is None else source}"',
            f'RESULT_FILE "{result}"',
            f'MATERIALS_DATABASE "{inst.materials_db}"',
            f'MODELS_PATH "{inst.units_dir}"',
        ]
        for units, ln in zip(self.topology.lines, lines, strict=True):
            humidity = ln.gas_humidity_g_per_kg / 1000.0
            gas = self._fractions(
                {"gas": 1.0 / (1.0 + humidity), "vapour": humidity / (1.0 + humidity)}
            )
            rows.append(
                self._feed(
                    units.gas_inlet,
                    ln.dry_gas_kg_per_h * (1.0 + humidity) / 3600.0,
                    ln.gas_celsius + KELVIN,
                    self._phase_fractions({"gas": 1.0}),
                    [f"GAS {t0:g} {gas} {t1:g} {gas}"],
                )
            )
            solid, liquid = self._fractions({"solid": 1.0}), self._fractions({"liquid": 1.0})
            rows.append(
                self._feed(
                    units.suspension,
                    ln.spray_kg_per_h / 3600.0,
                    ln.spray_celsius + KELVIN,
                    self._phase_fractions(
                        {"solid": ln.solid_fraction, "liquid": 1.0 - ln.solid_fraction}
                    ),
                    [
                        f"SOLID {t0:g} {solid} {t1:g} {solid}",
                        f"LIQUID {t0:g} {liquid} {t1:g} {liquid}",
                    ],
                )
            )
            for unit, parameter, value in (
                (units.heater, _P_OUTLET, ln.dryer_inlet_celsius),
                (units.heater, _P_EFFICIENCY, ln.heater_efficiency_percent),
                (units.heater, _P_CALORIFIC, ln.calorific_value_kj_per_kg),
                (units.dryer, _P_EQUILIBRIUM, ln.equilibrium_moisture_g_per_kg),
                (units.recovery, _P_MEDIUM, ln.medium_kg_per_h),
            ):
                rows.append(f'UNIT_PARAMETER "{unit}" "{parameter}" {value:.17g}')
        return "\n".join(rows) + "\n"

    def evaluate(self, lines: Sequence[LineInputs], tag: str) -> Evaluation:
        """Run Dyssol on the local copy of the source file in its own scratch directory.

        The scratch directory is removed on success.
        """
        scratch = self.work_dir / tag
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True)
        result = scratch / "result.dflw"
        script = scratch / "script.txt"
        script.write_text(self.script(lines, result, self.local_source()), encoding="utf-8")
        t0 = time.perf_counter()
        process = subprocess.Popen(
            [str(self.installation.executable), f"--script={script}"],
            cwd=scratch,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        log, _ = process.communicate()
        wall = time.perf_counter() - t0
        peak = peak_working_set(int(getattr(process, "_handle", 0)))
        (scratch / "dyssol.log").write_text(log, encoding="utf-8")
        failed = [ln for ln in log.splitlines() if "error" in ln.lower()]
        if process.returncode != 0 or failed or not result.exists():
            raise DyssolError(
                f"{tag}: return code {process.returncode}, messages {failed[:3]}, log in {scratch}"
            )
        outputs = extract_outputs(load_dyssol_steady_state(result), self.topology, self.names)
        shutil.rmtree(scratch, ignore_errors=True)
        return Evaluation(outputs, wall, peak)
