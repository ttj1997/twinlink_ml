r"""Model registry: the architecture slots of step 6.

Why a registry and not a dict in the script
-------------------------------------------
The work plan demands open-closed: adding a new architecture must not require any change
to training or evaluation code. A dict in the driver script does not fulfil this
as soon as there are three drivers -- then each contains its own selection, and two studies
compare different model sets under the same name.

Here each architecture appears **once**, with its label, its factory and the statement of what
it needs. Drivers ask for labels; which ones are available is decided by the environment.

Why a slot still stays registered, even as "not implemented"
------------------------------------------------------------
All four architecture candidates (``gru``, ``lstm``, ``xlstm``, ``transformer``,
``mamba``) are implemented by now -- the mechanism for a declared but
not yet written slot (``implemented=False``, ``ModelSlot.missing``/
``is_available``) nevertheless remains part of this file: a future further architecture
joins here in just the same way. The alternative design -- simply leaving out an
unfinished slot -- loses the information that it belongs to the comparison; the other
alternative design -- implementing it blindly -- produces code that looks finished and
is not. A slot that says "I am still missing and here is the reason" is neither.

``xlstm`` was in this position until step 6, part 2 and is now implemented -- but
**only** the mLSTM part (Beck et al. 2024), explicitly no sLSTM (see ``note``
below and :mod:`twinlink_ml.models.xlstm`). Whoever expects a complete
xLSTM with sLSTM branches via this label does not get it -- the ``note`` says so in
the same place where "nicht implementiert" previously stood.

``transformer`` was likewise only declared until step 6, part 2 and is now
implemented -- completely, not a partial architecture as with xLSTM (see ``note`` below
and :func:`twinlink_ml.models.torch_sequence._build_transformer_network`).

``mamba`` was last in this position (declared, ``implemented=False``) and is now
implemented -- as a pure PyTorch recurrence, not via ``mamba-ssm`` (whose
installation fails on this environment due to a missing ``nvcc``/CUDA toolkit, checked
in the session on this finding). See ``note`` below and
:mod:`twinlink_ml.models.mamba` for the equations and documented
simplifications.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from importlib.util import find_spec

from .baselines import (
    ClimatologyModel,
    LinearARXModel,
    PersistenceModel,
    StaticRegressionModel,
)
from .narx import RandomFeatureNARX

__all__ = [
    "MODEL_SLOTS",
    "ModelKind",
    "ModelSlot",
    "available_labels",
    "baseline_labels",
    "build_factories",
    "describe_slots",
]


class ModelKind(str, Enum):
    """Role of a model in the comparison."""

    PARAMETER_FREE = "parameter_free"
    """Ohne angepasste Beziehung. Referenz für das Auflösungsvermögen."""

    BASELINE = "baseline"
    """Angepasst, aber ohne gelernten Zustand oder ohne Nichtlinearität."""

    ARCHITECTURE = "architecture"
    """Kandidat des Architekturvergleichs."""


@dataclass(frozen=True, slots=True)
class ModelSlot:
    """One model in the comparison, together with its prerequisites.

    Attributes
    ----------
    label
        Short name under which the model appears in logs and curves. **Keep stable** --
        it is part of the evaluation contract (``model/label`` in the run log).
    factory
        Zero-argument factory. Zero-argument, so that a fresh instance is created per seed and
        support point and no state lives across learning-curve points. ``None``
        means: resolve only when needed via ``_resolve_torch_factory`` (see there) --
        this keeps the package importable without Torch. Whether a slot with ``factory=None``
        is actually callable is decided by **``implemented``**, not by ``factory``
        itself: otherwise a slot registered torch-free and lazily would be indistinguishable
        from one that simply has not been written yet.
    kind
        Role in the comparison.
    requires
        Import names that must be present.
    needs_full_observation
        Whether the model needs an inline measurement of all targets.
    implemented
        Whether the slot may be called at all. ``False`` for declared but not yet
        written architectures -- regardless of whether their dependencies
        happen to be satisfied. Currently no slot is in this position (xLSTM, Transformer and
        Mamba were, and are implemented by now); the mechanism remains for a
        future architecture.
    note
        Justification or status, appears in the report.
    """

    label: str
    factory: object
    kind: ModelKind
    requires: tuple[str, ...] = ()
    needs_full_observation: bool = False
    implemented: bool = True
    note: str = ""

    @property
    def is_available(self) -> bool:
        """Whether the slot is implemented and all dependencies are present."""
        return self.implemented and all(find_spec(name) is not None for name in self.requires)

    @property
    def missing(self) -> tuple[str, ...]:
        """Missing prerequisites."""
        return tuple(name for name in self.requires if find_spec(name) is None)


MODEL_SLOTS: tuple[ModelSlot, ...] = (
    ModelSlot(
        label="persistence",
        factory=PersistenceModel,
        kind=ModelKind.PARAMETER_FREE,
        needs_full_observation=True,
        note="Auflösungsreferenz im inline gemessenen Fall",
    ),
    ModelSlot(
        label="climatology",
        factory=ClimatologyModel,
        kind=ModelKind.PARAMETER_FREE,
        note="Auflösungsreferenz im Soft-Sensor-Fall",
    ),
    ModelSlot(
        label="static",
        factory=StaticRegressionModel,
        kind=ModelKind.BASELINE,
        note="quadratisch, horizontunabhängig; die Hürde für energy_target",
    ),
    ModelSlot(
        label="arx",
        factory=LinearARXModel,
        kind=ModelKind.BASELINE,
        note="linear, mit Dynamik; die Referenz für Nichtlinearität",
    ),
    ModelSlot(
        label="narx",
        factory=RandomFeatureNARX,
        kind=ModelKind.BASELINE,
        note="nichtlinear, ohne gelernten Zustand; die Referenz für Sequenzmodelle",
    ),
    ModelSlot(
        label="gru",
        factory=None,
        kind=ModelKind.ARCHITECTURE,
        requires=("torch",),
        note="Verbundmodell seit Befund 26 (ein Netz je Zielgröße, negativer Transfer "
        "sonst); Bestandsprüfungen grün, Lernkurve mit mehreren Seeds steht aus",
    ),
    ModelSlot(
        label="lstm",
        factory=None,
        kind=ModelKind.ARCHITECTURE,
        requires=("torch",),
        note="Verbundmodell seit Befund 26 (ein Netz je Zielgröße, negativer Transfer "
        "sonst); Bestandsprüfungen grün, Lernkurve mit mehreren Seeds steht aus",
    ),
    ModelSlot(
        label="xlstm",
        factory=None,
        kind=ModelKind.ARCHITECTURE,
        requires=("torch",),
        note="Verbundmodell wie gru/lstm (ein Netz je Zielgröße, Befund 26). NUR der "
        "mLSTM-Anteil nach Beck et al. (2024) implementiert, KEIN sLSTM -- "
        "twinlink_ml.models.xlstm für die Gleichungen und dokumentierten "
        "Vereinfachungen. Lernkurve mit mehreren Seeds steht aus.",
    ),
    ModelSlot(
        label="mamba",
        factory=None,
        kind=ModelKind.ARCHITECTURE,
        requires=("torch",),
        note="Verbundmodell wie gru/lstm/xlstm/transformer (ein Netz je Zielgröße, "
        "Befund 26). Selektiver State-Space-Kern (Gu & Dao 2023), reine PyTorch-"
        "Rekurrenz -- mamba-ssm scheitert auf dieser Umgebung an fehlendem nvcc/CUDA "
        "(geprüft, nicht nur vermutet). Siehe twinlink_ml.models.mamba für die "
        "Gleichungen und dokumentierten Vereinfachungen. Anders als beim Transformer "
        "kein Sonderzweig nötig: die SSM-Zelle hat wie GRU/LSTM/xLSTM einen "
        "fortgeschriebenen Zustand fester Größe (O(1) pro Schritt), passt also über "
        "den cell_type-Dict-Eintrag. Lernkurve mit mehreren Seeds steht aus.",
    ),
    ModelSlot(
        label="transformer",
        factory=None,
        kind=ModelKind.ARCHITECTURE,
        requires=("torch",),
        note="Verbundmodell wie gru/lstm/xlstm (ein Netz je Zielgröße, Befund 26). "
        "Encoder ungemaskiert über die Historie, Decoder sequenziell/autoregressiv über "
        "den Horizont -- keine explizite kausale Maske nötig, weil zu jedem Schritt keine "
        "'zukünftigen' Decoder-Token existieren (siehe "
        "torch_sequence._build_transformer_network). Kleiner Standard-hidden_size=16 "
        "statt 64 -- Kapazitäts-Matching gegen GRU nicht blind, siehe "
        "CompositeTransformerSequenceModel-Docstring. Lernkurve mit mehreren Seeds "
        "steht aus.",
    ),
)


def _resolve_torch_factory(label: str) -> object:
    """Import the Torch factories only on access, so that the package stays torch-free.

    Since Befund 26 (``docs/04_stand_ap1.md``), ``gru``/``lstm`` resolve to
    :class:`~twinlink_ml.models.composite.CompositeSequenceModel`, no longer to a
    single ``TorchSequenceModel`` with a shared loss over all targets -- the
    shared state demonstrably degrades ``energy_target`` in soft sensor operation.
    The old single network (``torch_sequence.GruSequenceModel``/``LstmSequenceModel``) remains
    unchanged and directly importable, only without a registry label -- among other things for
    the cost comparison in Befund 26.
    """
    from .composite import (
        CompositeGruSequenceModel,
        CompositeLstmSequenceModel,
        CompositeMambaSequenceModel,
        CompositeTransformerSequenceModel,
        CompositeXLSTMSequenceModel,
    )

    return {
        "gru": CompositeGruSequenceModel,
        "lstm": CompositeLstmSequenceModel,
        "xlstm": CompositeXLSTMSequenceModel,
        "transformer": CompositeTransformerSequenceModel,
        "mamba": CompositeMambaSequenceModel,
    }[label]


def _slot_factory(slot: ModelSlot) -> object:
    if slot.factory is not None:
        return slot.factory
    return _resolve_torch_factory(slot.label)


def available_labels(*, fully_observed: bool = True) -> tuple[str, ...]:
    """Labels that can run in the current environment and operating mode."""
    return tuple(
        slot.label
        for slot in MODEL_SLOTS
        if slot.is_available and (fully_observed or not slot.needs_full_observation)
    )


def baseline_labels(*, fully_observed: bool = True) -> tuple[str, ...]:
    """Only the mandatory baselines -- everything except the architecture candidates."""
    return tuple(
        slot.label
        for slot in MODEL_SLOTS
        if slot.kind is not ModelKind.ARCHITECTURE
        and slot.is_available
        and (fully_observed or not slot.needs_full_observation)
    )


def build_factories(
    labels: tuple[str, ...] | None = None, *, fully_observed: bool = True
) -> Mapping[str, object]:
    """Factories for labels, in the order of registration.

    A label that is unknown or not permitted in this operating mode is an error and not a
    silent omission: a comparison that lacks a candidate because its name was
    misspelt looks like a complete comparison.
    """
    by_label = {slot.label: slot for slot in MODEL_SLOTS}
    wanted = labels if labels is not None else available_labels(fully_observed=fully_observed)
    factories: dict[str, object] = {}
    for label in wanted:
        slot = by_label.get(label)
        if slot is None:
            raise KeyError(f"unbekanntes Modell-Label {label!r}, bekannt: {sorted(by_label)}")
        if not slot.implemented:
            raise NotImplementedError(
                f"Architektur {label!r} ist noch nicht implementiert: {slot.note}"
            )
        if slot.missing:
            raise ImportError(
                f"Modell {label!r} braucht {list(slot.missing)}: pip install 'twinlink-ml[torch]'"
            )
        if not fully_observed and slot.needs_full_observation:
            raise ValueError(
                f"Modell {label!r} braucht eine Inline-Messung aller Zielgrößen "
                "und ist im Soft-Sensor-Betrieb nicht definiert"
            )
        factories[label] = _slot_factory(slot)
    return factories


def describe_slots() -> str:
    """Overview for the report: what runs, what is missing, why."""
    lines = [f"{'Label':<14}{'Rolle':<16}{'Status':<28}Anmerkung"]
    for slot in MODEL_SLOTS:
        if not slot.implemented:
            status = "nicht implementiert"
        elif slot.is_available:
            status = "verfügbar"
        else:
            status = f"fehlt: {', '.join(slot.missing) or '—'}"
        lines.append(f"{slot.label:<14}{slot.kind.value:<16}{status:<28}{slot.note}")
    return "\n".join(lines)
