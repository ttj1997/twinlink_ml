r"""Modellregistrierung: die Architektur-Slots von Schritt 6.

Wozu eine Registrierung und nicht ein Dict im Skript
----------------------------------------------------
Der Arbeitsplan verlangt Open-Closed: eine neue Architektur hinzufügen darf keine Änderung
an Trainings- oder Evaluationscode erfordern. Ein Dict im Treiberskript erfüllt das nicht,
sobald es drei Treiber gibt -- dann steht in jedem eine eigene Auswahl, und zwei Studien
vergleichen unterschiedliche Modellmengen unter demselben Namen.

Hier steht jede Architektur **einmal**, mit ihrem Label, ihrer Fabrik und der Angabe, was
sie braucht. Treiber fragen nach Labels; welche verfügbar sind, entscheidet die Umgebung.

Warum ein Slot trotzdem eingetragen bleibt, auch als "nicht implementiert"
---------------------------------------------------------------------------
Alle vier Architektur-Kandidaten (``gru``, ``lstm``, ``xlstm``, ``transformer``,
``mamba``) sind inzwischen implementiert -- der Mechanismus für einen deklarierten, aber
noch nicht geschriebenen Slot (``implemented=False``, ``ModelSlot.missing``/
``is_available``) bleibt trotzdem Teil dieser Datei: eine künftige weitere Architektur
reiht sich hier genauso ein. Der Alternativentwurf -- einen unfertigen Slot einfach
weglassen -- verliert die Information, dass er zum Vergleich gehört; der andere
Alternativentwurf -- ihn blind implementieren -- erzeugt Code, der fertig aussieht und
es nicht ist. Ein Slot, der sagt "ich fehle noch und hier ist der Grund", ist beides
nicht.

``xlstm`` war bis Schritt 6, Teil 2 in dieser Lage und ist jetzt implementiert -- aber
**nur** der mLSTM-Anteil (Beck et al. 2024), ausdrücklich kein sLSTM (siehe ``note``
unten und :mod:`twinlink_ml.models.xlstm`). Wer über dieses Label einen vollständigen
xLSTM mit sLSTM-Zweigen erwartet, bekommt ihn nicht -- die ``note`` sagt das an
derselben Stelle, an der zuvor "nicht implementiert" stand.

``transformer`` war ebenfalls bis Schritt 6, Teil 2 nur deklariert und ist jetzt
implementiert -- vollständig, keine Teilarchitektur wie bei xLSTM (siehe ``note`` unten
und :func:`twinlink_ml.models.torch_sequence._build_transformer_network`).

``mamba`` war zuletzt in dieser Lage (deklariert, ``implemented=False``) und ist jetzt
implementiert -- als reine PyTorch-Rekurrenz, nicht über ``mamba-ssm`` (dessen
Installation auf dieser Umgebung an fehlendem ``nvcc``/CUDA-Toolkit scheitert, geprüft
in der Sitzung zu diesem Befund). Siehe ``note`` unten und
:mod:`twinlink_ml.models.mamba` für die Gleichungen und dokumentierten
Vereinfachungen.
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
    """Rolle eines Modells im Vergleich."""

    PARAMETER_FREE = "parameter_free"
    """Ohne angepasste Beziehung. Referenz für das Auflösungsvermögen."""

    BASELINE = "baseline"
    """Angepasst, aber ohne gelernten Zustand oder ohne Nichtlinearität."""

    ARCHITECTURE = "architecture"
    """Kandidat des Architekturvergleichs."""


@dataclass(frozen=True, slots=True)
class ModelSlot:
    """Ein Modell im Vergleich, samt Voraussetzungen.

    Attributes
    ----------
    label
        Kurzname, unter dem das Modell in Logs und Kurven erscheint. **Stabil halten** --
        er ist Teil des Auswertungsvertrags (``model/label`` im Run-Log).
    factory
        Nullstellige Fabrik. Nullstellig, damit je Seed und Stützstelle eine frische
        Instanz entsteht und kein Zustand über Lernkurvenpunkte hinweg lebt. ``None``
        heißt: erst bei Bedarf über ``_resolve_torch_factory`` auflösen (siehe dort) --
        so bleibt das Paket ohne Torch importierbar. Ob ein Slot mit ``factory=None``
        tatsächlich aufrufbar ist, entscheidet **``implemented``**, nicht ``factory``
        selbst: sonst wäre ein torchfrei-lazy registrierter Slot ununterscheidbar von
        einem, der schlicht noch nicht geschrieben ist.
    kind
        Rolle im Vergleich.
    requires
        Importnamen, die vorhanden sein müssen.
    needs_full_observation
        Ob das Modell eine Inline-Messung aller Zielgrößen braucht.
    implemented
        Ob der Slot überhaupt aufgerufen werden darf. ``False`` für deklarierte, aber noch
        nicht geschriebene Architekturen -- unabhängig davon, ob ihre Abhängigkeiten
        zufällig erfüllt sind. Aktuell kein Slot in dieser Lage (xLSTM, Transformer und
        Mamba waren es, sind inzwischen implementiert); der Mechanismus bleibt für eine
        künftige Architektur bestehen.
    note
        Begründung oder Status, erscheint im Bericht.
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
        """Ob der Slot implementiert ist und alle Abhängigkeiten vorhanden sind."""
        return self.implemented and all(find_spec(name) is not None for name in self.requires)

    @property
    def missing(self) -> tuple[str, ...]:
        """Fehlende Voraussetzungen."""
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
    """Torch-Fabriken erst beim Zugriff importieren, damit das Paket torchfrei bleibt.

    ``gru``/``lstm`` lösen seit Befund 26 (``docs/04_stand_ap1.md``) auf
    :class:`~twinlink_ml.models.composite.CompositeSequenceModel` auf, nicht mehr auf ein
    einzelnes ``TorchSequenceModel`` mit gemeinsamem Verlust über alle Zielgrößen -- der
    geteilte Zustand verschlechtert nachweisbar ``energy_target`` im Soft-Sensor-Betrieb.
    Das alte Einzelnetz (``torch_sequence.GruSequenceModel``/``LstmSequenceModel``) bleibt
    unverändert und direkt importierbar, nur ohne Registrierungs-Label -- unter anderem für
    den Kostenvergleich in Befund 26.
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
    """Labels, die in der aktuellen Umgebung und Betriebsart lauffähig sind."""
    return tuple(
        slot.label
        for slot in MODEL_SLOTS
        if slot.is_available and (fully_observed or not slot.needs_full_observation)
    )


def baseline_labels(*, fully_observed: bool = True) -> tuple[str, ...]:
    """Nur die Pflicht-Baselines -- alles außer den Architektur-Kandidaten."""
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
    """Fabriken zu Labels, in der Reihenfolge der Registrierung.

    Ein unbekanntes oder in dieser Betriebsart unzulässiges Label ist ein Fehler und keine
    stille Auslassung: ein Vergleich, dem ein Kandidat fehlt, weil sein Name falsch
    geschrieben war, sieht wie ein vollständiger Vergleich aus.
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
    """Übersicht für den Bericht: was läuft, was fehlt, warum."""
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
