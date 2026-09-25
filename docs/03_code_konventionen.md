# Code-Konventionen

## Sprache und Werkzeuge

- **Python** ≥ 3.11
- **PyTorch**, gerne PyTorch Lightning für Trainingsschleifen
- **Hydra** oder OmegaConf für Konfiguration
- **MLflow** oder Weights & Biases für Experiment-Tracking (ab Tag 1, nicht nachträglich)
- **Ruff** (Linting + Formatierung), **mypy** (strict), **pytest**
- Paketverwaltung: `uv` oder Poetry

## Struktur

Installierbares Paket mit `src`-Layout:

    twinlink_ml/
    ├── pyproject.toml
    ├── README.md
    ├── conf/                     # Hydra-Konfigurationen, versioniert
    ├── src/twinlink_ml/
    │   ├── simulation/           # Grey-Box-Prozessmodell, Anregungsdesign
    │   ├── data/                 # Datasets, Fensterung, Splits
    │   ├── models/               # Architekturen hinter gemeinsamem Interface
    │   ├── evaluation/           # Metriken, Rollouts, Lernkurven
    │   ├── constraints/          # physikalische Constraints (Projektion, Monotonie)
    │   └── utils/                # Seeding, Logging, Serialisierung
    ├── notebooks/                # importieren die Module, duplizieren keinen Code
    └── tests/

Strikte Trennung: **Config** (`conf/`) – **Code** (`src/`) – **Artefakte** (außerhalb der
Versionskontrolle, über Tracking-Tool referenziert).

## Objektorientierung

- **Type Hints durchgängig**, auch bei internen Funktionen. mypy im strict-Modus muss durchlaufen.
- **Dataclasses** (`frozen=True` wo möglich) oder **Pydantic** für alle Konfigurationsobjekte.
  Keine losen Dicts als Parametercontainer.
- **Registrierung statt Auswahl im Skript.** Modelle stehen genau einmal in
  `models/registry.py`, mit Label, Fabrik und Voraussetzungen. Ein Dict im Treiberskript
  reicht nur bis zum zweiten Treiber; danach vergleichen zwei Studien unterschiedliche
  Modellmengen unter demselben Namen. Fabriken sind nullstellig, damit je Seed und
  Stützstelle eine frische Instanz entsteht und kein Zustand über Lernkurvenpunkte lebt.
- **Abstrakte Basisklassen (ABC)** für die vier Kernabstraktionen:
  - `ProcessSimulator` – liefert Trajektorien (Grey-Box-Modell, später Dyssol-Adapter,
    später ggf. gelerntes Surrogat)
  - `SequenceDataset` – Fensterung, Normalisierung, Splits
  - `SequenceModel` – gemeinsames Interface für GRU/LSTM/xLSTM/Mamba/Transformer,
    inkl. Mehrschritt-Rollout
  - `Evaluator` – Metriken und Ergebnisformate
- **Dependency Injection** statt harter Kopplung: Modelle bekommen den Simulator bzw. das Dataset
  übergeben, konstruieren es nicht selbst. Ermöglicht später den Tausch Grey-Box → Dyssol → Realdaten
  ohne Änderung am Trainingscode.
- **SOLID** beachten, insbesondere Open-Closed: eine neue Architektur hinzufügen darf keine
  Änderung an Trainings- oder Evaluationscode erfordern.
- Komposition vor Vererbung. Vererbungstiefe > 2 vermeiden.

**Bewusste Abweichung: `steady_state.py` ist eine Modulfunktion, keine ABC.** Der exakte
Beharrungszustands-Löser (`solve_steady_state`) ist bewusst *kein* fünfter Punkt in der
ABC-Liste oben. Eine Abstraktion (z. B. `SteadyStateSolver`) über genau eine Implementierung
ist Vorratsbau: der einzige absehbare zweite Fall -- ein stationärer Fließschema-Modus von
Dyssol -- liefert keine rechte Seite, gegen deren Nullstelle sich bisezieren ließe, die
Schnittstelle des zweiten Falls ist also noch nicht einmal in der Form absehbar. Die ABC
wird nachgezogen, sobald dieser zweite Fall implementiert wird, nicht vorher.

**Bestätigung der Konvention: `sorption.SorptionIsotherm` ist eine ABC.** Anders als beim
`SteadyStateSolver`-Fall oben lagen hier von Anfang an zwei konkrete Implementierungen vor
(`ConstantIsotherm`, `GabIsotherm`) -- dieselbe Schwelle, die `excitation.DisturbanceDesign`
bereits über zwei Implementierungen zieht. Eine ABC ist hier also keine Vorratsabstraktion,
sondern folgt der bestehenden Konvention; siehe `04_stand_ap1.md`, Befund 34.

## Reproduzierbarkeit

- **Zentrales Seed-Handling** an einer Stelle (Python, NumPy, PyTorch, CUDA)
- **Config-Serialisierung**: die vollständige aufgelöste Konfiguration wird zu jedem Run gespeichert
- **Deterministische Splits**: Split-Zuordnung aus Lauf-IDs und Seed berechnet, nicht zufällig
  zur Laufzeit gezogen
- Git-Commit-Hash im Run-Log

## Tests

- Unit-Tests für alle physikalischen Bilanzen des Grey-Box-Modells (Masse, Energie schließen
  innerhalb Toleranz)
- Test, dass Splits **keine überlappenden Fenster** zwischen Train und Test enthalten
  (Leakage-Regression)
- Property-based Tests (Hypothesis) für Constraint-Projektionen: Output verlässt den
  physikalischen Wertebereich nie
- Smoke-Test, der die gesamte Pipeline mit winzigem Datensatz durchläuft
- **Leakage-Test für die Beobachtbarkeit:** ist eine Zielgröße als nicht inline gemessen
  konfiguriert, darf sie in der Fensterhistorie nirgends auftauchen. Ein Leck dort sieht in
  allen Metriken wie ein exzellenter Soft Sensor aus und ist an der Anlage nicht
  reproduzierbar — es macht einen Vergleich unbrauchbar, ohne ihn verdächtig zu machen.
- **Test für die Residual-Äquivalenz:** ein Modell mit nullgesetztem Kopf muss *exakt* die
  parameterfreie Baseline reproduzieren (Persistenz bei Inline-Messung,
  Mittelwertvorhersage ohne). Ohne PyTorch ist das am `RandomFeatureNARX` prüfbar, also
  auch die Vorgabe selbst und nicht nur eine Implementierung.
- **Kein blind geschriebener Architekturcode.** Ein Modul, in dem kein Tensor-Vorgang
  ausgeführt wurde, wird im Statusdokument als nicht ausgeführt gekennzeichnet und liefert
  keine berichtbare Zahl. Die zugehörigen Tests überspringen sich mit
  `pytest.importorskip` und sind die erste Aufgabe auf einer Maschine mit der fehlenden
  Abhängigkeit.

## Notebooks

- Dienen der **Visualisierung und Exploration**, nicht der Implementierung
- Importieren aus `src/`, enthalten keine Logik, die nicht auch als Modul existiert
- Kommentiert und erzählend: was wird gezeigt, was ist daran zu sehen
- Standard-Visualisierungen: Trajektorien des Prozessmodells, Anregungssignale,
  Lernkurven mit Streuband, Rollout-Vergleiche gegen Ground Truth
- **Jede Lernkurve trägt ihre Auflösungsschwelle und, wo bekannt, ihre Rauschgrenze.** Das
  Band ist Teil der Plot-Funktion in `evaluation/plots.py` und nicht des Aufrufers —
  liegt die Logik in der Notebook-Zelle, entsteht früher oder später eine Variante ohne
  Schwellenband, und die wandert in einen Vortrag.
- **Kurven verschiedener Beobachtbarkeits-Betriebsarten nicht in ein Diagramm.** Die
  Primärmetrik unterscheidet sich (laufweise gegen gepoolt); die y-Achsen bedeuten
  Verschiedenes.
- Ausgaben vor dem Commit leeren (`nbstripout`)

## Dokumentation

- Docstrings im NumPy- oder Google-Stil, mit Einheiten bei allen physikalischen Größen
- Einheiten immer explizit: `T_in_celsius`, `m_spray_kg_per_h` – nie nur `T` und `m`
- **Zeitspannen in Sekunden berichten, nicht in Abtastschritten.** Eine Historie von „zwei
  Schritten" ist bei Δt = 2 s etwas völlig anderes als bei Δt = 25 s. Schrittzahlen sind
  Implementierungsgrößen, Sekunden sind mit den Prozesszeitkonstanten vergleichbar — und nur
  der Vergleich mit ihnen ist eine Aussage.
