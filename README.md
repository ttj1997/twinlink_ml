# twinlink-ml

ML-Teil des BMWK-Verbundprojekts **TwinLink**, Teilvorhaben **TUHH-B** (TU Hamburg, SPE).

Stand: Schritte 1 und 2 des Arbeitsplans TUHH-B-AP 1 — **reduziertes Grey-Box-Prozessmodell
der Wirbelschichtsprühgranulation** und **Anregungsdesign**. Vertikaler Durchstich
(Schritt 3), Baselines (Schritt 4) und Evaluationsprotokoll (Schritt 5) sind noch offen.

## Installation

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"          # PyTorch wird für Schritt 1 nicht gebraucht
pytest                              # 35 Tests
ruff check . && mypy
```

## Erster Blick

```bash
# Prozessmodell: Beharrungszustand, Zeitkonstanten, statischer Varianzanteil
python scripts/report_operating_point.py conf/process/greybox_placeholder.yaml \
    --figure artifacts/operating_point.png

# Anregungsdesign: Abdeckung, Kollinearität, realisierte Bandbreite, Constraint-Verletzungen
python scripts/report_excitation.py conf/process/greybox_placeholder.yaml \
    --runs 16 --figure artifacts/excitation.png
```

Beide Skripte sind nach jedem Ersetzen von Platzhalterwerten zu wiederholen. Die
Verweilzeiten und Taktzeiten der Anregung werden aus den Zeitkonstanten des Prozessmodells
abgeleitet, nicht geraten — sie verschieben sich also mit jeder Änderung der Prozess-Config.

```python
from twinlink_ml.simulation import GreyBoxFluidBedGranulator, load_process_config

config = load_process_config("conf/process/greybox_placeholder.yaml")
simulator = GreyBoxFluidBedGranulator(config)
trajectory = simulator.rollout(actions, disturbances, seed=0)   # (batch, time, channel)
```

```python
from twinlink_ml.simulation import ExcitationConfig, MixedExcitationDesign

excitation = ExcitationConfig.from_time_constants(
    tau_fast_s=9.5, tau_slow_s=16_000.0, sample_time_s=5.0, run_duration_s=28_800.0
)
design = MixedExcitationDesign(
    excitation, config.actions, config.disturbances,
    sample_time_s=config.numerics.sample_time_s,
    geometry=config.geometry, material=config.material, operation=config.operation,
)
schedule = design.build(64, seed=0)         # Sobol -> schedule.subset(8) bleibt gültig
initial = simulator.state_from_physical(*schedule.initial_states.T)
trajectory = simulator.rollout(
    schedule.actions, schedule.disturbances,
    run_id=schedule.run_id, seed=0, initial_state=initial,
)
```

## Modell in einem Absatz

Vier Zustände — Feststoffmasse, Wassermasse, Partikelanzahl, Betttemperatur — für ein
ideal durchmischtes Bett mit quasistationärer Gasphase. Enthalpie-, Wasser- und
Feststoffbilanz plus reduzierte Populationsbilanz (zwei Momente). Der mittlere Durchmesser
folgt algebraisch aus Masse und Anzahl, wodurch die Bilanzen linear und im Integrator
exakt erhalten bleiben. Trocknung über die normierte Trocknungskurve mit Sättigungsgrad
der Abluft; Fluidisierungsfenster nach Wen & Yu bzw. Haider & Levenspiel.
Details und Herleitungen in den Docstrings von `simulation/greybox.py`.

## Bekannte Grenzen

| Nicht enthalten | Konsequenz |
|---|---|
| Partikelgrößen*verteilung* (nur zwei Momente) | Projektziel 10 % PSD-Genauigkeit ist mit diesem Modell nicht adressierbar → Dyssol / AP 2 |
| Agglomeration, Abrieb, Bruch | Nukleationsterm ist ein Platzhalter für Überspray |
| Elutriation / Zyklonrückführung | `u_t` wird als Diagnose geführt, der Austrag nicht bilanziert. Im Standarddesign liegt `u_gas` in 13 % der Abtastschritte über `u_t` — dort fehlt Modellphysik, nicht Betriebsfähigkeit |
| Tropfengrößenverteilung | `p_nozzle` hat reine Kostenwirkung ohne Prozessnutzen → im RL fixieren oder mit Qualitätsstrafe versehen |
| Heizer- und Aktordynamik | Momentanleistung ist fast algebraisch in den Stellgrößen, siehe unten |
| Räumliche Gradienten (Sprüh-/Trocknungszone) | mittlere Größen only |

## Anregungsdesign (Schritt 2)

Sprungfolgen mit zufälligen Amplituden und Verweilzeiten, PRBS als echte m-Sequenz
(Schieberegister, nicht Bernoulli-Münze), stückweise lineare Rampen, Konstantläufe als
Referenz. Störgrößen mit Tagesgang plus Ornstein-Uhlenbeck-Drift, Betriebspunkte und
Anfangszustände über eine Folge niedriger Diskrepanz.

Drei Abweichungen vom Arbeitsplan, jeweils begründet und durch Tests abgesichert:

| Abweichung | Grund |
|---|---|
| **Sobol statt Latin-Hypercube** | Sobol-Teilmengen sind selbst Folgen niedriger Diskrepanz. Bei LHS ist ein 5er-Design kein Teil des 100er-Designs, und die Lernkurve vermischt Datenmenge mit Abdeckungsqualität. `schedule.subset(n)` ist damit für die Stützstellen zulässig. |
| **Ratenbegrenzung ist Pflichtbestandteil** | Ein PRBS springt über die halbe Stellgrößenspanne in einem Abtastschritt; keine reale Zuluftregelung tut das. Weil im Grey-Box-Modell die Heizerdynamik fehlt, ist die Ratenbegrenzung derzeit der einzige Ort, an dem Aktordynamik überhaupt vorkommt. |
| **Anfangszustände werden mitgestreut** | Startet jeder Lauf bei $d_{50}=400\,\mu m$ und wächst monoton, ist $d_{50}$ innerhalb eines Laufs eine Funktion des Zeitindex. Modelle mit Positionsinformation können die langsame Mode dann aus der Schrittzahl ableiten, ohne Physik zu lernen — kein Fenster-Leakage, aber dieselbe Wirkung. |

## Zwei Befunde, die den weiteren Plan betreffen

**1. Der Prozess ist mehrskalig, nicht durch *eine* Zeitkonstante beschreibbar.**
Mit den Platzhalterwerten: Feuchte ≈ 10 s, Temperatur ≈ 84 s, Massenregelung ≈ 600 s,
Partikelgröße ≈ 16 000 s. Steifigkeitsverhältnis ≈ 1700. Bei Δt = 5 s bräuchte die
langsamste Mode ca. 9600 Abtastschritte — für einen Transformer mit voller Attention
außerhalb des Sinnvollen. Die Wahl der Sequenzlänge ist damit eine Entscheidung
*darüber, welche Mode* modelliert wird, keine Hyperparameterfrage.

**2. Die Momentanleistung ist zu ≈ 96 % durch eine statische Abbildung der aktuellen
Stell- und Störgrößen erklärt.** Die Qualitätsgrößen sind dynamisch (R² 0.41 … 0.69), der
Energieverbrauch praktisch nicht. Für einen aussagekräftigen Architekturvergleich gehört
deshalb eine **statische Regression** in die Pflicht-Baselines — Persistenz allein ist für
diese Zielgröße die falsche Referenz, weil sie jeden Stellgrößensprung verliert.

## Was das Anregungsdesign über die Terminlage verrät

Mit den Platzhalter-Änderungsraten ist die Ratenbegrenzung auf **allen vier Kanälen
dauerhaft aktiv** (`Rate 1.00`). Das Signaldesign wird damit weitgehend durch die
Aktordynamik ersetzt: Sprünge und PRBS werden zu Rampen. Realisierte Bandbreite (90 % der
Signalenergie) gegen die Eckfrequenz der schnellsten Prozessmode:

| Kanal | f90 / f_fast | t_switch/τ_fast (volle / kleinste Amplitude) |
|---|---|---|
| `T_in` | 0.16 | 12.7 / 1.9 |
| `u_gas` | 1.00 | 3.6 / 0.5 |
| `m_spray` | 0.23 | 13.7 / 2.1 |
| `p_nozzle` | 0.53 | 6.3 / 1.0 |

Folge: die schnellste Prozessmode (Feuchte, ≈ 10 s) ist über `T_in` und `m_spray` mit
**keiner** Amplitude anregbar. Die effektiv schnellste beobachtbare Zeitskala wird von den
Aktoren gesetzt, nicht vom Prozess — und damit ist Δt = 5 s zu fein:

| Δt | Schritte pro 8-h-Lauf | Persistenz-NRMSE `T_bed` | Persistenz-NRMSE `X_prod` |
|---|---|---|---|
| 5 s | 5760 | 0.016 | 0.017 |
| 25 s | 1152 | 0.137 | 0.130 |
| 60 s | 480 | 0.234 | 0.045 |

Bei Δt = 5 s ist die Ein-Schritt-Vorhersage praktisch trivial — Persistenz liegt bereits
bei 1.6 % Fehler, und Architekturunterschiede verschwinden in der Seed-Varianz. **Die
Abtastzeit ist damit ein Versuchsparameter erster Ordnung, kein Implementierungsdetail.**
Vorschlag: Δt ≈ 25 s als Standard, Δt als Faktor in einer Vorstudie mitführen, und den
Mehrschritt-Rollout-Fehler als primäre Metrik behandeln statt den Ein-Schritt-Fehler.

Vorbehalt: die Änderungsraten in `conf/process/greybox_placeholder.yaml` sind selbst
Platzhalter. Sie bestimmen jetzt aber die Versuchsplanung mit — sie sind damit von einem
Nebenparameter zu einer der wichtigsten offenen Zahlen aufgestiegen.

## Struktur

```
conf/process/            Zahlenwerte des Prozessmodells, versioniert, mit TODO-Markierungen
conf/excitation/         aufgelöster Anregungsstand, gehört zu jedem Datensatz
src/twinlink_ml/
  simulation/
    backend.py           Array-Backend (NumPy | PyTorch) per Dependency Injection
    config.py            Frozen Dataclasses, Validierung, YAML-Roundtrip, Fingerprint
    thermo.py            Psychrometrie, Trocknungskinetik
    fluidization.py      u_mf, u_t, Bettdruckverlust
    energy.py            Leistungsaufnahme der Aggregate, Zielgröße dE
    base.py              ABC ProcessSimulator, Trajectory  ← Schnittstelle zu TUHH-A
    greybox.py           das Modell
    diagnostics.py       Betriebspunkt, Zeitkonstanten, statischer Varianzanteil
    excitation.py        Signalgeneratoren, QMC-Sampler, Ratenbegrenzung, Anfangszustände
    excitation_diagnostics.py  Abdeckung, Kollinearität, Bandbreite, Constraint-Verletzungen
  utils/seeding.py       getrennte Seeds für Daten / Rauschen / Modell
scripts/                 importieren die Module, enthalten keine Fachlogik
tests/                   75 Tests, davon 12 auf physikalische Bilanzen und Constraints
```

## Status der Zahlenwerte

`conf/process/greybox_placeholder.yaml` ist mit `provenance.status: placeholder` markiert
und löst beim Laden eine `PlaceholderConfigWarning` aus. **Keine** Absolutaussage über
Energieverbrauch ist mit diesem Stand belastbar; für den relativen Architekturvergleich
in AP 1 genügt er. Acht offene Punkte sind in `provenance.open_todos` gelistet.

## Schritt 5 — Evaluationsprotokoll

```bash
python scripts/study_variance_sources.py  --log artifacts/runs/variance.jsonl
python scripts/study_noise_level.py       --log artifacts/runs/noise.jsonl
python scripts/study_validation_policy.py --log artifacts/runs/policy.jsonl
jupyter lab notebooks/05_lernkurven_und_aufloesung.ipynb
```

Drei Regeln, die aus Schritt 5 folgen und im Code erzwungen werden:

1. **Seeds werden je Quelle variiert** (`SeedPlan`), nicht gemeinsam. Wer den Rausch-Seed
   über einen neuen Master-Seed zieht, variiert in Wahrheit den Datensatz.
2. **Jede Kurve trägt ihre Auflösungsschwelle.** Unterschiede darunter sind keine
   Ergebnisse. Mit drei Seeds liegt sie bei 10–15 % relativ.
3. **Jede Kurve trägt ihre Rauschgrenze.** Eine Kurve, die bei 0.07 abflacht, kann die
   Rauschgrenze erreicht haben statt der Modellgrenze — bei `d50_um` ist genau das der Fall.

## Schritt 6 — Architekturvergleich

```bash
python scripts/study_architectures.py --mode both --log artifacts/runs/architectures.jsonl
python scripts/study_history_length.py --log artifacts/runs/history.jsonl
```

Drei Regeln, die aus Schritt 6 folgen:

1. **Zwei Beobachtbarkeits-Betriebsarten.** `INLINE` (Zielgröße wird gemessen) und
   `SOFT_SENSOR` (nicht). Sie haben verschiedene parameterfreie Baselines und verschiedene
   Primärmetriken — `evaluation.protocol.primary_metric` entscheidet, nicht der Aufrufer.
2. **Der Residual-Kopf gehört an einen Anker.** Ohne Inline-Messung ist die Vorhersage
   direkt zu parametrisieren, sonst ist sie ein Integrator ohne Rückführung.
3. **Ein Modul ohne ausgeführten Tensor-Vorgang liefert keine berichtbare Zahl.**
   `torch_sequence.py` ist vollständig und nie ausgeführt; sechs Tests warten auf eine
   Maschine mit PyTorch.
