# Stand TUHH-B-AP 1: Schritte 1 bis 5 fertig, Schritt 6 zur Hälfte

> Diese Datei enthält die Begründungen und Messwerte, die nicht im Code stehen.
> Pflegehinweis: nach jedem abgeschlossenen Arbeitsplan-Schritt fortschreiben.
>
> Stand: 13.08.2026 · Schritte 1–5 erledigt. Schritt 6: Vergleichsrahmen, Beobachtbarkeit
> und das erste nichtlineare Modell stehen und sind gemessen. GRU und LSTM sind seit
> Befund 24 zum ersten Mal ausgeführt und bestehen ihre Bestandsprüfungen; die
> **Lernkurve** mit mehreren Seeds — die eigentliche Architekturaussage — steht noch aus.

## Was existiert

Installierbares Paket `twinlink_ml` (src-Layout, pyproject, Ruff/mypy/pytest,
**187 Tests ohne PyTorch, 193 mit** — die sechs Torch-Tests sind seit Befund 24 tatsächlich
gelaufen, nicht nur geschrieben). Laufzeitabhängigkeiten: numpy, scipy, pyyaml.

```
conf/process/greybox_placeholder.yaml   alle Prozess-Zahlenwerte, TODO-markiert
conf/excitation/broadband_derived.yaml  aufgelöster Anregungsstand
src/twinlink_ml/simulation/
  base.py          ABC ProcessSimulator, Trajectory, reset_sensor(), remeasure()
  greybox.py  config.py  thermo.py  fluidization.py  energy.py  diagnostics.py
  excitation.py  excitation_diagnostics.py
src/twinlink_ml/data/
  splits.py        RunSplitter-ABC: Index-Block / Hash / vorgegeben
  windows.py       Dreiblock-Fenster, Standardizer, **Beobachtbarkeit** (Schritt 6)
src/twinlink_ml/models/
  base.py          ABC SequenceModel (rollout-zentriert)
  baselines.py     Persistenz, **Mittelwertvorhersage**, statische Regression, ARX
  narx.py          **RandomFeatureNARX** — nichtlinear, ohne Torch, mit Modell-Seed
  torch_sequence.py GRU und LSTM hinter *einem* Trainer   <- ausgeführt, Befund 24
  registry.py      **Architektur-Slots** samt Verfügbarkeitsprüfung
src/twinlink_ml/evaluation/
  evaluator.py     RolloutEvaluator, pooled_nrmse / per_run_nrmse
  experiments.py   DataSpec, generate_data, evaluate_point, Lernkurve, Δt-Vorstudie
  protocol.py      Pool-Cache, Varianzzerlegung, Rauschstudie, NoiseFloor,
                   Validierungsvergleich, **ObservationMode**, **primary_metric**
  resolution.py    SeedSpread, Auflösungsschwelle, **Verdict**
  plots.py         Lernkurve / Zerlegung / Rauschstudie
src/twinlink_ml/utils/
  seeding.py       SeedBundle, VarianceSource, SeedPlan
  tracking.py      ExperimentLogger-ABC, JSONL, MLflow, read_jsonl
scripts/  report_operating_point  report_excitation  report_baselines
          study_sample_time  study_learning_curve  study_variance_sources
          study_noise_level  study_validation_policy
          **study_architectures**  **study_history_length**
notebooks/05_lernkurven_und_aufloesung.ipynb
```

**Aus `models/gru.py` ist `models/torch_sequence.py` geworden.** Der rekurrente Kern ist
austauschbar (`core="gru"` oder `"lstm"`); Residual-Kopf, Rollout-Semantik,
Verlustgewichtung und Early Stopping existieren **einmal**. Das ist die
Open-Closed-Zusage des Arbeitsplans: hätte jede Architektur ihr eigenes Modul,
unterschieden sich beim dritten Modell die Trainingsschleifen — und ein Vergleich, dessen
Kandidaten verschieden trainiert werden, vergleicht die Trainingsschleifen.

**Vorbehalt aufgehoben, mit Einschränkung (Befund 24):** `torch_sequence.py` wurde
inzwischen auf einer Maschine mit PyTorch ausgeführt. Die Bestandsprüfungen halten
(nullinitialisierter Kopf = Persistenz, GRU/LSTM teilen den Trainer, kein Blick in die
Zukunft im Rollout). Was noch aussteht, ist die eigentliche Architekturaussage — ein
einzelner Trainingslauf mit einem Seed bei n = 8 sagt nichts darüber, ob GRU oder LSTM die
Persistenz schlagen; das ist Aufgabe der Lernkurve in Schritt 6, Teil 2 (siehe unten).
Sechs Tests überspringen sich weiterhin auf Maschinen ohne PyTorch — das bleibt der
Fallback, nicht der Regelfall. xLSTM und Mamba sind aus demselben Grund wie zuvor noch
nicht implementiert, sondern nur als Slots deklariert: blind geschriebener Code für
Architekturen mit nichttrivialen Zustandsupdates sieht fertig aus und ist es nicht.

---

## Schritt 6, Teil 1: der Rahmen

### Beobachtbarkeit — zwei Betriebsarten statt einer

Ob die Produktfeuchte an der Zielanlage inline gemessen wird, ist offen (Priorität 2 in
`02_prozess_setup.md`). Beide Fälle sind implementiert und werden gefahren:

| | `INLINE` | `SOFT_SENSOR` |
|---|---|---|
| Historie enthält vergangene Zielwerte | ja | nein, ersetzt durch den Trainingsmittelwert |
| parameterfreie Baseline | Persistenz | Mittelwertvorhersage (`ClimatologyModel`) |
| Primärmetrik | NRMSE laufweiser Median | **NRMSE gepoolt**, siehe Befund 20 |
| Anker der Residual-Parametrisierung | letzte Messung | keiner → Vorhersage **direkt** |

Die Maskierung passiert in der **Fensterung** und nicht im Modell. Sonst müsste jede
Architektur die Fallunterscheidung selbst korrekt implementieren, und ein Modell, das sie
vergisst, bekäme eine nicht existierende Messung geschenkt — ein Leck, das in allen
Metriken wie ein exzellenter Soft Sensor aussieht und an der Anlage nicht reproduzierbar
ist. Ein Test vergleicht die Historie kanalweise gegen das Messarray. Maskiert wird
**nach** der Standardisierung, damit die Skalierer in beiden Betriebsarten identisch
bleiben; sonst unterschieden sich die Kurven zusätzlich in der Normierung der Eingänge.

### Architektur-Slots

`models/registry.py` hält jedes Modell **einmal**, mit Label, Fabrik und Voraussetzungen.
Ein Dict im Treiberskript leistet das nicht: bei drei Treibern steht in jedem eine eigene
Auswahl, und zwei Studien vergleichen unterschiedliche Modellmengen unter demselben Namen.
Nicht implementierte Slots (`xlstm`, `mamba`, `transformer`) sind **deklariert** und melden
verständlich, was fehlt — sie weglassen würde die Information verlieren, dass sie zum
Vergleich gehören.

### Verdikte statt Prozentzahlen

`architecture_verdicts` liefert eine Differenz **nur zusammen mit** der Auflösungsschwelle
aus Schritt 5 und formuliert das Ergebnis als Satz. Der Zweck ist Formulierungsdisziplin,
nicht Rechnung: „12 % besser" ist ohne die Schwelle daneben keine Aussage, und in einem
Bericht steht die Zahl dann trotzdem.

---

## Befunde mit Zahlen

Fortlaufende Nummerierung; 1–15 in den vorigen Fassungen. Wenn nicht anders angegeben:
Platzhalter-Config, Δt = 25 s, Pool 128 Läufe (64/32/32), h = 24 (= 600 s), drei Seeds,
gestreute Anfangszustände, Status `MODELL`.

### 16. Nichtlinearität hilft bei `dE` nicht — aber ihre Kurve läuft weiter

`energy_target`, NRMSE laufweiser Median, Betriebsart `INLINE`:

| n_train | Persistenz | Mittelwert | statisch | ARX | **NARX** |
|---|---|---|---|---|---|
| 8 | 0.925 | 1.277 | 0.264 | **0.232** | 0.384 |
| 16 | 0.925 | 1.299 | 0.233 | **0.180** | 0.318 |
| 32 | 0.925 | 1.293 | 0.221 | **0.171** | 0.270 |
| 64 | 0.925 | 1.293 | 0.219 | **0.172** | 0.219 |

Das nichtlineare NARX ist an jeder Stützstelle **schlechter** als der lineare ARX, bei
n = 64 um 27 % und damit oberhalb der Auflösungsschwelle von 13.2 % — nachweisbar
schlechter. Das passt zum statischen Varianzanteil von 0.974: bei `dE` ist kaum
Nichtlinearität zu holen, und die zusätzliche Kapazität kostet an den kleinen Stützstellen.

**Die wichtigere Beobachtung steht in der Spalte, nicht im Vergleich.** Der ARX ist bei 16
Läufen fertig (0.180 → 0.172 über den Faktor 4 der Datenmenge), das NARX fällt über den
ganzen Bereich monoton: 0.384 → 0.318 → 0.270 → 0.219, also 43 % von n = 8 auf n = 64 und
allein 19 % im letzten Schritt. **Das ist die erste Lernkurve im Projekt, die am rechten
Rand noch nicht flach ist** — und sie kommt von einem Modell ohne Gradientenabstieg.

1. **„16 Läufe genügen" ist eine Eigenschaft der linearen Modellklasse, nicht der
   Aufgabe.** Für M2 ist das wesentlich: der Trainingsdatenbedarf hängt von der
   Modellklasse ab und ist keine Prozesskennzahl.
2. **Die Lernkurven müssen über 64 hinaus.** Extrapoliert schneidet das NARX den ARX
   jenseits von 128. Die Stützstellen 128 und 256 sind mit dem Grey-Box-Modell billig und
   **vor** dem ersten neuronalen Lauf zu fahren, sonst wird eine Sättigung berichtet, die
   nur die Baseline betrifft.

### 17. σ_model erstmals gemessen — und nicht klein

Bei allen Baselines war die Modell-Achse der Varianzzerlegung exakt Null; die Zerlegung aus
Schritt 5 war an keinem Fall geprüft, in dem sie etwas zu zeigen hat. Das NARX zieht seine
Merkmale aus dem Modell-Seed:

| Zielgröße | σ_model n=8 | 16 | 32 | 64 | σ_joint bei n=64 |
|---|---|---|---|---|---|
| `energy_target` | 0.0220 | 0.0160 | 0.0134 | 0.0069 | 0.0224 |
| `product_moisture` | 0.1025 | 0.0204 | 0.1085 | 0.0263 | 0.5083 |

Bei `dE` und n = 64 ist σ_model ≈ 0.007 gegen σ_joint ≈ 0.022, also **rund ein Drittel der
Gesamtstreuung** — und σ_model fällt mit der Datenmenge, wie es sein muss. Bei der Feuchte
ist der Anteil klein.

Konsequenz: bei `dE` sind Architekturunterschiede unter etwa 4 % relativ reine
Initialisierung. Da die Auflösungsschwelle mit 13.2 % darüber liegt, bleibt sie das
bindende Kriterium — aber bei einem Modell mit echtem Gradientenabstieg kann σ_model
größer werden, und dann ist die Zerlegung nicht Diagnostik, sondern Voraussetzung der
Aussage.

### 18. Bei der Feuchte scheitert auch das nichtlineare Modell an der Persistenz

`product_moisture_kg_per_kg`, NRMSE laufweiser Median, `INLINE`:

| n_train | **Persistenz** | ARX | NARX | statisch | Mittelwert |
|---|---|---|---|---|---|
| 8 | **0.794** | 1.544 | 1.912 | 2.741 | 3.786 |
| 16 | **0.794** | 1.295 | 2.184 | 5.101 | 3.914 |
| 32 | **0.794** | 0.945 | 1.788 | 3.787 | 3.794 |
| 64 | **0.794** | 1.080 | 1.713 | 3.022 | 2.496 |

Das NARX ist **schlechter als der lineare ARX**. Damit ist die Diagnose geschärft: das
Problem ist nicht die fehlende Nichtlinearität, sondern die **Fehlerakkumulation im
autoregressiven Rollout**. Ein nichtlineares Modell mit demselben Ein-Schritt-Kriterium
akkumuliert nicht weniger, sondern mehr — es hat mehr Freiheitsgrade, um in einem
Zustandsbereich falsch zu sein, den es im Training nie gesehen hat.

Genau diese Lücke soll ein **auf den Rollout trainiertes** Modell schließen. Die Erwartung
für Teil 2 ist damit vor der ersten Zahl formuliert: ein Gewinn auf der Feuchte muss aus
dem Rollout-Verlust kommen, nicht aus der Nichtlinearität. Kommt er nicht, ist das Ergebnis
von AP 1, dass bei dieser Zielgröße Persistenz die beste implementierbare Vorhersage ist —
auch ein Resultat, und für einen Regler ein wichtiges.

### 19. Ohne Inline-Messung ist die Feuchte für alle klassischen Modelle unlösbar

`SOFT_SENSOR`, n = 64, Primärmetrik NRMSE **gepoolt**:

| Zielgröße | Mittelwert | statisch | ARX | NARX |
|---|---|---|---|---|
| `energy_target` | 1.007 | 0.195 | **0.168** | 0.176 |
| `product_moisture` | 1.015 | 0.969 | **0.969** | 0.985 |
| `d50_um` | 1.000 | 0.984 | 0.999 | 1.094 |

**Bei der Feuchte kommt kein Modell mehr als 5 % unter die Mittelwertvorhersage, bei `d50`
keines darunter.** NRMSE 0.97 heißt: die Vorhersage ist praktisch der Mittelwert.

Die Verdikte melden für n ≥ 16 „nachweisbar besser" — weil die Streuung der
Mittelwertvorhersage winzig ist und 5 % darüber liegen. Das ist der Fall, in dem
**statistische Nachweisbarkeit und praktische Relevanz auseinanderfallen**: nachweisbar
5 % besser als der Mittelwert ist kein Soft Sensor. Die Auflösungsschwelle schützt gegen zu
starke Behauptungen, nicht gegen irrelevante — die Relevanzgrenze ist getrennt zu setzen.

**`energy_target` braucht die eigene Messung dagegen nicht:** gepoolt 0.1675 mit
Inline-Messung gegen 0.1677 ohne, identisch bis zur vierten Stelle. Das passt zum statischen
Varianzanteil von 0.974 und ist für einen energieoptimierenden Regler eine gute Nachricht:
die Energievorhersage hängt an keiner Zustandsmessung.

### 20. Die Primärmetrik hängt von der Beobachtbarkeit ab

Der Befund aus Schritt 3 lautete: der gepoolte NRMSE ist zu klein, weil die gepoolte
Streuung die Niveauunterschiede zwischen den Läufen enthält, „und die muss kein Modell
vorhersagen, sie stehen in der Fensterhistorie". **Der zweite Halbsatz ist eine
Voraussetzung, nicht eine Tatsache** — er gilt nur bei Inline-Messung.

Ohne Messung ist das Niveau Teil der Aufgabe. Zusätzlich wird die laufweise Normierung dort
instabil: bei einem Lauf mit kleiner Eigenstreuung geht der Nenner gegen Null. Gemessen an
der parameterfreien Mittelwertvorhersage auf der Feuchte:

| Normierung | Median | σ über Seeds |
|---|---|---|
| laufweiser Median | 2.50 … 3.91 | bis **19.8** |
| gepoolt | 1.007 … 1.027 | 0.004 … 0.109 |

Deshalb: `INLINE` → laufweiser Median, `SOFT_SENSOR` → gepoolt (`primary_metric`). Beide
werden immer protokolliert; die Wahl betrifft, welche berichtet wird. **Kurven beider
Betriebsarten dürfen nicht in ein Diagramm** — die y-Achsen bedeuten Verschiedenes.

### 21. Residual-Parametrisierung ohne Anker ist ein Integrator ohne Rückführung

Beim ersten Durchlauf lag das NARX im Soft-Sensor-Betrieb auf der Feuchte bei NRMSE 40 bis
60 statt bei 1. Ursache: die vom Arbeitsplan vorgeschriebene Residual-Parametrisierung
`ŷ(t+1) = y(t) + Δ` setzt voraus, dass `y(t)` **bekannt** ist. Ohne Messung tritt im Rollout
die eigene Vorhersage des Vorschritts an ihre Stelle, und über 24 Schritte summiert sich
jeder Fehler des Zuwachses auf.

Korrigiert: für nicht beobachtete Kanäle ist der Anker konstant der standardisierte
Trainingsmittelwert, die Vorhersage also **direkt** statt inkrementell. Damit gilt die
Vorgabe des Arbeitsplans — Nullgewichte entsprechen der besten parameterfreien Baseline —
in beiden Betriebsarten; nur ist diese Baseline im einen Fall die Persistenz und im anderen
die Mittelwertvorhersage. **Das ist eine Präzisierung der Vorgabe für alle Architekturen in
Teil 2:** der Residual-Kopf gehört an einen Anker, und wo es keinen gibt, gehört er
weggelassen.

### 22. Nebenbefund zur Ridge-Regularisierung

Bei erdrückter Regularisierung konvergiert das NARX **nicht** gegen die Persistenz, sondern
gegen Persistenz plus eine Konstante — den mittleren Ein-Schritt-Zuwachs des
Trainings-Splits. Ursache ist die bewusste Entscheidung in `_ridge_solve`, den
Achsenabschnitt nicht zu bestrafen. Das ist gewollt (eine driftkorrigierte Persistenz), aber
es ist **nicht** die Persistenz-Baseline. Zwei Tests halten beide Fälle getrennt fest,
damit die untere Schranke des Vergleichs nicht stillschweigend verrutscht.

### 23. Die Historienlänge ist nicht der Engpass — der Rollout-Horizont ist es

Anlass war die Frage, ob die Historie von zwei Abtastschritten (4 s) aus dem Vorprojekt zu
kurz ist. Δt = 25 s, Horizont 600 s fest, n_train = 32, `INLINE`, ARX- und NARX-Ordnung auf
`min(4, L)` mitgeführt:

| Historie | Schritte | Persistenz | ARX `dE` | NARX `dE` | ARX `X_prod` | NARX `X_prod` |
|---|---|---|---|---|---|---|
| 25 s | 1 | 0.949 | 0.204 | 0.302 | 1.035 | 1.877 |
| 50 s | 2 | 0.948 | 0.191 | 0.291 | **0.758** | **1.263** |
| 100 s | 4 | 0.951 | **0.189** | 0.299 | 0.810 | 1.487 |
| 200 s | 8 | 0.943 | 0.190 | **0.265** | 0.828 | 1.414 |
| 800 s | 32 | 0.940 | 0.192 | 0.273 | 0.822 | 1.464 |
| 1600 s | 64 | 0.939 | 0.195 | 0.282 | 0.831 | 1.463 |

**Der Sprung liegt zwischen einem und zwei Schritten, danach ist die Kurve flach.** Auf der
Feuchte verbessert der zweite Lag den ARX um 27 % (1.035 → 0.758); alles darüber
verschlechtert leicht.

**Wichtige Einschränkung, die diese Tabelle nicht überschreiten darf:** ARX und NARX nutzen
höchstens vier Verzögerungen. Ab L = 4 wird die längere Historie vom Modell **gar nicht
gelesen**; die Zeilen darunter messen nur, wie sich die Fensterpopulation verschiebt. Die
Frage „nützt einem Modell, das 32 Schritte *lesen kann*, die längere Historie?" ist mit
dieser Modellklasse **nicht** beantwortbar. Sie ist die eigentliche Aufgabe von Teil 2 —
und der Grund, warum die Sequenzlänge dort als Modellentscheidung und nicht als
Hyperparameter geführt wird.

Dieselbe Studie bei Δt = 2 s, also der Abtastzeit des Vorprojekts, mit demselben
physikalischen Horizont von 600 s (= 300 Rollout-Schritte), Laufdauer 2 h, n_train = 16:

| Historie | Schritte | Persistenz | ARX `dE` | ARX `X_prod` | NARX `X_prod` |
|---|---|---|---|---|---|
| **4 s** | 2 | 1.129 | 0.462 | 3.016 | 16.77 |
| 50 s | 25 | 1.140 | **0.398** | **2.742** | 13.21 |
| 350 s | 175 | 1.196 | 0.549 | 4.202 | 18.00 |

Von 4 s auf 50 s Historie gewinnt der ARX auf der Feuchte 9 %, auf 350 s verliert er 40 %.
**Die Historienlänge ist also der schwächere Hebel; die Zahl der Rollout-Schritte ist der
starke.** Bei Δt = 2 s liegt der ARX-Feuchtefehler bei 2.7 bis 4.2, bei Δt = 25 s bei 0.76
bis 0.83 — Faktor 4, allein durch 300 statt 24 autoregressive Schritte über denselben
physikalischen Horizont. Das bestätigt eine frühere Beobachtung aus Schritt 3 (Δt-Vorstudie)
und schärft sie: Δt fein ist nicht nur teuer, es ist bei autoregressivem Rollout
**schädlich**.

**Antwort auf die Ausgangsfrage:** die 4 s des Vorprojekts sind für dessen Aufgabe — eine
Ein-Schritt-Schätzung bei Δt = 2 s — nicht offensichtlich zu kurz. Für die hiesige Aufgabe
sind sie es, aber nicht *weil* das Gedächtnis kurz ist, sondern weil ein Modell mit 4 s
Eingangshorizont keine Aussage über 600 s treffen kann, ohne 300-mal auf sich selbst
zurückzugreifen. Der Unterschied ist die Aufgabenstellung, nicht die Fensterlänge.

---

## Was das Paper des Vorgängers ändert

Kräuter, Zhou & Heinrich (ABS23, Vorprojekt TwinGuide) vergleichen einen modellbasierten
(Dyssol-Fließschema) und einen datengetriebenen Soft Sensor für die Partikelfeuchte an der
SPE-Laboranlage. Die verwertbaren Zahlenwerte sind in `02_prozess_setup.md` als `LAB`
eingearbeitet. Vier Punkte gehören hierher, weil sie die Methodik betreffen.

### Die Feuchte ist im SPE-Labor inline messbar

Mikrowellenresonanz (TEWS MW4200), kalibriert gegen thermogravimetrische Referenzproben.
Insgesamt elf Inline-Kanäle. Für AP 2 heißt das: der Merkmalsraum an der realen Anlage ist
**reicher** als im Grey-Box-Modell, nicht ärmer. Für Anwendungspartner bleibt die Frage offen — und
genau deshalb sind ab jetzt beide Betriebsarten implementiert und werden beide berichtet.

### Das berichtete R² = 0.994 ist kein Vergleichsmaßstab für AP 1

Der datengetriebene Soft Sensor (Extra Trees auf 12 Hauptkomponenten aus 74 Merkmalen)
erreicht auf den Testdaten ein Bestimmtheitsmaß von 0.994, das Fließschema 0.163. Meine
Zahlen liegen im Soft-Sensor-Betrieb bei NRMSE ≈ 0.97, was einem R² nahe Null entspricht.
Der Unterschied ist **fast vollständig methodisch**, nicht modellbedingt:

1. **Die Aufteilung erfolgt zufällig über Beobachtungen, nicht über Chargen.** 31 281
   Trainings- gegen 1 738 Testpunkte, zufällig gemischt, zusätzlich per
   Kolmogorow-Smirnow-Test darauf geprüft, dass alle drei Teilmengen derselben Verteilung
   folgen. Bei Δt = 2 s liegt der Nachbarpunkt einer Testbeobachtung fast immer im
   Training. Das ist genau das Fenster-Leck, gegen das die Splits hier auf **Lauf-Ebene**
   liegen. Der KS-Test macht es nicht besser, sondern schlimmer: er wählt aktiv die
   Aufteilung *ohne* Verteilungsverschiebung — also den Fall, den ein Übertragungsmodell
   von Simulation auf Anlage gerade nicht bekommt.
2. **Die Zielgröße ist nicht kausal geglättet.** Savitzky-Golay dritter Ordnung über 100
   Punkte, bei Δt = 2 s also 200 s. Ein zentriertes Fenster benutzt Zukunft; als
   Online-Messung ist das nicht implementierbar.
3. **Es ist eine Ein-Schritt-Aufgabe.** Die Zeitinformation erreicht das Modell über erste
   und zweite Ableitung, also zwei Zeitschritte. Für eine Regelung braucht man Minuten.
4. **17 bis 34 % der Rohdaten wurden je Charge als Ausreißer entfernt.** Das ist keine
   Nebenbemerkung, sondern eine Aussage über die Sensorqualität (siehe unten).

**Das entwertet die Arbeit nicht.** Für Sensorrekonstruktion und Prozessüberwachung — den
erklärten Zweck — ist genau das die richtige Aufgabenstellung, und 0.994 ist unter diesen
Bedingungen eine legitime obere Schranke. Aber die Zahl wird im Verbund fallen, und dann
sind meine Lernkurven ohne diese Erklärung nicht verteidigbar.

**Vorschlag, konkret und billig:** die Gegenüberstellung beider Aufteilungen auf demselben
Datensatz messen und veröffentlichen. Der Abstand zwischen fenster- und laufweisem Split
ist selbst ein Ergebnis, er ist mit dem vorhandenen `RunSplitter`-ABC in einem Tag zu
erzeugen (der `HashRunSplitter` ist auf Lauf-Ebene, ein Fenster-Splitter wäre die
Vergleichsvariante), und er ist genau die Methodenaussage, die M2 verlangt.

### Das Fließschema scheiterte an der Sorptionsisotherme

Für die Simulation war die Isotherme eines gröberen Materials hinterlegt; erste Versuche mit
dem passenden Material zeigten deutlich bessere Übereinstimmung. Die Sorptionsisotherme
rückt damit in der Prioritätenliste von Platz 7 auf **Platz 3**: sie ist keine
Feinjustierung, sondern im Vorprojekt der Unterschied zwischen R² = 0.16 und brauchbar.

Sie ist außerdem der Grund, warum das Feuchteniveau im Platzhalter um **Faktor 10** falsch
liegt — mit Folgen für die Zeitkonstanten (siehe `02_prozess_setup.md`): die schnelle Mode
steigt von 13.7 s auf ~35 s, die zweite von 85 s auf ~370 s, die Steifigkeit fällt von 1171
auf ~450. Damit ist Δt = 60 s nicht mehr ausgeschlossen, und die Δt-Vorstudie ist zu
wiederholen.

### Das Sensormodell ist zu gutmütig

0.5 % relatives Rauschen plus ein PT1 mit 10 s beschreiben **nicht** ein Signal, aus dem
ein Viertel der Punkte als Ausreißer entfernt werden muss. Ein realistisches Sensormodell
braucht einen **Ausreißerprozess** (schwerschwänzig, nicht gaußisch) neben dem additiven
Rauschen. Das ist ein Modellierungsauftrag und kein Parameterwert — und er ist relevant,
weil die Rauschstudie aus Schritt 5 zeigt, dass die Rangfolge der Modelle gegen das
*Niveau* des gaußischen Rauschens robust ist. Gegen Ausreißer ist sie nicht geprüft, und
Ausreißer sind an der realen Anlage der dominierende Störmechanismus.

---

## Reproduzierbarkeit

Sieben Fingerprints je Lernkurvenpunkt, dazu je Record `variance/source`,
`variance/replicate`, `noise/relative_std`, **`observation/mode`**,
**`observation/unobserved`** und in der Historienstudie **`window/history_s`**. Alle
Metriknamen entstehen ausschließlich in `EvaluationResult.as_flat_metrics()`.

`collect_spreads` **verweigert** die Auswertung, wenn dasselbe Replikat mit zwei
verschiedenen Werten vorkommt. Das hat in diesem Schritt gegriffen: nach der Korrektur von
Befund 21 lagen im angefügten JSONL-Log beide Codegenerationen, und der Schutz hat es
gemeldet statt eine „Seed-Streuung" auszurechnen, die in Wahrheit der Effekt der
Codeänderung war.

### Fingerprint-Historie: `conf/process/greybox_gab_placeholder.yaml`, `sample_time_s=25 s` (angewendet)

**Zweck dieses Abschnitts.** In sechs Monaten muss aus diesem Dokument hervorgehen, ob zwei
Datensätze mit unterschiedlichem `process/fingerprint` inhaltlich vergleichbar sind oder
nicht -- ohne dass jemand die Git-Historie durchsuchen oder den Code nachvollziehen muss.
Anlass: Befund 33 (`arch_composite.jsonl`, `process/fingerprint=b24c5b728681bca9`) stimmt
mit keiner heutigen `conf/process/*.yaml` überein und liegt zeitlich vor dem ältesten Commit
dieses Repos -- die erzeugende Codebasis ist unauffindbar, der Fingerprint ist vorhanden,
aber nicht auflösbar. Diese Tabelle soll genau das für die GAB-Platzhalter-Config
verhindern.

Fünf Werte für `ProcessConfig.fingerprint` sind für diese Config/Abtastzeit-Kombination im
Umlauf, je nach Code-Stand. `fingerprint` gilt immer für die ANGEWENDETE Config, nicht für
den Dateidefault (Docstring von `ProcessConfig.fingerprint`, `config.py`, ergänzt in
Commit `89a91fd`) -- die Datei selbst lädt mit `sample_time_s=5.0` (Datei-Default,
Fingerprint `36de70f171c8fe32`); `DataSpec.sample_time_s=25.0` überschreibt das in
`generate_data()`, bevor der Fingerprint gebildet wird. Alle fünf Werte unten beziehen sich
auf den angewendeten Fall (`sample_time_s=25 s`):

| Fingerprint | Code-Stand | Welche Logs ihn tragen |
|---|---|---|
| `82f65f576a2e5253` | vor Einführung von `NumericsConfig.enforce_gas_velocity_projection` | Alle committeten Logs vor Befund 46: `mamba_reduced_comparison.jsonl`, `mamba_reduced_comparison_before_rmsnorm_fix.jsonl`, `mamba_reduced_comparison_rmsnorm_only.jsonl`, `arch_gab_constant_holdout.jsonl`, `xlstm_reduced_comparison.jsonl`, `befund42_tbed_exhaust_sweep.jsonl` u. a. |
| `40aae2b4526a5ed7` | nach Einführung des Feldes, Schalter AUS (Default) | **Kein** committetes Log trägt diesen Wert -- stand nur als `EXPECTED`/Regressionswert in Skripten und Test (siehe unten, jetzt beide veraltet). Kein Trainingslauf wurde unter diesem Fingerprint durchgeführt und geloggt. |
| `eb70e9c6d6590cf7` | nach Einführung des Feldes, Schalter AN | Trägt **kein** Log und **keine** Skript-Assertion -- einmalig berechnet während der Befund-46-Untersuchung (`scripts/probe_gas_velocity_projection.py`), jetzt nur noch hier in der Doku und in Befund 46 selbst festgehalten (ebenfalls veraltet, siehe unten). |
| `078572b3a7a0fb45` | nach Einführung von `ExhaustConfig`/`outlet_duct_heat_loss_k` (Kommit "ExhaustConfig: Rohrleitungsverlust und benannte Ablufttemperatur"), Schalter AUS | Aktueller Stand für `enforce_gas_velocity_projection=False`. Trägt als Erstes `run_arch_constant_holdout_worker.py` für X2 (GRU-Lauf gegen `outlet_temperature_celsius`, siehe Befund 50). Verschiebt sich gegenüber `40aae2b4526a5ed7` aus **zwei** unabhängigen Gründen gleichzeitig: (1) das neue `exhaust`-Feld selbst, (2) ein zusätzlicher `provenance.open_todos`-Eintrag in `greybox_gab_placeholder.yaml` (reiner Dokumentationstext) -- beide Änderungen liegen im selben Commit und sind aus dem Gesamthash allein nicht auseinanderzuhalten. Genau das ist der Fingerprint als monolithischer Hash: ein Dokumentationskommentar ist darin nicht von einer Physikänderung unterscheidbar. Anlass, künftige Reproduktionsprüfungen gegen die relevanten `section_fingerprints` (`config.py`, seit G4b) statt gegen den Gesamthash zu führen -- hier nur als Begründung notiert, nicht umgesetzt. |
| `2d2679eeed716577` | nach Einführung von `ExhaustConfig`, Schalter AN | Analog zu `078572b3a7a0fb45`, mit `enforce_gas_velocity_projection=True`. Trägt seit Befund 51 das erste committete Log: `arch_learning_curve_projected.jsonl` (200 Records, Commit `0b3eaaf`), Ersatz (nicht Korrektur) für `arch_composite.jsonl` (Befund 33). |
| `6ecedc6bd993fb3d` | Pseudonymisierungs-Schnitt (2026-09-25): `provenance.open_todos`/`description` aus dem Fingerprint ausgeschlossen (`without_documentary_provenance`, siehe `twinlink_ml.utils.fingerprint_history`), Schalter AUS | Aktueller Stand seit dem Schnitt für `enforce_gas_velocity_projection=False`. Äquivalent zu `078572b3a7a0fb45` -- reine Dokumentationsänderung (Partnernamen durch Pseudonyme ersetzt), kein Schema, keine Physik; Nachweis: Vorgänger- und Nachfolger-Config lieferten unter dem neuen Hashing-Verfahren identische Werte (Details `docs/ARCHIV.md`). Alte, eingefrorene Records/Checkpoints tragen weiterhin `078572b3a7a0fb45`; `fingerprints_equivalent()` erkennt beide als gleichwertig. |
| `bddca37f716f5cfc` | wie oben, Schalter AN | Äquivalent zu `2d2679eeed716577`, analog zur Zeile darüber. |
| `2acd06f38963c3af` | wie oben, `greybox_gab_placeholder.yaml` unverändert geladen (kein Override) | Äquivalent zu `abbe8325f1326d80` -- der VA2-Ankerpunkt (`twinlink_ml.rl.env.EXPECTED_CONFIG_FINGERPRINT`, `artifacts/runs/bb1_bb3_operating_point_regret.json`). |

**Vergleichbarkeit:** `82f65f576a2e5253` → `40aae2b4526a5ed7` → `078572b3a7a0fb45` sind bei
Schalter AUS inhaltlich äquivalent (bitgleiches Simulationsverhalten, zwei zusätzliche
Schema-Felder mit jeweils identitätserhaltendem Default-Wert) -- belegt durch
`test_gas_velocity_projection_disabled_matches_legacy_baseline`
(`tests/test_gas_velocity_projection.py`, prüft die Kette bis zum jeweils aktuellen
Fingerprint gegen dasselbe eingefrorene Trajektorien-Fixture). Sie sind aber NICHT bitgleich
und ihre `process/fingerprint`-Werte dürfen nicht als Gleichheitskriterium in einer
Skript-Assertion gegeneinander verwendet werden (die `EXPECTED`-Dicts in den betroffenen
Skripten -- `run_xlstm_reduced_comparison.py`, `diag_xlstm_worst_run.py`,
`run_mamba_reduced_comparison.py`, `probe_gas_velocity_projection.py`,
`probe_e2_e3_e4_lag_depth.py`, `probe_sweep_contamination.py`,
`probe_e0_normalisation_artifact.py`/`probe_f3_f4_arch_holdout_inference.py`,
`tests/test_script_target_names_schema.py` -- tragen weiterhin `40aae2b4526a5ed7`, bewusst
nicht angepasst, siehe Aufgabenstellung zu `ExhaustConfig`). `eb70e9c6d6590cf7` und
`2d2679eeed716577` (Projektion aktiv) sind NICHT äquivalent zu den AUS-Werten -- andere
Trajektorien, siehe Befund 46 für die gemessene Wirkung (Restverletzung 0.16 % statt
6.0 %). Volle Herleitung und Kontext: Befund 46.

**Nachtrag (Pseudonymisierungs-Schnitt, 2026-09-25).** Jede Erwähnung von
`078572b3a7a0fb45`/`2d2679eeed716577`/`abbe8325f1326d80` an anderer Stelle in diesem
Dokument bezieht sich auf den Stand VOR dem Schnitt -- diese Werte bleiben unverändert
stehen, weil sie die Records so tragen. Ab dem Schnitt gilt die geschlossene
Äquivalenztabelle oben (`twinlink_ml.utils.fingerprint_history.
PRE_CUT_FINGERPRINT_EQUIVALENTS`) und `docs/ARCHIV.md`.

---

## Offene Fragen, nach Dringlichkeit

1. **Maximale Änderungsraten der Stellgrößen.**
2. **Inline-Messbarkeit der Produktfeuchte bei Anwendungspartner** (neu auf Platz 2). Nach Befund 19
   haben die beiden Betriebsarten verschiedene Baselines, verschiedene Primärmetriken und
   verschiedene Ergebnisse — die Antwort entscheidet, welches Ergebnis gilt.
3. **Sorptionsisotherme des tatsächlichen Stoffsystems** (von Platz 7 auf 3).
4. **Zeitkonstante des Lufterhitzers.**
5. **Definition von `dE`.**
6. Betriebsart kontinuierlich vs. Batch — im SPE-Labor Charge, im Modell kontinuierlich mit
   Füllstandsregelung (τ = 600 s). Für AP 2 abzuschalten.
7. Rauschniveau der Anlagensensoren — abgestuft, aber mit neuem Inhalt: gebraucht wird ein
   **Ausreißerprozess**, nicht ein anderer Rauschwert.
8. **Normierte Trocknungskurve, kritische Gutsfeuchte und Sättigungsgrenze der Abluft des
   realen Stoffsystems** (`thermo.evaporation_rate_kg_per_s`, `critical_moisture_kg_per_kg`,
   `max_saturation_degree` — alle drei Felder von `config.py:MaterialConfig` ohne
   Quellenangabe, ergänzt zu Punkt 3). **Nach Befund 54 (DA1) weiterhin zu erfragen** —
   die qualitative Regret-Aussage aus Befund 52 hängt nicht an dem genauen Wert, seine
   Größenordnung (Faktor 2-3 zwischen 0,10 und 0,20 kg/kg) aber schon.
9. **Bettmasse/Geometrie der Laboranlage** (`geometry.bed_mass_solid_setpoint_kg`, SPE-TODO
   im Dateikopf) — **nach Befund 54 (DA2) für Befund 52s Regret nicht mehr entscheidend**
   (robust über den vollen getesteten `tau_4`-Skalenbereich, 4000-32000 s; der Parameter
   ist im Modell kein isolierter Verweilzeitparameter, sondern verschiebt gleichzeitig
   Verweilzeit, Bettwärmekapazität und Druckverlust, Befund 54e). **Für die
   `Q_rec`-Chargenmittel-Abweichung aus Befund 53 nach Befund 54d NICHT mehr die
   erfragenswerte Größe** — DB1 zeigt, die Abweichung kommt vom nominalen
   Anfahr-Anfangszustand, nicht von `tau_4`/Bettmasse selbst; für kontinuierlichen
   Betrieb ist der stationäre `Q_rec`-Wert die richtige Referenz, unabhängig vom
   konkreten Bettmasse-Wert. Bleibt relevant nur, falls Punkt 6 (Betriebsart) sich als
   tatsächlich chargenweise statt kontinuierlich herausstellt.

### 24. Torch zum ersten Mal ausgeführt — die Bestandsprüfungen halten, die Trainingsbehauptung nicht

Auf einer Maschine mit PyTorch: fünf der sechs übersprungenen Tests grün, ein sechster
(`test_gru_trains_and_beats_persistence_on_energy`) rot. Die drei Bestandszusicherungen,
auf die es ankommt, halten alle:

- **Nullinitialisierter Kopf = Persistenz, exakt** (`test_untrained_network_equals_persistence`).
- **GRU und LSTM teilen den Trainer** (`test_lstm_core_shares_the_trainer`).
- **Kein heimliches Fortschreiben der Wahrheit im Rollout** (`test_gru_rollout_does_not_see_future_targets`).

Der rote Test behauptete mehr, als er mit seinem Aufbau zeigen konnte: `n_train = 8`, ein
einziger Seed, 40 Epochen mit `batch_size = 256` bei ≈144 Trainingsfenstern — also **ein**
Gradientenschritt je Epoche, 40 insgesamt. Ergebnis: laufweiser NRMSE 0.691 (trainiert)
gegen 0.651 (Persistenz) auf `energy_target`, h = 8. Das GRU wurde durch das Training
**schlechter**, nicht nur „nicht besser".

**Das ist kein Infrastrukturfehler.** Es ist genau der Fall, den die Auflösungsschwelle aus
Schritt 5 vorhersagt: bei drei Seeds liegt sie auf `dE` bei 13.2 % relativ; mit einem
einzigen Seed ist noch weniger nachweisbar. Ein Test, der „GRU schlägt Persistenz bei
n = 8, Seed 0" als Bestandsprüfung verlangt, verkleidet eine empirische Frage als
Regressionstest — dieselbe Verwechslung, gegen die `architecture_verdicts` in Schritt 6
Teil 1 bereits eingeführt wurde. Der Test ist korrigiert: er prüft jetzt, dass der
Optimierer seinen eigenen Trainingsverlust senkt, dass sich die Kopfgewichte vom Nullinit
entfernen und dass die Vorhersage endlich bleibt — nicht, in welche Richtung sich die
Generalisierung bewegt. Die eigentliche Frage — schlägt ein GRU die Persistenz bei
realistischem Seed-Budget — ist Aufgabe von `study_architectures.py` mit mehreren Seeds,
nicht eines Unit-Tests mit einem.

**Vorläufige Lesart, mit Vorbehalt (ein Seed, n = 8, 40 Gradientenschritte):** passt zu
Befund 16/18 — auch das nichtlineare NARX brauchte mehr als 64 Läufe, um zu zeigen, wo
seine Kurve hingeht, und war bei kleinem n durchgehend schlechter als der lineare ARX. Ein
GRU mit Zufallsinitialisierung im rekurrenten Kern (nur der Kopf ist nullinitialisiert) hat
bei acht Trainingsläufen dieselbe Verwundbarkeit. Ob das bei mehr Epochen, größerem
Lernraten-Budget oder erst bei größerem n verschwindet, ist offen — genau die Frage für
Schritt 6, Teil 2, Punkt 2 unten.

### 25. Registrierungsbug: GRU/LSTM galten nach Torch-Installation weiterhin als „nicht verfügbar"

Beim ersten realen Lauf von `study_architectures.py` ohne explizite `--models`-Liste
aufgefallen: die Statustabelle zeigte `gru`/`lstm` als `fehlt: —` an, obwohl Torch
installiert war. Ursache: `ModelSlot.is_available` prüfte `self.factory is not None` —
aber `gru`/`lstm` sind absichtlich mit `factory=None` registriert, damit ihre Fabrik erst
bei Bedarf aus `torch_sequence.py` aufgelöst wird (das Paket soll ohne Torch importierbar
bleiben). Die Prüfung erklärte damit **jeden** lazy registrierten Slot dauerhaft für nicht
verfügbar, unabhängig vom tatsächlichen Torch-Status.

Mit explizitem `--models gru lstm ...` (wie im ersten Lauf) blieb das folgenlos, weil
`build_factories` bei einer expliziten Liste nicht über `is_available` filtert. Ohne
`--models` — dem Normalfall für `available_labels()` — wäre die Lernkurve für GRU/LSTM
dagegen leer geblieben, ohne Fehlermeldung.

Behoben durch ein eigenes Feld `implemented: bool`, das die Frage „ist der Slot
geschrieben" von der Frage „ist die Fabrik schon konkret oder noch lazy" trennt.
`xlstm`/`mamba`/`transformer` sind jetzt explizit `implemented=False` und bleiben damit
auch bei installiertem Torch als „nicht implementiert" markiert — vorher hätte
`describe_slots()`s Sonderfall (`factory is not None and kind is ARCHITECTURE`) das
korrekt verdeckt, aber aus dem falschen Grund. Regressionstest ergänzt
(`test_registry_availability_tracks_dependencies_not_the_lazy_factory`).

### 26. Negativer Transfer im Soft-Sensor-Betrieb: der geteilte Verlust schadet `energy_target`

**Ausgangsbeobachtung** (`study_architectures.py --mode both`, `artifacts/runs/arch_gru.jsonl`,
Feld `observation/mode`): im `SOFT_SENSOR`-Betrieb ist `energy_target` für GRU (NRMSE
0.28–0.46) und LSTM (0.60–0.68) an **jeder** Stützstelle nachweisbar schlechter als ARX
(0.17–0.24) — obwohl dieselben Kerne dieselbe Zielgröße im `INLINE`-Betrieb klar besser als
ARX treffen. Hypothese: `torch_sequence.py` trainiert `energy_target`,
`product_moisture_kg_per_kg` und `d50_um` mit einem gemeinsamen Encoder und einem einzigen
Skalarverlust; im Soft-Sensor-Betrieb sind die beiden Qualitätsgrößen für jedes bisher
geprüfte Modell praktisch unlösbar (Climatology-NRMSE dort ≈ 1.0, Befund 19). Ihr
Gradientenrauschen könnte über den geteilten Encoder die sonst gute Lösung für
`energy_target` verschlechtern — negativer Transfer zwischen Teilaufgaben ungleicher
Schwierigkeit.

**1. Verlustberechnung gelesen, nicht vermutet.** `_weighted_mse` in `torch_sequence.py`
bildet `(prediction - target)**2`, mittelt mit `.mean(dim=-1)` **ungewichtet** über die
Zielgrößen-Achse, gewichtet erst danach den Horizont (`horizon_weight_decay`) und mittelt
über den Batch. Es gibt **keine Gewichtung pro Zielgröße**, an keiner Stelle. Da
`Standardizer.fit` (`data/windows.py`) jeden Zielgrößenkanal auf dem Trainingssplit auf
Varianz 1 normiert, tragen alle drei Kanäle bei Trainingsbeginn mit vergleichbarer
Größenordnung zum Skalarverlust bei — unabhängig davon, wie lernbar sie sind. Die
Bestätigung ist damit Teil 1 der Hypothese: nicht „vielleicht dominieren die schweren
Zielgrößen", sondern strukturell **können** sie das, weil nichts sie davon abhält.

**2. Nicht-invasive Diagnose ergänzt.** `training_history` protokolliert jetzt zusätzlich
zum aggregierten `train_loss` denselben Wert **pro Zielgröße je Epoche**
(`train_loss/<name>`), auf dem bereits vorhandenen `prediction`-Tensor berechnet, ohne in
`loss.backward()` einzugehen. Bestehendes Verhalten unverändert — alle sechs
Torch-Bestandsprüfungen (`tests/test_experiments.py`) bleiben grün, unverändert 6 bestanden,
1 übersprungen bei `-k "torch or gru or lstm"`.

**3. Experiment** (`scripts/diagnose_soft_sensor_interference.py`, `SOFT_SENSOR`,
`n_train = 32`, `h = 24`, Seeds 1–3, gepoolter NRMSE — die Primärmetrik dieser Betriebsart,
Befund 20): (a) `target_names` wie bisher alle drei, (b) `target_names = ("energy_target",)`.
Beide Varianten laufen auf demselben Datenpool und demselben Modell-Seed
(`dataclasses.replace` auf demselben `GeneratedData`, nur `spec.target_names` unterscheidet
sich) — ein Unterschied kann also nicht aus einer anderen Datenrealisierung oder
Initialisierung kommen.

| Seed | (a) alle drei | (b) nur `energy_target` | ARX-Referenz | Climatology |
|---|---|---|---|---|
| 1 | 0.2476 | 0.0238 | 0.1706 | 1.0436 |
| 2 | 0.6495 | 0.0285 | 0.1737 | 1.0020 |
| 3 | 0.3422 | 0.0226 | 0.1310 | 1.0002 |
| **Median** | **0.3422** | **0.0238** | **0.1706** | 1.0020 |

**Die Hypothese ist bestätigt, deutlicher als erwartet.** Fall (b) schlägt nicht nur Fall
(a) (Faktor 14 im Median), sondern schlägt auch die ARX-Referenz um Faktor 7 — dieselbe
Zielgröße, mit der GRU im `INLINE`-Betrieb bereits gut abschneidet, ist im Soft-Sensor-Fall
allein trainiert nicht nur „näher an ARX", sondern deutlich besser als ARX. Fall (a) bleibt
dagegen bei allen drei Seeds schlechter als ARX — konsistent mit der Ausgangsbeobachtung.
Kein einzelner Seed kippt das Bild; bei drei Seeds ist das kein Beleg für einen exakten
Effektwert, aber der Abstand (14×) liegt weit über jeder Seed-Streuung, die in diesem
Projekt bisher gemessen wurde (Befund 17, größte gemessene Streuung ~20 % relativ).

**Zusatzbeobachtung aus der Per-Ziel-Diagnose** (letzter Seed, letzte Epoche, Fall a):

| | `energy_target` | `product_moisture_kg_per_kg` | `d50_um` |
|---|---|---|---|
| `train_loss/<name>` | 0.0151 | 0.1396 | 0.1682 |

`energy_target`s eigener Trainingsverlust ist am Ende des Trainings ~10× kleiner als der
der beiden Qualitätsgrößen — das Netz *lernt* den leichten Kanal durchaus. Der Schaden
entsteht nicht daran, dass `energy_target` im gemeinsamen Verlust untergeht, sondern
vermutlich an zwei Stellen, die dieses Skript nicht einzeln misst: (i) der rekurrente Kern
ist ein einziger geteilter Zustand, dessen Gradient aus allen drei Residuen gemischt wird —
ein „gutes" verstecktes Repräsentation für `energy_target` muss gleichzeitig eine für zwei
kaum lernbare Ziele sein; (ii) `gradient_clip` (Voreinstellung 1.0) wirkt über
`clip_grad_norm_` auf die **globale** Norm über alle Netzparameter gemeinsam, nicht getrennt
je Kopf oder Ziel — ein großer, verrauschter Gradient aus den Qualitätsgrößen skaliert damit
auch den an sich brauchbaren Gradienten für `energy_target` mit herunter. Beides ist
plausibel aus dem Code, aber **nicht gemessen** — eine Messung bräuchte Gradientennormen je
Ziel vor dem Clipping, was dieses Diagnoseskript bewusst nicht ergänzt hat (Auftrag: nur
messen, nicht eingreifen).

**Fortsetzung (14.08.2026): die beiden Mechanismen getrennt.** Mechanismus (i) und (ii)
oben sind nicht dasselbe Experiment — (ii) lässt sich billig isolieren, ohne den geteilten
Zustand anzufassen: Fall (a) erneut, identischer Datenpool und Modell-Seed je
Wiederholung, aber `gradient_clip = 100.0` statt der Voreinstellung `1.0`
(`scripts/diagnose_soft_sensor_interference.py --gradient-clip 100.0`; bewusst nicht
deaktiviert, um keine Divergenz als neues Störsignal einzuführen). Bei diesem Wert greift
`clip_grad_norm_` in der Praxis nicht mehr, solange die Gradientennorm nicht explodiert —
sie tat es nicht, kein Lauf brach ab oder zeigte `NaN`.

| Seed | Fall (a), `clip=1.0` | Fall (a), `clip=100.0` |
|---|---|---|
| 1 | 0.2476 | 0.2447 |
| 2 | 0.6495 | 0.6637 |
| 3 | 0.3422 | 0.3574 |
| **Median** | **0.3422** | **0.3574** |

Kein Seed bewegt sich in Richtung des Werts aus Fall (b) (0.0238), zwei der drei werden
minimal schlechter statt besser. Das ist die zweite der beiden im Vorfeld genannten
Alternativen: **Mechanismus (ii), das globale `gradient_clip`, ist nicht die Ursache.**
Der geteilte rekurrente Zustand (Mechanismus (i)) bleibt als einziger noch nicht
widerlegter Kandidat übrig — nicht direkt bewiesen (dafür fehlt weiterhin die
Gradientennorm-Messung je Ziel vor dem Clipping), aber der einzige der beiden Kandidaten,
den dieses zweite Experiment nicht ausschließen konnte.

**Das ist keine Codereparatur, sondern eine Architekturentscheidung.** Optionen, mit
Vor-/Nachteilen, keine Vorentscheidung. Die vier Optionen unten stammen aus der ersten
Fassung von Befund 26; die Markierung `[Fortsetzung]` zeigt, wie das Clip-Experiment jede
einzelne einordnet:

1. **Verlustgewichtung pro Zielgröße** (z. B. inverse Aufgaben-Schwierigkeit, Uncertainty
   Weighting nach Kendall et al. 2018, oder GradNorm). *Vorteil:* behält den einen Encoder
   und damit die im Modul-Docstring begründete Prämisse — geteilte Dynamik, ein Trainer für
   alle Kerne. *Nachteil:* neue Hyperparameter, deren Kalibrierung selbst eine offene Frage
   ist; bei zwei praktisch unlösbaren Zielgrößen (Befund 19) ist unklar, ob eine Gewichtung
   das Grundproblem löst oder nur seine Stärke verschiebt. *[Fortsetzung: unverändert
   plausibel — setzt am Verlust an, nicht am Clipping, bleibt also unabhängig vom
   Experiment im Rennen.]*
2. **Getrennte Modelle je Zielgröße im Soft-Sensor-Betrieb** (ein GRU nur für
   `energy_target`, für die Qualitätsgrößen ggf. weiterhin Climatology, da ohnehin
   ungeschlagen, Befund 19). *Vorteil:* genau das, was Fall (b) hier zeigt — einfach und
   nachweislich besser. *Nachteil:* gibt die einzige im Modul-Docstring genannte
   strukturelle Stärke eines Rollout-trainierten Netzes gegenüber NARX auf — dass es die
   Rückführung nicht gemessener Zielgrößen aus dem eigenen Zustand lernen kann. Ob diese
   Stärke im Soft-Sensor-Fall überhaupt greift (die Qualitätsgrößen sind ja gerade nicht
   lernbar), ist nach Befund 19 fraglich, aber nicht durch dieses Experiment beantwortet.
   *[Fortsetzung: gestützt — umgeht Mechanismus (i) vollständig, unabhängig davon, ob er
   allein oder zusammen mit anderen Effekten wirkt; die einzige Option, die das tut, ohne
   selbst ein neues offenes Hyperparameter-Problem zu sein.]*
3. **Getrennte Köpfe mit eigener Gradienten-Normierung** (ein gemeinsamer Encoder, aber
   `clip_grad_norm_` separat je Parametergruppe statt global). *Vorteil:* adressiert gezielt
   Mechanismus (ii) oben, ohne den geteilten Encoder aufzugeben. *Nachteil:* löst vermutlich
   nur einen Teil des Effekts, falls Mechanismus (i) — der gemischte Encoder-Gradient selbst
   — der dominante ist; das wäre eine eigene Messung wert, bevor man sich darauf verlässt.
   *[Fortsetzung: entkräftet als primärer Fix — das Clip-Experiment zeigt, dass Mechanismus
   (ii) am aktuellen Effekt (0.34 gegenüber 0.024) keinen messbaren Anteil hat; eine getrennte
   Normierung würde also am Kernproblem nichts ändern. Bliebe nur als Zusatzmaßnahme relevant,
   falls eine der anderen Optionen den Haupteffekt bereits behoben hat und ein kleinerer
   Restanteil übrig ist — dafür gibt es bisher keinen Hinweis.]*
4. **Zweiphasiges Training** (erst `energy_target` allein, dann warmgestartetes
   Weitertrainieren mit allen drei Zielgrößen bei kleiner Lernrate für den Kopf).
   *Vorteil:* könnte beide Effekte kombinieren, falls die Qualitätsgrößen doch nützliche
   Zustandsinformation beisteuern. *Nachteil:* mehr Trainingsschritte, ein zusätzlicher
   Umschalt-Hyperparameter, Risiko, die gute `energy_target`-Lösung in Phase 2 wieder zu
   verlieren — genau der Effekt, der hier erst diagnostiziert wurde. *[Fortsetzung:
   unverändert plausibel — zielt wie Option 2 auf den geteilten Zustand, nur ohne ihn ganz
   aufzugeben; ob der Warmstart robust gegen Mechanismus (i) ist, bräuchte ein eigenes
   Experiment.]*

**Zwischenstand:** von den vier Optionen adressiert eine (3) nachweislich nicht die
Hauptursache mehr; die anderen drei (1, 2, 4) setzen alle direkt oder indirekt am geteilten
rekurrenten Zustand an und sind durch dieses Experiment weder gestützt noch entkräftet,
weil keine von ihnen implementiert und gemessen wurde — nur Option 3 wurde durch das
Clip-Experiment stellvertretend geprüft. Die Wahl zwischen 1, 2 und 4 bleibt offen und ist
keine, die aus den bisherigen Messungen folgt.

Für Schritt 6, Teil 2, Punkt 1 (Lernkurve mit mehreren Seeds) folgt daraus: eine
Architekturaussage für `energy_target` im Soft-Sensor-Betrieb, die den aktuellen
Drei-Ziel-Trainer verwendet, misst mit, wie stark die Interferenz mit den beiden
unlösbaren Zielgrößen ist — nicht nur, wie gut GRU/LSTM `energy_target` selbst treffen
könnten. Das ist keine Aufforderung, den Trainer vor der Lernkurve zu ändern, sondern eine
Einschränkung der Lesart, bis eine der Optionen 1, 2 oder 4 entschieden ist.

*Nachtrag: diese Einschränkung ist mit der Fortsetzung unten aufgehoben — Option 2 ist
entschieden und umgesetzt.*

**Fortsetzung (14.08.2026, Teil 2): Option 2 entschieden, umgesetzt, bestätigt.** Drei
Gründe für Option 2 gegenüber Verlustgewichtung (1) und dem Zwei-Phasen-Vorschlag (4):

1. **Verlustgewichtung dämpft nur die Gradientenmagnitude.** Sie ändert, wie stark eine
   Zielgröße im Skalarverlust zählt, aber nicht, dass ihr Gradient weiterhin durch
   denselben rekurrenten Zustand läuft wie der von `energy_target` — die Verschränkung in
   der geteilten Repräsentation, der eigentliche Mechanismus aus diesem Befund, bliebe
   bestehen, nur schwächer.
2. **Die klassischen Baselines sind bereits pro Zielgröße entkoppelt, ARX war also nie ein
   fairer Vergleich.** `LinearARXModel` und `RandomFeatureNARX` lösen
   `_ridge_solve` (`(X^T X + \lambda I)^{-1} X^T Y`) mit einer mehrspaltigen
   Zielgrößenmatrix `Y`. Numerisch geprüft, vor der Implementierung: eine gemeinsame
   Lösung über alle drei Spalten und drei getrennte Ein-Spalten-Lösungen unterscheiden
   sich um 1.1 · 10⁻¹⁶ — Maschinenpräzision, kein Rundungsartefakt. Das folgt aus der
   Normalgleichung selbst (`gram = X^T X + λI` ist für alle Spalten identisch, jede Spalte
   der Lösung hängt nur von der zugehörigen Spalte der rechten Seite ab), nicht aus einer
   Eigenschaft dieses Datensatzes. Der bisherige Vergleich GRU/LSTM gegen ARX im
   Soft-Sensor-Betrieb war dadurch strukturell unfair: ARX war nie demselben
   Transferrisiko ausgesetzt wie das gemeinsame Netz.
3. **Keine neuen Hyperparameter** — bei einer Lernkurve, die bei acht Trainingsläufen
   beginnt, wäre jeder zusätzliche Freiheitsgrad (Verlustgewichte, Umschaltzeitpunkt) selbst
   kalibrierungsbedürftig, ohne dass dafür Daten übrig sind.

**Ein Entwurfskonflikt musste vorab geklärt werden.** "Jedes Teilmodell sieht die volle
Historie aller Zielgrößen" stand im Auftrag, kollidierte aber mit "`torch_sequence.py`
bleibt unverändert": der rekurrente Kern hat eine feste Eingangsbreite, geteilt zwischen
Historienkodierung und Rollout-Schleife. Drei Auflösungen standen zur Wahl — nur die
eigene Historie je Teilmodell (reproduziert exakt Fall (b) oben), ein Persistenz-Platzhalter
für fremde Zielgrößen im Rollout (bräuchte eine neue Kanalart in `data/windows.py`), oder
ein gekoppelter Schritt-für-Schritt-Rollout mit `.detach()`-Werten zwischen den Teilmodellen
(bräuchte eine zu `torch_sequence.py` parallele Trainingsschleife). Entscheidung: nur die
eigene Historie je Teilmodell — geringstes Risiko, keine Änderung an `data/windows.py`,
exakt das im Diagnoseexperiment bereits geprüfte Setup.

**Umsetzung.** `src/twinlink_ml/models/composite.py`, neu: `CompositeSequenceModel` hält
`n_targets` unabhängige `TorchSequenceModel`-Instanzen, eine je Zielgröße, Seed
`model_seed + i` (Position `i` in `target_names`). `_SingleTargetView` (ein
`SequenceDataset`-Wrapper) reduziert einen mehrspaltigen Datensatz kanalweise auf eine
Zielgröße — keine Neuberechnung, reine Auswahl auf den bereits standardisierten Arrays,
denn `Standardizer.fit` schätzt Mittelwert/Streuung je Kanal unabhängig von den übrigen
(`data/windows.py`); `scaler.select([i])` ist deshalb bitgleich mit einer Anpassung, die
von Grund auf nur auf diesem einen Kanal geschätzt worden wäre. `torch_sequence.py` bleibt
unverändert. `registry.py`: `gru`/`lstm` lösen jetzt auf
`CompositeGruSequenceModel`/`CompositeLstmSequenceModel` auf; das alte Einzelnetz bleibt
über `torch_sequence.GruSequenceModel`/`LstmSequenceModel` direkt importierbar, nur ohne
Registrierungs-Label — u. a. für den Kostenvergleich unten.

**Neue Tests** (`tests/test_composite.py`, sechs Stück, wie die GRU-Tests nur mit PyTorch
prüfbar): die Kernzusicherung ist
`test_composite_matches_three_independently_fitted_models` — der Verbund reproduziert
exakt drei einzeln aufgerufene `TorchSequenceModel`-Instanzen mit derselben
Seed-Formel; dazu Nullinit-gleich-Persistenz **pro Teilmodell**, keine Zukunftssicht im
Rollout, GRU/LSTM teilen weiterhin denselben Trainer, `n_parameters` als Summe über
Teilmodelle, und ein Regressionstest für die `registry.py`-Verdrahtung selbst. `pytest -q`
grün in beiden Umgebungen: **200 bestanden, 1 übersprungen** mit PyTorch (Conda-Env
`twinlink`), **189 bestanden, 12 übersprungen** ohne (Conda-Env `rl`, `twinlink_ml` nicht
installiert, `PYTHONPATH=src`) — keine Regression.

**Bestätigungsexperiment**, `scripts/diagnose_soft_sensor_interference.py` um Fall (c)
erweitert (`CompositeSequenceModel`, sonst identisch zu Fall (a)), Soft-Sensor,
`n_train = 32`, `h = 24`, Seeds 1–3, gepoolter NRMSE, (a) und (c) in dieselbe
Skriptausführung gelegt, damit die Zeitmessung unten nicht durch unterschiedliche
Maschinenlast zwischen zwei Läufen verzerrt ist:

| Seed | (a) gemeinsam | (b) nur `energy_target` | (c) Composite | ARX |
|---|---|---|---|---|
| 1 | 0.2476 | 0.0238 | 0.0238 | 0.1706 |
| 2 | 0.6495 | 0.0285 | 0.0285 | 0.1737 |
| 3 | 0.3422 | 0.0226 | 0.0226 | 0.1310 |
| **Median** | **0.3422** | **0.0238** | **0.0238** | 0.1706 |

(c) trifft (b) nicht nur "nahe", sondern bis auf die letzte angezeigte Nachkommastelle
exakt — das ist kein zweiter unabhängiger Beleg, sondern die erwartete Konsequenz der
Verdrahtung: das `energy_target`-Teilmodell im Verbund liegt an Position 0 in
`target_names` und bekommt damit exakt denselben Seed (`model_seed + 0`) wie das
eigenständige Modell in Fall (b), auf demselben Datenpool und demselben, auf eine
Zielgröße reduzierten Datensatz. Die exakte Übereinstimmung bestätigt, dass der Verbund
korrekt verdrahtet ist — keine versteckte Kopplung zwischen den Teilmodellen, kein
Leck —, nicht, dass die Architektur unabhängig ein zweites Mal an derselben Zahl
gemessen hätte. Die Erwartung aus dem Auftrag ("nahe an 0.0238") ist damit exakt
getroffen: die Verschränkung im geteilten Zustand ist die Ursache, und getrennte Netze
beseitigen sie vollständig, nicht nur teilweise.

**Kosten, an diesem einen Diagnoselauf gemessen** (`fit()`-Zeit, Median über Seeds,
dieselbe Skriptausführung):

| | (a) gemeinsames Netz | (c) `CompositeSequenceModel` |
|---|---|---|
| Median | 21.82 s | 217.99 s |
| Spanne | 18.36 – 22.25 s | 203.58 – 227.98 s |

**Faktor ≈ 10×, nicht die naiv erwartbaren ≈ 3×** (drei Netze statt eines, jedes mit
schmalerem Kopf — ein Ausgang statt drei — aber derselben `hidden_size = 64`).
Plausibelste, hier **nicht** direkt gemessene Erklärung: Early Stopping (`patience = 20`)
läuft für jedes Teilmodell unabhängig. Für die beiden praktisch unlösbaren Zielgrößen
(Befund 19) dürfte der Validierungsverlust ohne klaren Trend schwanken, was Early
Stopping mal sehr früh (zufälliges Minimum in den ersten Epochen, dann 20 Epochen ohne
Verbesserung) und mal erst nach dem vollen Epochenbudget auslöst — während der
gemeinsame Verlust in Fall (a) alle drei Verläufe zu einer einzigen, gemittelten
Stop-Entscheidung glättet. Eine Bestätigung bräuchte die Epochenzahl je Teilmodell aus
`training_history`; das ist mit der Diagnose aus Punkt 2 dieses Befunds leicht
nachrüstbar, war aber nicht Teil dieses Laufs — hier nur als plausibelste, nicht als
geprüfte Erklärung notiert.

*Nachtrag: gemessen, unten — und die Zahlen sagen das Gegenteil der Vermutung hier.*

**Für Schritt 6, Teil 2:** Punkt 1 (Lernkurve mit mehreren Seeds) kann jetzt mit den
Labels `gru`/`lstm` aus der Registrierung gefahren werden, ohne die
Interferenz-Einschränkung aus dem ersten Teil dieses Befunds — beide Labels liefern seit
dieser Änderung `CompositeSequenceModel`. Die gemessene ~10-fache Trainingszeit gehört in
die Budgetierung des vollen `study_architectures.py`-Laufs (mehrere Seeds, mehrere
Stützstellen, beide Betriebsarten), nicht nur in diesen einen Diagnoselauf.

**Fortsetzung (14.08.2026, Teil 3): drei Nachprüfungen vor dem vollen Lernkurven-Lauf.**

**1. Seed-Ableitung korrigiert.** `CompositeSequenceModel` leitete die Teilmodell-Seeds
zunächst als `model_seed + i` ab (arithmetisch, `i` = Position in `target_names`).
`utils/seeding.py` begründet für `SeedBundle` ausdrücklich den Wechsel zu SHA-256, "damit
benachbarte Master-Seeds keine korrelierten Teilseeds erzeugen" — dasselbe Argument gilt
zwischen benachbarten Zielgrößen-Indizes. Ersetzt durch `_submodel_seed(seed, target)`,
das dieselbe private `_derive`-Funktion aus `utils/seeding.py` wiederverwendet (Präzedenz
für den Wiederverwendungs-Weg: `narx.py` importiert ebenso `_ridge_solve`/`_thin` aus
`baselines.py`), mit dem Zielgrößennamen statt des Index als Salt — stabil, falls sich die
Reihenfolge von `target_names` einmal ändert. Neuer Test
`test_submodel_seed_is_reproducible_and_decorrelated` (kein Torch nötig, reine
Hash-Ableitung) prüft genau die Regression, die `seed + index` wäre: benachbarte
Zielgrößen dürfen keine benachbarten Seeds bekommen. Die
"bitgleich-zu-drei-einzeln-trainierten-Modellen"-Zusicherung gilt mit der neuen Ableitung
unverändert exakt (`test_composite_matches_three_independently_fitted_models`, jetzt mit
`_submodel_seed` statt `seed + index` auf beiden Seiten des Vergleichs). `pytest -q`
danach: **201 bestanden, 1 übersprungen** mit Torch, **190 bestanden, 12 übersprungen**
ohne — keine Regression.

Nebenwirkung der Korrektur: das `energy_target`-Teilmodell im Verbund bekommt jetzt einen
anderen Seed als das eigenständige Modell in Fall (b) (vorher, mit `seed + 0`, zufällig
identisch, weil `energy_target` an Position 0 steht). Die NRMSE-Werte von (c) und (b) sind
seither nicht mehr bitgleich, sondern nur noch nah beieinander — wie es sein soll, siehe
Tabelle unten.

**2. Trainingskosten gemessen, nicht vermutet — Hypothese widerlegt, echter Mechanismus
gefunden.** Vermutung war: die beiden praktisch unlösbaren Zielgrößen
(`product_moisture_kg_per_kg`, `d50_um`) laufen nahe `max_epochs = 200` durch, weil ihre
Validierungskurve ohne Trend pendelt, während `energy_target` früh abbricht.
`training_history` je Teilmodell ausgewertet (`n_train = 32`, Soft-Sensor, Seeds 1–3,
`max_epochs = 200`, `patience = 20`):

| Seed | (a) gemeinsam, Epochen | (c) `energy_target` | (c) `product_moisture` | (c) `d50_um` | (c) Summe |
|---|---|---|---|---|---|
| 1 | 32 | 200 | 39 | 21 | 260 |
| 2 | 22 | 200 | 21 | 21 | 242 |
| 3 | 26 | 192 | 58 | 23 | 273 |

**Das Gegenteil der Vermutung.** Nicht die beiden schweren Zielgrößen laufen durch —
`energy_target` tut es (Median 200 von 200 möglichen Epochen), während
`product_moisture` (Median 39) und `d50_um` (Median 21, auffällig konstant über alle drei
Seeds) sehr früh stoppen. Erklärung, die zu diesen Zahlen passt: für die beiden praktisch
unlösbaren Zielgrößen (Climatology-Niveau, Befund 19) trifft der Validierungsverlust sein
Rauschminimum fast sofort und verbessert sich danach 20 Epochen lang nicht mehr messbar —
Early Stopping greift schnell und zuverlässig. `energy_target` dagegen hat ein echtes,
langsam abklingendes Optimierungsproblem: der Validierungsverlust findet über viele Dutzend
bis knapp 200 Epochen hinweg immer wieder ein neues Minimum, Patience wird praktisch nie
zwanzig Epochen am Stück unterschritten.

**Das erklärt jetzt auch, warum Fall (a) selbst so wenige Epochen läuft** (22–32, dritte
Spalte oben): dessen Early Stopping basiert auf demselben ungewichteten, über alle drei
Zielgrößen gemittelten Validierungsverlust wie der Trainingsverlust (Punkt 1 dieses
Befunds, `_evaluate_loss`/`_weighted_mse`). Sobald zwei von drei Kanälen ihr Rauschminimum
erreicht haben und nur noch pendeln, sieht die **gemittelte** Verlustkurve wie ein Plateau
aus — selbst wenn `energy_target`s eigener Anteil real weiter sinkt. Der gemeinsame
Trainer bricht dadurch strukturell zu früh ab, nicht nur wegen des Gradientenrauschens aus
Teil 1 dieses Befunds, sondern **auch**, weil dieselbe Mittelung die
Model-Selection-Entscheidung verzerrt. Das ist ein dritter, jetzt gemessener Mechanismus
neben (i) geteilter Zustand/Gradient und dem bereits widerlegten (ii) globales
`gradient_clip`.

Die Summen bestätigen, dass Epochenzahl — nicht ein teurerer Aufwand je Epoche im
Verbund — die ~10-fache Trainingszeit erklärt: Summe(c)/Epochen(a) liegt bei 8.1×, 11.0×,
10.5× je Seed, der tatsächlich gemessene Zeitfaktor bei 9.51× (Median). Übereinstimmung
im erwarteten Rahmen.

Wiederholung des Bestätigungsexperiments mit der korrigierten Seed-Ableitung, zur
Vollständigkeit (`n_train = 32`, Soft-Sensor, `h = 24`, Seeds 1–3):

| Seed | (a) gemeinsam | (b) nur `energy_target` | (c) Composite | ARX |
|---|---|---|---|---|
| 1 | 0.2476 | 0.0238 | 0.0230 | 0.1706 |
| 2 | 0.6495 | 0.0285 | 0.0276 | 0.1737 |
| 3 | 0.3422 | 0.0226 | 0.0234 | 0.1310 |
| **Median** | **0.3422** | 0.0238 | **0.0234** | 0.1706 |

Nah an (b), nicht mehr bitgleich (siehe Punkt 1) — die Kernaussage aus Teil 2 bleibt
unverändert: Faktor ~14 gegenüber (a), Faktor ~7 besser als ARX.

**3. `n_parameters`- und `net_*`-Aggregation bestätigt.** Direkt geprüft, nicht nur aus dem
Code gelesen: an einem `hidden_size = 8`-Modell hat jedes Teilnetz 441 Parameter,
`composite.n_parameters` summiert korrekt zu 1323 — identisch mit
`sum(m.n_parameters for m in composite._submodels.values())`, und das Log-Feld
`model/n_parameters` trägt damit die richtige Zahl (`describe()` ruft `self.n_parameters`
erst zum Zeitpunkt des Aufrufs ab, nicht beim Konstruieren). Zum Vergleich: das alte
gemeinsame Netz kommt bei derselben `hidden_size` auf 507 Parameter — der Verbund ist
größer, weil der rekurrente Kern dreimal existiert und nur die drei Köpfe schmaler sind
(ein Ausgang statt drei), nicht weil er ineffizient wäre. Die `net_*`-Felder
(`net_core`, `net_hidden_size`, …) sind unverändert sinnvoll, weil alle drei Teilmodelle
exakt dieselbe `TorchSequenceConfig` teilen (`CompositeSequenceModel` nimmt bewusst nur
eine Config für alle Zielgrößen, siehe Grund 3 der Entscheidung oben) — es gibt keinen
Fall, in dem die einmal geloggten `net_*`-Werte von den tatsächlich verwendeten
abweichen könnten.

**Offene Entscheidung, jetzt mit einer zusätzlichen Zahl:** die ~10-fache Trainingszeit
ist kein Artefakt der Verbund-Architektur, sondern zu einem großen Teil `energy_target`,
das zum ersten Mal die Epochen bekommt, die es bräuchte — im gemeinsamen Netz hat es sie
nie bekommen, weil die gemittelte Early-Stopping-Entscheidung es mit den beiden schweren
Zielgrößen mitgerissen hat. Ob `max_epochs`/`patience` für den Composite-Fall separat
kalibriert werden (z. B. `max_epochs` für die beiden schweren Zielgrößen senken, da sie
ohnehin nach 20–60 Epochen abbrechen, oder `patience` für `energy_target` erhöhen, falls
200 Epochen die Lösung noch nicht ausgereizt haben) oder ob die gemessenen Kosten
hingenommen werden, ist eine Entscheidung, keine Ableitung aus diesen Zahlen — hier bewusst
offengelassen.

**Fortsetzung (14.08.2026, Teil 4): Seed-Empfindlichkeit von GRU/LSTM bei kleinem
`n_train`, INLINE-Betrieb — mit ausdrücklichem Vorbehalt.**

**Beobachtung im Rohlog** (`artifacts/runs/arch_gru.jsonl`, erzeugt **vor** dem
Composite-Fix aus Teil 1–3 dieses Befunds, also mit dem alten gemeinsamen Netz): in der
`INLINE`-Betriebsart liegt "n für 5 %" für `gru`/`lstm` auf `energy_target` bei jeder
Stützstelle (8/16/32/64) am Deckel von 200, während ARX/Persistenz bei 15–90 liegen.
`collect_spreads` getrennt nach `variance/source=joint` und `variance/source=model`
ausgewertet, um zu prüfen, ob die Streuung aus der Datensatz-Realisierung oder aus der
Netzinitialisierung kommt:

| n_train | σ(joint) | n5%(joint) | σ(model) | n5%(model) | σ(model)/σ(joint) |
|---|---|---|---|---|---|
| 8 (gru) | 0.4458 | 200 | 0.0099 | 23 | 0.02 |
| 16 (gru) | 0.3934 | 200 | 0.0075 | 88 | 0.02 |
| 32 (gru) | 0.0093 | 200 | 0.0009 | 3 | 0.10 |
| 64 (gru) | 0.0145 | 200 | 0.0007 | 3 | 0.05 |
| 8 (lstm) | 0.4274 | 200 | 0.0112 | 22 | 0.03 |
| 16 (lstm) | 0.3792 | 200 | 0.0042 | 19 | 0.01 |

**Eindeutig: die `JOINT`-Achse dominiert, nicht `MODEL`.** σ(model) ist überall nur
1–10 % von σ(joint) — reine Initialisierungsstreuung (fixierte Daten, nur `model_seed`
variiert) erklärt die riesigen Werte nicht. Die Rohwerte zeigen, warum: bei `gru @ n=8`
liegen die fünf `JOINT`-Replikate bei `(0.126, 1.111, 0.094, 0.074, 0.978)` — **zwei von
fünf brechen auf Climatology-Niveau (≈ 1.0) zusammen**, die anderen drei sind gut
(0.07–0.13). Dieselben fünf Datensatz-Realisierungen, aber nur `model_seed` variiert
(`MODEL`-Achse): `(0.126, 0.116, 0.108, 0.122, 0.103)` — **kein einziger Ausreißer.** Bei
`n = 16` derselbe Befund (1 von 5 `JOINT`-Replikaten bricht zusammen, 0 von 5 auf der
`MODEL`-Achse); bei `n = 32`/`64` verschwindet das Muster ganz (keine Werte nahe 1.0 mehr
auf keiner Achse). Zum Vergleich, dieselben fünf Datensatz-Realisierungen bei `arx`:
`(0.232, 0.198, 0.237, 0.268, 0.229)` — durchgehend eng, kein Zusammenbruch.

**Das ist also kein "manche Zufalls-Initialisierungen sind schlecht", sondern ein
"manche Datensatz-Realisierungen bringen GRU/LSTM zum katastrophalen Totalausfall, ARX
nicht"** — spezifisch bei kleinem `n_train` (8, 16), verschwindet bei 32/64. Eine direkte
Prüfung der zunächst naheliegenden Erklärung — `d50_um` sei auch inline ein "hartes"
Ko-Ziel (Climatology-Niveau) und ziehe `energy_target` mit herunter — stützt das nicht
unmittelbar: `d50_um`s **eigener** gemessener NRMSE ist inline unauffällig (Persistenz
0.102, ARX 0.075–0.088, GRU 0.091–0.109, ohne den bimodalen Zusammenbruch). Der Verweis
auf "84 % Rauschgrenze" für `d50_um` (so zitiert in `02_prozess_setup.md` und
`05_arbeitsplan_ap1.md` als "Befund 13") ließ sich in der aktuellen Fassung von
`04_stand_ap1.md` nicht auffinden — die Nummerierung beginnt hier bei 16, ein Befund 13
existiert in dieser Datei nicht (mehr). Das ist ein eigenständiger, kleiner
Dokumentationsbefund (vermutlich eine Umnummerierung bei einer früheren Überarbeitung,
die die Querverweise in den anderen beiden Dateien nicht mitgezogen hat) und hier nur
notiert, nicht behoben.

Die naheliegende Verfeinerung der Verbindung zu Befund 26 Teil 3 braucht dafür aber
nicht, dass ein Ko-Ziel **schlecht** abschneidet — nur, dass seine
Validierungsverlustkurve **schnell** ein Plateau erreicht (gut oder schlecht), weil das
gemeinsame, ungewichtete `_evaluate_loss` das gemittelte Signal aller drei Zielgrößen
für die Stop-Entscheidung verwendet. Ob `product_moisture_kg_per_kg` oder `d50_um` bei
gerade den Datensatz-Realisierungen, die hier zusammenbrechen, zufällig sehr schnell
konvergieren und damit die gemeinsame Early-Stopping-Uhr vorzeitig auslösen, ist mit den
Daten in `arch_gru.jsonl` **nicht** prüfbar — die Epochenzahl-Diagnose aus Teil 3 dieses
Befunds existierte zum Zeitpunkt dieses Laufs noch nicht, und `training_history` wird
nicht in JSONL-Records geschrieben.

**Deshalb ausdrücklich kein Befund über GRU/LSTM als Modellklasse:** gemessen wurde mit
dem Modell **vor** der Korrektur aus Befund 26, das nachweislich zwei
Schadmechanismen hatte (verrauschte Gradienten durch geteilten Zustand, Teil 1; verzerrter
Trainingsabbruch durch gemittelten Validierungsverlust, Teil 3). Dass die Streuung hier
eindeutig an der Datensatz-Realisierung hängt und nicht an der Initialisierung, passt zu
einem datenabhängigen Auslöser eines dieser Mechanismen (z. B. eine bestimmte
Trainingsstichprobe lässt ein Ko-Ziel besonders schnell plateauen) — beweist das aber
nicht. Ob die hohe Seed-Empfindlichkeit von GRU/LSTM bei kleinem `n_train` inline eine
intrinsische Eigenschaft dieser Architekturen ist oder teilweise ein Artefakt derselben,
jetzt behobenen Schadmechanismen war, ist **ungeklärt** und mit `CompositeSequenceModel`
neu zu messen, bevor es als Architektureigenschaft in einen Bericht geht. Keine
Codeänderung aus diesem Abschnitt.

**Fortsetzung (14.08.2026, Teil 5, Abschluss): die offene Frage aus Teil 4 beantwortet.**
Exakt dieselben fünf Seed-Bundles aus `arch_gru.jsonl` (`gru`, `INLINE`, `n_train = 8`,
`variance/source = joint` — dieselben `data_seed`/`noise_seed`/`model_seed`-Tripel, nicht
nur dieselben Master-Seeds) mit `CompositeGruSequenceModel` statt des alten gemeinsamen
Netzes wiederholt, sonst nichts geändert:

| Replikat | NRMSE alt (gemeinsam) | NRMSE neu (Composite) | Verdikt alt | Verdikt neu |
|---|---|---|---|---|
| 0 | 0.1258 | 0.0851 | gut | gut |
| 1 | **1.1113** | 0.0639 | **ZUSAMMENBRUCH** | gut |
| 2 | 0.0945 | 0.0717 | gut | gut |
| 3 | 0.0744 | 0.0559 | gut | gut |
| 4 | **0.9781** | 0.0690 | **ZUSAMMENBRUCH** | gut |

**Das Phänomen ist mit dem korrigierten Modell vollständig verschwunden**, nicht nur
abgeschwächt: beide vormals katastrophalen Replikate (1 und 4) fallen auf 0.06–0.07 —
in derselben engen Bandbreite wie die drei vorher schon guten Replikate (0.056–0.085).
Die gesamte Streuung über die fünf Datensatz-Realisierungen schrumpft von einer Spanne
0.074–1.111 (Faktor 15) auf 0.056–0.085 (Faktor 1.5). Das beantwortet die in Teil 4
offengelassene Frage: die hohe `JOINT`-Seed-Empfindlichkeit von GRU bei kleinem
`n_train` inline war **kein** intrinsisches Merkmal der Architektur, sondern ein Artefakt
der in Teil 1 und Teil 3 gefundenen Schadmechanismen des gemeinsamen Trainers — mit
demselben einen Datenpool, derselben Netzbreite, nur ohne geteilten Zustand, verschwindet
der katastrophale Ausfallmodus vollständig.

**Reichweite dieser Aussage, offen benannt:** geprüft wurde ein Ausschnitt (`n_train = 8`,
`energy_target`, 5 Realisierungen) und nicht die volle Lernkurve über alle Stützstellen,
Zielgrößen und beide Betriebsarten. Die vollständige Bestätigung ist weiterhin Aufgabe der
Lernkurve mit `CompositeSequenceModel` aus Schritt 6, Teil 2, Punkt 1 — aber die
Kernbehauptung aus Teil 4 ("ungeklärt, ob intrinsisch oder Artefakt") ist für diesen
Ausschnitt jetzt entschieden: Artefakt. Keine Codeänderung aus diesem Abschnitt.

**Fortsetzung (16.08.2026, Teil 6, vollständige Bestätigung): die ganze Lernkurve, nicht
nur ein Ausschnitt.** Teil 5 hatte die Kernbehauptung nur für `n_train = 8` entschieden und
ausdrücklich offengelassen, ob sich das über die volle Lernkurve hält. Nachgeholt mit dem
genauen Wiederholungslauf von `study_architectures.py --mode both --models persistence
climatology arx narx gru lstm --sizes 8 16 32 64 --seeds 1 2 3 4 5` — derselbe Aufruf, der
`arch_gru.jsonl` erzeugt hatte, jetzt mit `CompositeSequenceModel` hinter den Labels
`gru`/`lstm`, geloggt nach `artifacts/runs/arch_composite.jsonl` (neue Datei, nicht dieselbe
wie beim alten Lauf — sonst hätte der Duplikat-Schutz in `collect_spreads` mitten im Lauf
angeschlagen, weil `gru`/`lstm` jetzt andere Werte liefern als beim alten Lauf mit
denselben Seeds). 440 Records, 0 Duplikate, Modell-Filterung korrekt (`persistence` fehlt
in den 200 `soft_sensor`-Records, wie von der Betriebsart verlangt).

*Nebenbemerkung zur Durchführung:* der Lauf wurde durch einen Akku-Ausfall nach 179 von 440
Records unterbrochen und mit einem eigenen Skript exakt an der Stelle fortgesetzt, an der er
abbrach (`inline`/`model`, Replikat 3/5, `n_train = 16`, fehlend nur `lstm`) — dieselben
Seeds, dieselbe Datenpool-Rekonstruktion über `DataPoolCache`/`spec_for_mode`, an dieselbe
Logdatei angehängt. Vor dem Fortsetzen geprüft: 29 von 30 begonnenen Punkten waren bereits
vollständig (alle 6 bzw. 5 Modelle), genau ein Punkt fehlte nur `lstm` — keine Lücke, kein
Überschreiben bereits geloggter Werte.

**Der Vergleich, `SOFT_SENSOR`, `energy_target`, gepoolter NRMSE gegen ARX, alle vier
Stützstellen:**

| n_train | GRU alt | GRU neu (Composite) | LSTM alt | LSTM neu (Composite) |
|---|---|---|---|---|
| 8 | −103.2 % schlechter | **+76.8 % besser** | −238.8 % schlechter | **+61.5 % besser** |
| 16 | −160.4 % schlechter | **+82.3 % besser** | −283.9 % schlechter | **+79.7 % besser** |
| 32 | −121.0 % schlechter | **+85.0 % besser** | −256.3 % schlechter | **+85.3 % besser** |
| 64 | −68.6 % schlechter | **+87.4 % besser** | −276.4 % schlechter | **+87.4 % besser** |

**Nicht nur näher an ARX — eine vollständige Umkehr, an jeder einzelnen Stützstelle,**
über den gesamten Bereich von `n_train = 8` bis `64`. Die Auflösungsschwellen an diesen
Stützstellen (0.3 % bis 5.4 %) sind dabei so klein gegenüber den beobachteten Differenzen,
dass an der Nachweisbarkeit kein Zweifel besteht. Damit ist die in Teil 5 offengelassene
Reichweite geklärt: der Effekt gilt nicht nur bei `n_train = 8`, sondern über die gesamte
Lernkurve.

**Zwei Nuancen, die zur Ehrlichkeit dieses Befunds dazugehören — mit derselben
Vorher-Nachher-Tabelle wie oben, hier für die beiden Qualitätsgrößen:**

**`INLINE`, `product_moisture_kg_per_kg`, laufweiser Median-NRMSE gegen Persistenz:**

| n_train | GRU alt | GRU neu (Composite) | LSTM alt | LSTM neu (Composite) |
|---|---|---|---|---|
| 8 | +8.4 % (kein Unterschied) | **+20.2 % besser** | +3.8 % (kein Unterschied) | +6.0 % (kein Unterschied) |
| 16 | +60.8 % besser | **+54.7 % besser** | +32.9 % besser | **+59.0 % besser** |
| 32 | +79.0 % besser | **+74.4 % besser** | +71.5 % besser | **+81.6 % besser** |
| 64 | +85.9 % besser | **+84.9 % besser** | +81.1 % besser | **+87.1 % besser** |

**`INLINE`, `d50_um`, laufweiser Median-NRMSE gegen Persistenz:**

| n_train | GRU alt | GRU neu (Composite) | LSTM alt | LSTM neu (Composite) |
|---|---|---|---|---|
| 8 | −6.7 % (kein Unterschied) | +11.5 % (kein Unterschied) | +0.9 % (kein Unterschied) | +11.5 % (kein Unterschied) |
| 16 | +7.9 % (kein Unterschied) | +11.0 % (kein Unterschied) | +4.6 % (kein Unterschied) | +11.3 % (kein Unterschied) |
| 32 | +8.5 % (kein Unterschied) | **+26.3 % besser** | +10.2 % (kein Unterschied) | +11.4 % (kein Unterschied) |
| 64 | +10.5 % (kein Unterschied) | **+36.6 % besser** | +10.4 % (kein Unterschied) | +12.5 % (kein Unterschied) |

**`SOFT_SENSOR`, `product_moisture_kg_per_kg`, gepoolter NRMSE gegen Climatology:**

| n_train | GRU alt | GRU neu (Composite) | LSTM alt | LSTM neu (Composite) |
|---|---|---|---|---|
| 8 | +6.7 % (kein Unterschied) | +8.8 % (kein Unterschied) | +1.8 % (kein Unterschied) | +6.9 % (kein Unterschied) |
| 16 | +6.6 % besser | **+9.2 % besser** | +3.6 % besser | **+5.9 % besser** |
| 32 | +4.3 % besser | **+5.8 % besser** | +3.6 % besser | **+5.5 % besser** |
| 64 | +1.9 % besser | **+5.2 % besser** | +4.7 % besser | **+10.6 % besser** |

**`SOFT_SENSOR`, `d50_um`, gepoolter NRMSE gegen Climatology:**

| n_train | GRU alt | GRU neu (Composite) | LSTM alt | LSTM neu (Composite) |
|---|---|---|---|---|
| 8 | −19.4 % schlechter | −3.5 % schlechter | −6.5 % schlechter | −0.3 % schlechter |
| 16 | −15.0 % schlechter | −10.0 % schlechter | −18.8 % schlechter | −3.0 % schlechter |
| 32 | −6.7 % schlechter | −6.3 % schlechter | −9.4 % schlechter | −6.9 % schlechter |
| 64 | −0.7 % schlechter | **+0.3 % besser** | −2.3 % schlechter | −0.8 % schlechter |

**Lesart:** `INLINE`/`product_moisture` zeigt dieselbe Handschrift wie `energy_target` —
durchgehend besser, an mehr Stützstellen nachweisbar (`n=8` kippt von "kein Unterschied"
auf "nachweisbar besser" bei GRU). `INLINE`/`d50_um` bleibt größtenteils unterhalb der
Auflösungsschwelle, aber dort, wo sie überschritten wird (`n=32`, `n=64`, nur GRU), zeigt
sie in dieselbe Richtung. `SOFT_SENSOR`/`product_moisture` verbessert sich moderat, aber
konsistent an jeder Stützstelle. `SOFT_SENSOR`/`d50_um` ist die Ausnahme: bleibt bei fast
allen Stützstellen "nachweisbar schlechter" als Climatology, alt wie neu — der Composite-Fix
räumt hier nichts auf. Konsistent mit Befund 26 Teil 3: `product_moisture` und `d50_um`
sind die praktisch unlösbaren Ko-Ziele (Befund 19), deren eigene Composite-Teilmodelle nach
wenigen Epochen abbrechen — die Trennung rettet `energy_target` vor ihrem
Gradientenrauschen, macht sie selbst aber nicht lernbar. Dass `product_moisture` trotzdem
moderat profitiert, `d50_um` aber kaum, passt zu der in `02_prozess_setup.md` zitierten,
in dieser Datei aber nicht mehr auffindbaren "84 % Rauschgrenze" für `d50_um` (Befund 26
Teil 4 dokumentiert diese hängende Querverweis-Lücke bereits): wenn `d50_um` tatsächlich zu
einem größeren Anteil an der Rauschgrenze liegt als `product_moisture`, bleibt dort schlicht
weniger lernbares Signal übrig, das eine Architekturänderung heben könnte — das ist an
dieser Stelle aber eine Vermutung, keine an diesen Daten geprüfte Aussage.

**Damit ist Befund 26 vollständig abgeschlossen** — Diagnose (Teil 1), Ausschluss der
Alternativursache (Teil 2), Umsetzung und Bestätigung von Option 2 (Teil 3), Prüfung der
Seed-Ableitung und der wahren Trainingskostenursache (Teil 3, Nachprüfung), Klärung der
offenen Frage zur inline-Seed-Empfindlichkeit (Teil 4–5), und die volle
Lernkurven-Bestätigung (Teil 6). `artifacts/runs/arch_composite.jsonl` ersetzt
`arch_gru.jsonl` als die gültige Referenz für GRU/LSTM in jedem künftigen Bericht; die
Baseline-Zeilen (`persistence`, `climatology`, `arx`, `narx`) in `arch_gru.jsonl` bleiben
davon unberührt gültig.

**Provenienzvorbehalt (nachträglich ergänzt, siehe Befund 33).** `arch_composite.jsonl`
trägt `process/fingerprint = b24c5b728681bca9` -- stimmt mit keiner der drei aktuellen
`conf/process/*.yaml` überein, welche Sorptionsisotherme (Platzhalter oder GAB) den obigen
Zahlen zugrunde lag, ist unbekannt. `git/commit = "unknown"` in allen 440 Records;
`created_unix_s` (15./16.08.2026) liegt vor dem ältesten Commit dieses Repos (`313e455`,
25.08.2026). Nicht als falsch zu lesen, sondern als vor einer Veröffentlichung neu zu
erzeugen.

### 27. Vier fast identische Studien-Schleifen in `protocol.py` auf einen Kern konsolidiert

**Problem.** `run_variance_decomposition`, `run_noise_study`,
`run_validation_policy_comparison` und `run_architecture_comparison` wiederholten
dasselbe Gerüst — Spec/`SeedPlan`/`DataPoolCache`-Aufbau, eine verschachtelte
(Achse × Replikat × `n_train`)-Schleife über `evaluate_point`, `RunRecord`-Sammlung —
viermal, ohne dass eine der vier Docstrings einen Grund dafür nannte. Risiko: eine
Korrektur (z. B. an der Cache-Behandlung) an drei Stellen nachgezogen, an der vierten
vergessen — stilles Auseinanderdriften, nicht durch einen Fehler beim Schreiben, sondern
beim *nicht* Ändern.

**Kern.** Zwei neue private Bausteine in `protocol.py`: `_StudyPoint` (ein Achsenpunkt —
Label, roher Achsenwert, Seed-Quelle, optionale Spec-/Messkonfigurations-/Policy-Overrides,
punktfeste Extra-Parameter) und `_run_grid` (die eine Schleife, parametrisiert über eine
Liste von `_StudyPoint`s und drei Hooks: `extra_fn` für replikatabhängige Zusatzparameter
ohne Seiteneffekt, `on_variant` für Seiteneffekte wie die Rauschgrenzenberechnung,
`on_point` ausschließlich für Fortschrittsereignisse). Alle vier `run_*`-Funktionen sind
jetzt dünne Aufrufe: sie bauen ihre `_StudyPoint`-Liste und reichen sie an `_run_grid`
weiter. Öffentliche Signaturen und Docstrings unverändert — reiner interner Umbau.

**Die Cache-Falle, vor der Umsetzung geklärt.** `_run_grid`s Cache-Wiederverwendung darf
sich nicht auf Objektidentität oder `!=` auf `DataSpec` verlassen: `spec_for_mode` baut bei
jedem Aufruf ein neues Objekt, ein Identitätsvergleich hätte bei *jedem* Punkt einen neuen
`DataPoolCache` ausgelöst statt nur bei echten Moduswechseln — ein Fehler, der in keinem
Test sichtbar geworden wäre, nur in der Laufzeit. Behoben über `_spec_digest`, wertbasiert,
nach demselben Muster wie `_measurement_digest`/`ProcessConfig.fingerprint`. Test vorab:
zwei `spec_for_mode(base, INLINE)`-Aufrufe liefern denselben Fingerprint, `INLINE` und
`SOFT_SENSOR` verschiedene (`test_spec_digest_is_stable_and_distinguishes_observation_mode`).

**Migration, einzeln je Funktion, einfachste zuerst.** Für jede der vier: Migrationskandidat
neben der unveränderten Originalfunktion, Äquivalenztest (identische Argumente und Seeds,
`RunRecord`-Inhalt verglichen — `group`/`params`/`metrics`/`tags`, mit Ausnahme von
`inference_us_per_window`, einer Wanduhr-Messung, die zwischen zwei Läufen nie gleich ist),
grün, dann Austausch, alter Code und Kandidat entfernt, `pytest -q`. Reihenfolge:
`run_validation_policy_comparison` → `run_noise_study` (zusätzlich `on_variant` für die
`NoiseFloor`-Seiteneffekte, mit voller `NoiseFloor`-Objektgleichheit geprüft) →
`run_variance_decomposition` (zusätzlich `extra_fn` für `variance/master_seed` und
`seed_plan.as_parameters()`) → `run_architecture_comparison` (verschachtelte Achse, kein
`cache`-Parameter nach außen, `on_point` über einen Adapter mit stabiler Signatur — geprüft
über den Vergleich der *Aufruffolge*, nicht nur der Anzahl). Ein `mypy`-Lauf nach jeder
Migration deckte einen echten Tippfehler auf: `_StudyPoint.axis_value` ist bewusst `object`
(muss je Studie einen `VarianceSource`, ein Rauschniveau, eine `ValidationPolicy` oder ein
Modus-Quelle-Tupel tragen können), und `floors[point.axis_value]` in der `run_noise_study`-
Migration brauchte deshalb eine explizite `isinstance`-Prüfung, bevor der Wert als
`dict[float, NoiseFloor]`-Schlüssel taugte — ohne `mypy` wäre das erst zur Laufzeit
aufgefallen, und nur mit dem richtigen Testfall.

`run_architecture_comparison` hatte vor diesem Umbau **keinen einzigen Test** in
`test_protocol.py` (nur `scripts/study_architectures.py` rief es auf) — die Migration hat
das mit `test_architecture_comparison_covers_modes_and_sources` dauerhaft geschlossen,
nicht nur für die Dauer der Migration.

**Bestätigung jenseits der Tests.** `study_validation_policy.py` und
`study_architectures.py` (Platzhalter-Config, `n_train_max = 16`, zwei Seeds, fünf
rechenfreie Modelle) je einmal vor und einmal nach dem Umbau gefahren.
Stdout byte-identisch bis auf den geloggten Dateipfad; die 72 bzw. 24 `RunRecord`s aus
den JSONL-Logs beider Läufe programmatisch verglichen (`params`, `metrics` ohne
`inference_us_per_window`, `tags`, `group`) — **0 Abweichungen**. Kein Git-Repository
vorhanden, daher kein `git stash`; die Vorher-Läufe wurden stattdessen vor der ersten
Quelländerung separat gesichert.

**Ergebnis:** `protocol.py` von vier Kopien auf einen Kern plus vier dünne Aufrufe, ohne
beobachtbare Verhaltensänderung. Keine der vier öffentlichen Funktionssignaturen geändert;
kein Aufrufer (Skripte, Tests) musste angepasst werden außer `test_protocol.py` selbst
(neue permanente Tests: die Fingerprint-Prüfung und die Architekturvergleich-Abdeckung).

### 28. Die GRU/LSTM-Lernkurve (Punkt 1 der Liste unten): `energy_target` klar vor ARX, GRU und LSTM bei `n=64` gleichauf

**Nachtrag zu einem bereits gelaufenen Lauf.** `study_architectures.py --mode both
--sizes 8 16 32 64 --seeds 1 2 3 4 5` gegen die Verbundmodelle aus Befund 26 lief bereits
(`artifacts/runs/arch_composite.jsonl`, 440 `RunRecord`s: `persistence` 40×, `climatology`/
`arx`/`narx`/`gru`/`lstm` je 80×), aber die Auswertung stand noch aus -- Punkt 1 der Liste
unten war damit nur zur Hälfte erledigt. Nachgeholt hier, mit `architecture_verdicts`
(`evaluation/resolution.py`) gegen die jeweils passende Referenz (Modul-Docstring von
`study_architectures.py`): `energy_target` gegen den linearen ARX, die beiden
Qualitätsgrößen gegen die parameterfreie Referenz der Betriebsart (`persistence` inline,
`climatology` im Soft-Sensor-Betrieb). Primärmetrik `nrmse_run_median` (`INLINE`) bzw.
gepoolter `nrmse` (`SOFT_SENSOR`), Befund 20.

**Provenienzvorbehalt (nachträglich ergänzt, siehe Befund 33).** `arch_composite.jsonl`
trägt `process/fingerprint = b24c5b728681bca9` -- stimmt mit keiner der drei aktuellen
`conf/process/*.yaml` überein, welche Sorptionsisotherme (Platzhalter oder GAB) den unten
stehenden Zahlen zugrunde lag, ist unbekannt. `git/commit = "unknown"` in allen 440
Records; `created_unix_s` (15./16.08.2026) liegt vor dem ältesten Commit dieses Repos
(`313e455`, 25.08.2026). Nicht als falsch zu lesen, sondern als vor einer Veröffentlichung
neu zu erzeugen.

**`energy_target`, `INLINE`, gegen ARX** (Auflösungsschwelle konstant 9.3 %, aus
`persistence`):

| `n_train` | ARX (Median) | GRU (Median) | LSTM (Median) | GRU-Verdikt | LSTM-Verdikt |
|---|---|---|---|---|---|
| 8 | 0.2319 | 0.0690 | 0.0820 | +70.3 % nachweisbar besser | +64.6 % nachweisbar besser |
| 16 | 0.2009 | 0.0486 | 0.0519 | +75.8 % nachweisbar besser | +74.2 % nachweisbar besser |
| 32 | 0.1902 | 0.0366 | 0.0358 | +80.8 % nachweisbar besser | +81.2 % nachweisbar besser |
| 64 | 0.1820 | 0.0329 | 0.0314 | +81.9 % nachweisbar besser | +82.8 % nachweisbar besser |

Bei jeder Stützstelle und beiden Kernen weit oberhalb der Schwelle -- das ist die
robusteste Aussage in diesem Lauf. GRU gegen LSTM direkt (`reference_label="gru"`):
bei `n=8` ist LSTM nachweisbar schlechter (-18.9 %, Schwelle 9.3 %), ab `n=16` liegt der
Unterschied (-6.8 %, +2.1 %, +4.5 %) durchgehend **unterhalb** der Schwelle -- **kein
nachweisbarer Unterschied zwischen GRU und LSTM ab `n=16`**, auf der Zielgröße, die den
Vergleich überhaupt trägt.

**`product_moisture_kg_per_kg`, `INLINE`, gegen `persistence`** (Referenz-Median
konstant 0.8701, Schwelle konstant 8.4 %):

| `n_train` | GRU (Median) | LSTM (Median) | GRU-Verdikt | LSTM-Verdikt |
|---|---|---|---|---|
| 8 | 0.6947 | 0.8178 | +20.2 % nachweisbar besser | +6.0 % unterhalb der Schwelle |
| 16 | 0.3943 | 0.3568 | +54.7 % nachweisbar besser | +59.0 % nachweisbar besser |
| 32 | 0.2229 | 0.1598 | +74.4 % nachweisbar besser | +81.6 % nachweisbar besser |
| 64 | 0.1314 | 0.1125 | +84.9 % nachweisbar besser | +87.1 % nachweisbar besser |

**`d50_um`, `INLINE`, gegen `persistence`** (Referenz-Median konstant 0.1020, Schwelle
konstant 12.9 %):

| `n_train` | GRU (Median) | LSTM (Median) | GRU-Verdikt | LSTM-Verdikt |
|---|---|---|---|---|
| 8 | 0.0903 | 0.0902 | +11.5 % unterhalb der Schwelle | +11.5 % unterhalb der Schwelle |
| 16 | 0.0908 | 0.0904 | +11.0 % unterhalb der Schwelle | +11.3 % unterhalb der Schwelle |
| 32 | 0.0752 | 0.0904 | +26.3 % nachweisbar besser | +11.4 % unterhalb der Schwelle |
| 64 | 0.0647 | 0.0892 | +36.6 % nachweisbar besser | +12.5 % unterhalb der Schwelle |

Auf `d50_um` lernt GRU erkennbar, LSTM bleibt bei jeder Stützstelle unterhalb der
Auflösungsschwelle -- der einzige der vier Tabellen dieses Befunds, in dem sich GRU und
LSTM klar unterscheiden. Konsistent mit Befund 19 (beide Qualitätsgrößen an der Grenze
des mit klassischen Modellen Erreichbaren): selbst dort, wo ein Kern nachweisbar
gewinnt, bleibt die Verbesserung gegenüber der Persistenz kleiner als bei
`energy_target`.

**`energy_target`, `SOFT_SENSOR`, gegen ARX** (Auflösungsschwelle schrumpft mit
`n_train`, Befund aus der Liste unten, Punkt 6: 5.4 % → 2.4 % → 0.9 % → 0.3 %):

| `n_train` | ARX (Median) | GRU (Median) | LSTM (Median) | GRU-Verdikt | LSTM-Verdikt |
|---|---|---|---|---|---|
| 8 | 0.2000 | 0.0464 | 0.0769 | +76.8 % nachweisbar besser | +61.5 % nachweisbar besser |
| 16 | 0.1773 | 0.0314 | 0.0359 | +82.3 % nachweisbar besser | +79.7 % nachweisbar besser |
| 32 | 0.1672 | 0.0251 | 0.0247 | +85.0 % nachweisbar besser | +85.3 % nachweisbar besser |
| 64 | 0.1677 | 0.0212 | 0.0212 | +87.4 % nachweisbar besser | +87.4 % nachweisbar besser |

Bei `n=64` sind GRU und LSTM auf `nrmse` (vier Nachkommastellen) **identisch** (0.0212).
GRU gegen LSTM direkt: `n=8` LSTM nachweisbar schlechter (-65.7 %, Schwelle 5.4 %),
`n=16` nachweisbar schlechter (-14.3 %, Schwelle 2.4 %), `n=32` nachweisbar **besser**
(+1.6 %, Schwelle 0.9 % -- bei dieser sehr feinen Schwelle reicht ein kleiner Unterschied
schon für ein Verdikt), `n=64` unterhalb der Schwelle (+0.0 %, Schwelle 0.3 %) -- **kein
nachweisbarer Unterschied**. Dasselbe Bild wie inline: bei kleinem `n` ist LSTM klar
zurück, bei realistischer Datenmenge (`n=64`) sind beide Kerne nicht mehr
unterscheidbar.

**Kapazität, zum späteren Vergleich:** `n_parameters` bei `hidden_size=64`,
`num_layers=1`, Verbundmodell (Summe über die drei Teilnetze): GRU 42819, LSTM 57027 --
Faktor 1.33, wie im Docstring von `LstmSequenceModel` vorhergesagt (der Zellzustand ist
ein zweiter additiver Speicher).

**Einordnung.** `energy_target` ist für beide Kerne bei jeder geprüften Stützstelle und
in beiden Betriebsarten nachweisbar besser als ARX -- das Ergebnis, das den
Architekturvergleich für diese Zielgröße trägt, hält über fünf Seeds. GRU und LSTM sind
sich dabei bei `n=64` auf `energy_target` in beiden Betriebsarten nicht mehr
unterscheidbar (Differenz unterhalb der Auflösungsschwelle) -- ein legitimes
Nullresultat, keine offene Frage: LSTMs zweiter Speicher (Faktor 1.33 mehr Parameter)
zahlt sich bei dieser Datenmenge auf dieser Zielgröße nicht messbar aus. Genau diese vier
Tabellen -- `INLINE energy_target`, `INLINE product_moisture`, `INLINE d50`,
`SOFT_SENSOR energy_target` -- und dieselbe Kapazitätskennzahl sind der Maßstab für
Befund 29 (xLSTM).

### 30. Beharrungszustand war ein Zwischenzustand

**Was falsch war.** `steady_state_report` fuhr den Simulator transient in den
vermeintlichen Beharrungszustand (`duration_s = 30 000 s`) und meldete dort
`d50 = 610.1 µm`. Der exakte Wert (geschlossene Zahlenbilanz, Residuum
`dN/dt` relativ 1e-19 bis 5e-19 über mehrere Keimzufuhrwerte geprüft) ist
`750.0 µm` -- eine Abweichung von 18.7 %. Ursache: drei der vier Zustände
(Feststoffmasse, Wassermasse, Betttemperatur; `tau` bis 600 s) waren bei
30 000 s längst eingeschwungen, die Partikelanzahl/`d50` (`tau = 16 000 s`)
dagegen erst bei 1.9`tau` statt der für praktische Konvergenz nötigen ~7`tau`.
Ein reiner Endwertbericht zu einem festen Zeitpunkt kann diesen Unterschied
zwischen den vier Zeitskalen nicht sehen.

**Warum das nicht mit einer größeren `duration_s` zu lösen ist: Kosten.**
Selbst gemessen (Env `twinlink`, `conf/process/greybox_placeholder.yaml`,
Nominalpunkt): der exakte Löser (`steady_state.py`) braucht **231.8 ms**, eine
transiente Integration über 150 000 s (~9.4`tau_slow`, selbst damit noch
1.6e-4 relative Restabweichung auf `d50`, siehe `test_steady_state.py`)
braucht **85.5 s** -- Faktor **≈370**. Um mit reiner Transiente auf 1e-6
relative Genauigkeit für `d50` zu kommen, wären `t/tau ≈ ln(1e6) ≈ 13.8`,
also ≈221 000 s Simulationszeit nötig, mehr als das Anderthalbfache von
150 000 s, für eine Zahl, die der Löser in 232 ms exakt liefert.

**Was NICHT betroffen war: die Zeitkonstanten.** Die Jacobi-Eigenwerte sind im
Zwischenzustand (transient bei 30 000 s, `d50 ≈ 610 µm`) und im exakten
Beharrungszustand (`d50 = 750 µm`) selbst gemessen identisch:

| | bei 30 000 s (Zwischenzustand) | exakter Beharrungszustand |
|---|---|---|
| `tau_1_s` | 9.459665806 | 9.459665806 |
| `tau_2_s` | 84.048751 | 84.048751 |
| `tau_3_s` | 600.0000000 | 600.0000000 |
| `tau_4_s` | 16000.000005 | 16000.000002 |
| Steifigkeit | 1691.39168 | 1691.39168 |

Auf die berichtete Genauigkeit (9.5 / 84 / 600 / 16000 s, Steifigkeit 1691)
identisch -- Befund 3 (Zeitkonstanten) bleibt damit unberührt. Das folgt aus
der Struktur der Jacobi-Matrix, nicht aus Zufall: Feststoffmasse, Wassermasse
und Betttemperatur bilden einen von Partikelanzahl/Durchmesser algebraisch
unabhängigen 3x3-Block (keine der drei Bilanzen enthält `d50` oder `N`); der
vierte Eigenwert ist linear in `N` mit Steigung `-m_prod/m_s`, exakt bestimmt
durch Feststoffmasse und Produktmassenstrom, die beide längst konvergiert
waren.

**Geschlossene Form für `tau_4` -- Nebenergebnis, selbst verifiziert.** Aus
derselben Steigung folgt im Beharrungszustand (`m_prod = m_spray*w_solid +
m_seed`):

    tau_4 = m_s,soll / (m_spray * w_solid + m_seed)

also die Feststoff-Verweilzeit im Bett. Geprüft über vier Konfigurationen:

| Sprührate | `mass_control_time_constant_s` | `tau_3` (Löser) | `tau_4` (Löser) | `tau_4` (geschlossen) |
|---|---|---|---|---|
| 1.5 kg/h | 600 s | 600.0000 | 16000.0000 | 16000.0000 |
| 3.0 kg/h | 600 s | 600.0000 | 8000.0000 | 8000.0000 |
| 0.6 kg/h | 600 s | 600.0000 | 40000.0000 | 40000.0000 |
| 1.5 kg/h | 300 s | 300.0000 | 16000.0000 | 16000.0000 |

`tau_3` folgt exakt der Reglerkonstante, `tau_4` exakt `m_s/m_prod` -- zwei
der vier Zeitkonstanten sind damit keine identifizierten Größen, sondern
algebraische Folgen der Konfiguration, sobald Bettmasse, Sprührate und
Reglerzeit feststehen. Nebenbei erklärt das, warum `tau_4` so auffällig rund
ist. Konsequenz, hier nur notiert und nicht weiterverfolgt: im Batch-Betrieb
ist `m_prod = 0`, also existiert `tau_4` nicht -- die langsame Mode ist keine
Wachstumskinetik, sondern eine Verweilzeit.

**Konsequenz.** Alle bisher berichteten `d50`-Beharrungswerte sind zu
ersetzen. Das Repo wurde vor dieser Änderung nach `610` und nach abgeleiteten
Werten durchsucht: kein Vorkommen von `610` an keiner Stelle; `750` taucht in
`docs/02_prozess_setup.md` (Zeilen 97, 100, 387) und als Kommentar in beiden
`conf/process/*.yaml` auf, dort überall korrekt als Zielkorngröße, nichts zu
ändern. `steady_state_report` selbst hatte vor dieser Korrektur keinen
einzigen Test (weder Unit-Test noch Notebook-Nutzung) -- der 610.1-µm-Wert
konnte deshalb unbemerkt bleiben. Ein Regressionstest hält jetzt `750.0 µm`
fest (`tests/test_diagnostics.py`).

**Modellannahme des Keimstroms, hier festgehalten statt stillschweigend
vorausgesetzt.** Mit der allgemeinen Formel (siehe oben) wird der Pfad
`seed_mass_flow_kg_per_h > 0` in `_rhs` erstmals über den Löser benutzbar.
Der Keimstrom geht dort in Feststoff- und Zahlenbilanz ein, aber nicht in die
Wasser- und Energiebilanz -- das Modell unterstellt implizit **trockene Keime
auf Betttemperatur** (kein Feuchte-, kein Enthalpieterm für diesen Strom).
Der Betrag ist am Nominalpunkt klein (grob 0.1-1 % der Gesamtleistung im
plausiblen Keimzufuhrbereich), aber die Annahme war bisher nirgends
aufgeschrieben. `OperationConfig` sieht weder eine Keimtemperatur noch eine
Keimfeuchte vor -- eine offene Physikfrage, die mangels `docs/06_physik_und_
offene_fragen.md` (Datei existiert im Repo nicht) hier vermerkt ist, bis eine
solche Datei angelegt wird.

### 31. Abluft als Schnittstellengröße

Mit dem exakten Löser sind Massenstrom, Feuchtebeladung und Temperatur der
Abluft erstmals ohne den Umweg über eine transiente Integration verfügbar.
Fünf Betriebspunkte (Nominalpunkt sowie die vier Kombinationen aus
niedriger/hoher Zulufttemperatur mit niedriger/hoher Sprührate, sonst
nominell):

| Betriebspunkt | zulässig | (Sensibel+Latent)/Leistung | `Y_out` [kg/kg] | Taupunkt [°C] |
|---|---|---|---|---|
| nominal | ja | 87.9 % | 0.0171 | 22.5 |
| `T_in` 65 °C, Spray 0.6 kg/h | ja | 84.8 % | 0.0111 | 15.6 |
| `T_in` 65 °C, Spray 2.7 kg/h | **nein** | 86.6 % | 0.0207 | 25.5 |
| `T_in` 115 °C, Spray 0.6 kg/h | ja | 87.3 % | 0.0116 | 16.4 |
| `T_in` 115 °C, Spray 2.7 kg/h | ja | 89.8 % | 0.0253 | 28.8 |

Der als unzulässig markierte Punkt (niedrige Zulufttemperatur, hohe
Sprührate) zeigt eine Partikelfeuchte von 0.397 kg/kg -- weit über der
kritischen Feuchte von 0.020 kg/kg: die Trocknungskapazität reicht bei dieser
Kombination nicht für den Sprühstrom, ein physikalisch sinnvoller Befund des
Zulässigkeitsflags, keine numerische Auffälligkeit. Er ist als Beleg dafür
aufgeführt, dass das Flag greift, nicht als Datenpunkt.

Über die vier zulässigen Punkte (der unzulässige geht nicht in die
Spannweiten ein): der Anteil der Eintragsleistung, der mit der Abluft
(bezogen auf den Umgebungszustand) abgeht, liegt zwischen 84.8 % und 89.8 %;
die Abluftfeuchte zwischen 0.0111 und 0.0253 kg/kg; der Taupunkt zwischen
15.6 °C und 28.8 °C -- durchgehend unter der jeweiligen Betttemperatur.

**Robust gegen die offene Sorptionsisotherme -- selbst nachgerechnet.**
Verglichen Platzhalter (`greybox_placeholder.yaml`) gegen Laborniveau
(`greybox_lab_moisture.yaml`), beide am Nominalpunkt:

| | Platzhalter | Laborniveau |
|---|---|---|
| `X_ss` | 0.00468 kg/kg | 0.04614 kg/kg (Faktor 9.9) |
| `Y_out` | 0.01714 kg/kg | 0.01698 kg/kg (Δ 0.9 %) |
| verdampfter Anteil des eingesprühten Wassers | 99.80 % | 98.02 % |

Die Partikelfeuchte unterscheidet sich um Faktor 9.9, Abluftfeuchte und
Taupunkt praktisch nicht. Grund: im Beharrungszustand verlässt fast das
gesamte eingesprühte Wasser (1.05 kg/h) die Anlage über die Abluft, nicht
über das Produkt -- das Produkt trägt nur rund 2 g/h davon aus (20.8 g/h beim
Laborniveau, immer noch < 2 % des Eintrags). Massenstrom, Feuchtebeladung und
Taupunkt der Abluft hängen damit praktisch nicht an der noch offenen
Sorptionsisotherme (`02_prozess_setup.md`, Prioritätenliste Platz 3) -- im
Unterschied zu jeder Aussage über die Partikelfeuchte selbst. Befund 31 ist
damit das erste Ergebnis im Projekt, das trotz offener Platzhalterphysik
belastbar an den Verbund gehen kann.

**Einordnung.** Massenstrom, Feuchtebeladung und Temperatur der Abluft sind
die Schnittstelle zum Standort-Fließschema (TUHH-A); der Taupunkt entscheidet,
ob an einem kondensierenden Wärmeübertrager latente Wärme zurückgewonnen
werden kann. Am Nominalpunkt liegt der Taupunkt bei 22.45 °C, nur 2.5 K über
`T_amb` (20 °C) -- latente Rückgewinnung am kondensierenden Übertrager fände
dort nur mit sehr kaltem Kühlwasser überhaupt statt.

### 32. `mypy --strict` hat 115 vorbestehende Fehler in 8 Testdateien -- unabhängig von Schritt 1-3b

Beim `mypy --strict`-Lauf in Schritt 4 (Env `twinlink`): **115 Fehler in 8
Dateien** (`test_step6.py`, `test_protocol.py`, `test_pipeline.py`,
`test_experiments.py`, `test_composite.py`, `test_resolution.py`,
`test_greybox_numerics.py`, sowie eine vorbestehende Zeile in
`test_excitation.py`), überwiegend `no-untyped-def` (fehlende
Typannotationen an Testfunktionsargumenten/-fixtures) sowie einzelne
`"object" not callable` und ein `Literal`-Typfehler in
`test_greybox_numerics.py:50`. Keine der Fehlerstellen liegt in den in
Schritt 1-3b geänderten oder neuen Dateien.

**Nachweis (Messung 1, da weder `git` noch ein `.git`-Verzeichnis auf diesem
Rechner vorhanden sind):** Kopie von `src/`+`tests/`, darin die sechs in
Schritt 1-3b geänderten Dateien auf ihren Wortlaut vor dieser Bearbeitung
zurückgesetzt und die drei neuen Dateien entfernt, `mypy --strict` erneut
laufen lassen: **115 Fehler in denselben 8 Dateien** -- die Differenz der
geprüften Quelldateien (43 gegenüber 46) entspricht exakt den drei neuen
Dateien, die 0 zusätzliche Fehler beitragen. Bitgleich mit dem Befund nach
Schritt 1-3b.

**Messung 2, Versionen (Env `twinlink`):**

| Paket | Version | pyproject-Grenze |
|---|---|---|
| numpy | 2.4.6 | `>=2.0` (offen) |
| scipy | 1.17.1 | `>=1.13` (offen) |
| torch | 2.12.1+cpu | `>=2.4` (offen) |
| hypothesis | 6.165.10 | `>=6.100` (offen) |
| pytest | 8.4.2 | `>=8.0,<9` |
| mypy | 1.18.2 | `>=1.11,<1.19` -- innerhalb der Grenze |
| ruff | 0.12.12 | `>=0.6,<0.13` -- innerhalb der Grenze |

mypy und ruff laufen innerhalb der in `pyproject.toml` bewusst gesetzten
Obergrenzen -- kein entwichenes Werkzeug. numpy, scipy, torch und hypothesis
sind nach oben offen. Die Fehlerverteilung (nur Testdateien, `src/`
durchgehend sauber; Hypothesis wird ausschließlich in Tests verwendet) passt
zum Verdacht, dass eine dieser offenen Obergrenzen die Ursache ist -- geprüft
ist das nicht.

`numpy` (2.4.6) und `scipy` (1.17.1) in dieser Umgebung stimmen exakt mit
`version/numpy` und `version/scipy` in `artifacts/runs/arch_composite.jsonl`
überein -- die Grey-Box-Zahlen aus Befund 26/28 sind von dieser Seite nicht
betroffen. `torch` und die Plattform stimmen dagegen **nicht** überein:
`arch_composite.jsonl` trägt `params.platform =
Linux-6.8.0-134-generic-x86_64-with-glibc2.39` und `params.version/torch =
2.13.0+cu130`, diese Umgebung ist Windows mit `torch 2.12.1+cpu`. Die Läufe
hinter Befund 26 und 28 stammen von einer anderen Maschine mit anderer
torch-Version -- geklärt, keine offene Frage, aber festzuhalten, bevor die
dortigen Zahlen an den Verbund gehen. Keine Neurechnung. (Provenienzvorbehalt zu
`arch_composite.jsonl` insgesamt: siehe Befund 33.)

### 33. `git/commit` steht in jedem aufgezeichneten Lauf auf `"unknown"`

`utils/tracking.py::_git_commit()` ruft `git rev-parse HEAD` auf und fängt
`(OSError, subprocess.CalledProcessError)` ab, dann `return "unknown"` --
ohne Warnung oder Log-Eintrag an der Aufrufstelle. In
`artifacts/runs/arch_composite.jsonl` steht `params.git/commit = "unknown"`.
Diese Umgebung hat weder ein `.git`-Verzeichnis noch `git` selbst auf dem
PATH; dasselbe gilt vermutlich für die Maschine, auf der die bisherigen Läufe
erzeugt wurden. Config-Fingerprint und Seed-Bündel sind in den vorhandenen
Läufen vorhanden; der Git-Stand als dritte tragende Reproduzierbarkeits-Zusage
(`03_code_konventionen.md`, Abschnitt Reproduzierbarkeit: "Git-Commit-Hash im
Run-Log") ist in jedem bisher aufgezeichneten Lauf leer, ohne dass es
aufgefallen wäre.

**Nachtrag (diese Sitzung).** Für `artifacts/runs/arch_composite.jsonl` geht die Lücke über
das fehlende `git/commit` hinaus: `process/fingerprint = b24c5b728681bca9` stimmt mit
keiner der drei aktuellen `conf/process/*.yaml`-Dateien überein (bei Δt=25 s wie bei
nativer Abtastzeit geprüft), und `created_unix_s` der Records (15./16.08.2026) liegt vor
dem ältesten Commit dieses Repos (`313e455`, 25.08.2026) -- die erzeugende Codebasis liegt
außerhalb der Git-Historie. Der oben angenommene Trost ("Config-Fingerprint... sind
vorhanden") trägt für diese Datei also nicht: der Fingerprint ist vorhanden, aber nicht
auflösbar. Betrifft mindestens Befund 26, 28, 43(a); welche Sorptionsisotherme
(Platzhalter oder GAB) den dortigen Zahlen zugrunde lag, ist unbekannt. Nicht als falsch
zu lesen, sondern als vor einer Veröffentlichung neu zu erzeugen.

**Nachtrag: Encoding-Fehler in der Diff-Digest-Erfassung.** Die in G4a eingeführte
Diff-Digest-Erfassung (`utils/tracking.py::_git_diff_digest`, `git/diff_digest`) rief
`subprocess.run(["git", "diff", "HEAD"], text=True)` ohne explizites `encoding=` auf. Unter
Windows fiel das auf die Locale-Codepage zurück (cp1252), die an UTF-8-Sonderzeichen in
Doku-Diffs (τ, Δ, hochgestellte Ziffern) mit `UnicodeDecodeError` scheiterte -- der Fehler
passierte dabei in einem internen `subprocess`-Reader-Thread und erreichte den Aufrufer nie
als Exception, `stdout` blieb still `None`. Wirkungslos war das ausschließlich bei dirty
Arbeitsbaum (Vorbedingung für den Aufruf) unter Windows -- also genau im Arbeitszustand,
nie im Prüfzustand eines sauberen Checkouts, und ließ dabei zehn Tests in
`tests/test_protocol.py` fehlschlagen, ohne dass die Ursache im Hauptpfad sichtbar wurde.
Behoben (`encoding="utf-8"`, `errors="replace"` an allen drei `subprocess.run`-Aufrufen des
Moduls; ein still `None` bleibender Digest bei dirty Arbeitsbaum wird jetzt zusätzlich
selbst als eigener Fehlerfall behandelt statt mit einem sauberen Stand verwechselbar zu
sein, s. `DIFF_DIGEST_UNAVAILABLE`/`DiffDigestUnavailableWarning`). Regressionstest
vorhanden (`tests/test_tracking.py`, patcht die Locale-Auflösung direkt statt sich auf
Umgebungsvariablen zu verlassen, damit er auch unter Linux/CI greift).

### 34. Konstante Gleichgewichtsfeuchte durch zustandsabhängige Sorptionsisotherme ersetzt

`X_eq` war eine Konstante (`MaterialConfig.equilibrium_moisture_kg_per_kg`); physikalisch
richtig ist `X_eq(phi_Abluft, T_bed)` aus der Sorptionsisotherme. Neues Modul
`simulation/sorption.py`: ABC `SorptionIsotherm`, `ConstantIsotherm` (Altverhalten exakt)
und `GabIsotherm` (GAB-Gleichung mit Arrhenius-T-Abhängigkeit für `C` und `K`).

**a) Struktureller Platzhalter, nicht kalibriert.** Fünf Parameter
(`X_m=0.020`, `C_ref=25.0`, `K_ref=0.75`, `dH_C=6000 J/mol`, `dH_K=500 J/mol`,
`T_ref=62°C`), verankert an genau einem Punkt: phi=0.5, T=62°C -> `X_eq=0.030 kg/kg`
(Laborwert aus `greybox_lab_moisture.yaml`). `GabIsotherm.status` ist deshalb fest
`"placeholder"`, unabhängig von `provenance.status` der Config. Echte Kalibrierung
braucht die Sorptionsisotherme des tatsächlichen Anwendungspartner-Stoffsystems -- weiterhin offen
(`02_prozess_setup.md`, Prioritätenliste Platz 3).

**b) Sicherheitsraster** (`sorption_safety_grid`), an `greybox_placeholder.yaml` gemessen:
`t_min=-5.0°C`, `t_max=130.0°C`, `phi_max=0.99`, `phi_min=0.000812`. `phi_min` ist
envelope-transformiert, nicht die mathematische Grenze phi->0: dieselbe absolute Feuchte
ergibt bei höherer Temperatur eine deutlich niedrigere relative Feuchte, weil der
Sättigungsdampfdruck stark mit `T` steigt -- die trockenste erreichbare Zuluft, erwärmt auf
`t_max`, ist die tatsächlich konservative Untergrenze, nicht phi->0.

**c) Picard-Iteration und der Kondensationszweig-Fund.** `Y_out` hängt über `eta` von
`phi(Y_out)` ab -- gelöst über gedämpfte Picard-Iteration (alpha=0.7, n=5), aber **nur im
Trocknungszweig**. Nachgewiesener Fehler bei ungedämpfter Unterscheidung: mit Dämpfung auf
beiden Zweigen verletzten 48 von 2000 Stichproben (`test_outlet_humidity_never_exceeds_
saturation`-Szenario) die Sättigungsgrenze `Y_out <= Y_s(T_bed)`, maximales Residuum
4.51e-05; mit der branch-selektiven Korrektur (Dämpfung nur im Trocknungszweig): 0 von
2000. Grund: im Kondensationszweig ist `Y_out = Y_s(T_bed)` bei jeder Iteration exakt und
zustandsunabhängig -- ein konstanter, kein iterativer Fixpunkt, also keine Oszillationsgefahr,
die Dämpfung bräuchte. Konvergenz (Iteration 4 vs. 5) an drei Referenzpunkten: `knick`
3.99e-05, `befund31_unzulaessig` 3.69e-03, `befund31_real` 1.54e-06 -- alle deutlich unter
der 1-%-Schwelle.

**d) k1-basierter lokaler Floor statt globaler Konstante.** `greybox._floor`s zweite
Schranke (`m_w >= X_eq * m_s`) nutzte bisher eine feste, envelope-transformierte Konstante.
Mit zustandsabhängiger Isotherme wäre diese Konstante genau in der heißen/trockenen Ecke
praktisch wirkungslos, dort wo die Picard-Iteration am fragilsten ist (eine monoton in phi
steigende Isotherme geht bei phi->0 selbst gegen null). Ersetzt durch einen pro
Integrationsschritt aus `k1` gewonnenen lokalen Wert (`_local_equilibrium_moisture`,
konsistent in RK4- und Euler-Zweig). Die alte envelope-transformierte Zahl (0.000812) bleibt
als Referenzwert in `test_local_floor_at_nominal_never_below_envelope_reference`, kein
Laufzeit-Fallback mehr.

**e) Randecke in `InitialConditionDesign`.** Gemessen (nicht angenommen), direkt an `_floor`
instrumentiert (echtes Klemmen der Wassermasse-Untergrenze, nicht nur "Feuchte ist nach dem
Schritt gestiegen" -- Letzteres liefert wegen normaler Sprüh-/Verdampfungsdynamik einen
deutlich größeren, irreführenden Wert). Maßgeblich: **1 von 4096** Sobol-Läufen löst den
Floor beim allerersten internen Schritt sichtbar aus (T_bed0=35.75°C, T_amb=33.48°C,
phi_amb=0.822, X_p0=0.03559), gezogen exakt wie `excitation.MixedExcitationDesign.build` es
in der Produktion tut: **ein** kombinierter Sobol-Zug über alle 7 Dimensionen (3
Störgrößen- + 4 Anfangszustandsdimensionen, Spalten `[:3]`/`[3:]`, ein Seed). Punkt liegt nah
am festen Regressionsfall (`test_scattered_cold_start_below_local_equilibrium_is_floored_up`:
T_bed=T_amb=35°C, phi_amb=0.85, X_p0=0.0301).

Ein früherer Bericht dieser Kennzahl nannte 0 von 4096 mit knappstem Punkt bei
T_bed0=36.35°C, T_amb=33.82°C, phi_amb=0.403 (Marge X_eq−X_p0=−0.00588). Nach Sitzungsverlust
zunächst nicht rekonstruierbar; ein verwaistes Scratchpad-Skript aus der abgebrochenen
Sitzung (`%TEMP%\claude\...\scratchpad\diversity_real.py`) wurde nachträglich gefunden und
reproduziert beide Zahlen exakt. Ursache der Diskrepanz: **nicht** Y_in vs. Y_out (beide
Skripte nutzen das Picard-konvergierte `outlet_humidity_kg_per_kg`), sondern die
Sampling-Methodik -- das Skript zog Anfangszustand und Störgrößen über **zwei unabhängige**
Sobol-Folgen (Seed 0 für die 4 Anfangszustandsdimensionen, Seed 1 für `T_amb`/`phi_amb`, `solid_fraction`
fix auf Nominalwert), was nicht der Produktions-Stichprobe entspricht. Der hier berichtete
Wert (1/4096) verwendet dieselbe kombinierte Sobol-Ziehung wie `MixedExcitationDesign.build`
und ist der für die tatsächliche Datenerzeugung repräsentative.

`InitialConditionDesign` bleibt bewusst unverändert: der Corner Case ist real erreichbar
(unter beiden Methodiken, nur an unterschiedlichen Punkten), und die Grenzen des Samplers so
zu verschieben, dass er ihn meidet, würde einen physikalisch legitimen Zustand wegdefinieren
statt ihn -- wie jetzt durch den Floor -- korrekt zu behandeln.

**f) Gradiententests.** An einem floor-inaktiven Punkt (`X_p=0.05`, `T_bed=70°C`):
`torch.autograd.gradcheck` `True`; zentrale Finite Differenzen und Autograd liefern
identisch `0.9528535540` (relative Differenz 2.37e-14). Am floor-aktiven Regressionspunkt
(T_bed=T_amb=35°C, phi_amb=0.85, X_p0=0.0301): Gradient `dm_w_final/dm_w0 = 4.1641e-06`,
endlich (kein NaN/Inf) -- kein exakter FD-Abgleich verlangt, an der Knickstelle von
`maximum()` nicht zu erwarten.

**g) ABC-Entscheidung.** `SorptionIsotherm` ist von Anfang an eine ABC mit zwei
Implementierungen (`ConstantIsotherm`, `GabIsotherm`) -- anders als beim
`SteadyStateSolver` (`03_code_konventionen.md`), wo nur ein Fall vorlag und die
Abstraktion deshalb bewusst unterblieb. Dieselbe Schwelle zieht `excitation.
DisturbanceDesign` bereits über zwei Implementierungen; `SorptionIsotherm` folgt dieser
Konvention, keine Vorratsabstraktion.

Abluft-Schnittstellengrößen (Massenstrom, Feuchtebeladung, Taupunkt) unter der offenen
Isotherme bereits in Befund 31 behandelt, hier nicht wiederholt.

### 35. Δt-Vorstudie wiederholt (GAB-Isotherme, 900-s-Horizont) -- und ein Seed-Zahl-Befund, der über diese Studie hinausgeht

**a) τ1/τ2: was aus der Niveaukorrektur kommt, was aus der GAB-Struktur.** Drei-Punkte-Messung
(`time_constant_report`, Nominalpunkt, `sample_time_s=5 s` in allen drei Configs), um den
Sprung aus Befund 34 zu zerlegen:

| Config | X_eq / X_cr | τ1 [s] | τ2 [s] | τ3 [s] | τ4 [s] | Steifigkeit |
|---|---|---|---|---|---|---|
| `greybox_placeholder` | 0.003 / 0.020 (konstant) | 9.459665806 | 84.048751 | 600.0000000 | 16000.000005 | 1691.39 |
| `greybox_lab_moisture` | 0.030 / 0.200 (konstant) | 31.238404672 | 281.612062236 | 600.000000016 | 16000.000001938 | 512.19 |
| `greybox_gab_placeholder` | GAB, Anker 0.030 / 0.200 | 30.226261928 | 359.927610339 | 600.000000016 | 16000.000001938 | 529.34 |

τ3/τ4 über alle drei Configs bitgleich (Rauschfloor ~1e-10 relativ) -- Befund 30 bestätigt,
kein Fehler. Spannweite (X_cr−X_eq) wächst von alt zu lab exakt um Faktor 10 (0.017→0.170,
nachgerechnet). Zerlegung des Gesamtsprungs alt→GAB in Niveaukorrektur (alt→lab) und
GAB-Eigenanteil (lab→GAB):

| | Gesamtfaktor alt→GAB | davon Niveaukorrektur (alt→lab) | davon GAB-Struktur zusätzlich (lab→GAB) |
|---|---|---|---|
| τ1 | ×3.195 | ×3.302 | ×0.968 |
| τ2 | ×4.282 | ×3.351 | ×1.278 |

**τ1: die Niveaukorrektur erklärt den Sprung praktisch vollständig**, die GAB-Zustandsabhängigkeit
selbst wirkt sogar leicht gegenläufig (×0.968). **τ2 hat dagegen einen echten, eigenständigen
GAB-Anteil** (×1.278, Temperaturabhängigkeit von C(T)/K(T) in der GAB-Gleichung) -- die beiden
Zeitkonstanten dürfen in Bezug auf künftige GAB-Neukalibrierung nicht gleich behandelt werden,
siehe (g).

**b) Versuchsaufbau.** `scripts/study_sample_time.py`, gegen `conf/process/
greybox_gab_placeholder.yaml`, Kandidaten `{5, 15, 25, 60} s` (historische Basis {5,25,60}
ergänzt um 15 s, siehe τ1/Δt unten), Horizont neu auf **900 s** gesetzt (zuvor 600 s) --
primärer Entscheidungshorizont angelehnt an die Viertelstunden-Taktung von
Strommarkt-/Erneuerbaren-Signalen, nicht an eine Sekundenregelung des Bettes. Historie
weiterhin 800 s (unverändert). Nur Baselines (Persistenz, statische Regression, linearer
ARX), kein Torch -- eine Δt-Vorstudie soll nicht von einer Architektur abhängen, und die
Baseline-Kosten bleiben trivial (Median 26.9 s / 9.5 s / 5.8 s / 2.7 s je Seed für
5/15/25/60 s, alle vier Kandidaten × 10 Seeds zusammen 224.6 s gemessene Rechenzeit für den
ersten 5-Seed-Block). **10 Seeds statt 5** -- Begründung in (f). Log:
`artifacts/runs/dt_study_gab_900s.jsonl` (120 Records: 4 Kandidaten × 10 Seeds × 3 Modelle).

τ1/Δt und die aus `time_constant_report` zitierte empfohlene Sequenzlänge je Kandidat:

| Δt/s | τ1_gab/Δt | `recommended_sequence_length_fast` |
|---|---|---|
| 5 | 6.05 | 31 |
| 15 | 2.02 | 11 |
| 25 | 1.21 | 7 |
| 60 | 0.50 | 3 |

**c) `energy_target`: Rollout-Fehler Δt-invariant, bestätigtes bekanntes Muster.** ARX-Horizont-
NRMSE (Median über 10 Seeds) liegt für alle vier Kandidaten eng beieinander: 0.2145 (5 s),
0.1957 (15 s), 0.2121 (25 s), 0.2358 (60 s). Bestätigt Arbeitsplan-Zeile 24 ("Δt fein ist
besser" ist falsch) -- keine offene Frage hier.

**d) `product_moisture_kg_per_kg` (maßgebliche Zielgröße): Δt = 25 s empfohlen.**

| Δt/s | Persistenz h=1 | ARX h=1 | ARX Horizont (Median [Min, Max]) | Kosten/Seed |
|---|---|---|---|---|
| 5 | 0.0241 | 0.0212 | 0.5518 [0.4263, 0.9426] | 26.94 s |
| 15 | 0.0515 | 0.0306 | 0.5803 [0.3433, 0.9270] | 9.48 s |
| **25** | 0.0677 | 0.0296 | **0.4057** [0.3109, 1.1163] | 5.81 s |
| 60 | 0.1558 | 0.0568 | 0.4574 [0.3272, 0.5072] | 2.70 s |

25 s hat den niedrigsten ARX-Horizont-Median sowohl bei n=5 (0.3746) als auch bei n=10
(0.4057) -- robust gegenüber der Seed-Erhöhung, kein Zufallstreffer kleiner Stichprobe.
Deckt sich mit der historischen, aus Rechenkosten gewählten Standard-Abtastzeit
(`05_arbeitsplan_ap1.md`, Zeile 50) -- jetzt unter korrigierter GAB-Physik und 900-s-Horizont
zusätzlich bestätigt, nicht nur unverändert übernommen.

**τ1/Δt ist hier KEIN verlässlicher Prädiktor für Rollout-Performance.** Die Faustregel
"Δt deutlich unter τ1" favorisiert 5 s (τ1/Δt=6.05) klar vor 25 s (τ1/Δt=1.21, nur noch knapp
unter 1). Die tatsächliche Rollout-Performance zeigt das Gegenteil: 5 s hat sowohl einen
höheren Median (0.5518 vs. 0.4057) als auch höhere Rechenkosten (26.94 s vs. 5.81 s je
Seed, Faktor ~4.6) als 25 s. Das erweitert Arbeitsplan-Zeile 24 ("Δt fein ist besser" ist
falsch): bei festem physikalischem Horizont ist feineres Δt nicht nur nicht genauer, es ist
tendenziell auch nicht stabiler (siehe (f) für den Stabilitätsaspekt).

**Δt = 60 s bei `product_moisture` isoliert stark, aber nicht generalisierbar.** Zweitbester
Median (0.4574), niedrigste Streuung (39.3 %) und geringste Rechenkosten (2.70 s/Seed) unter
den vier Kandidaten -- das gilt **nur mit Blick auf `product_moisture`**. Bei `energy_target`
ist Δt=60 s dagegen der schlechteste Kandidat überhaupt, sowohl im Median als auch in der
Streuung (siehe (f)). Keine Empfehlung, 60 s statt 25 s zu wählen -- auch nicht als
günstigeren Rückfallkandidat, weil "günstiger" hier zielgrößenabhängig kippt.

**Δt = 5 s und 15 s: nicht empfohlen** für den Architekturvergleich -- schlechtere Mediane
UND höhere Streuung als 25 s/60 s; 5 s zusätzlich mit deutlich höheren Rechenkosten (Faktor
~4.6 gegenüber 25 s, ~10 gegenüber 60 s).

**e) `d50_um`: fallendes Muster mit gröberem Δt, nicht vertieft.** ARX-Horizont-Median (n=10):
0.1729 (5 s) → 0.0846 (15 s) → 0.0835 (25 s) → 0.0780 (60 s). Plausible, nicht weiter geprüfte
Erklärung: `d50` ändert sich über 900 s real kaum (Beobachtbarkeitsproblem, siehe Arbeitsplan),
bei feinem Δt hat der ARX-Rollout mehr Schritte, über die sich Fehler akkumulieren kann, ohne
dass real mehr Information vorliegt. Keine tiefere Untersuchung nötig -- `d50_um` ist bereits
aus dem Architekturvergleich entfernt (zu 84 % Rauschgrenze, Arbeitsplan Zeile 106).

**f) Eigenständiger Methodenbefund, über diese Studie hinaus relevant: 5 Seeds unterschätzen
die Streuung von Rollout-Metriken bei `n_train=32` systematisch.** Relative Spreizung
((Max−Min)/Median) der `product_moisture`-ARX-Horizont-NRMSE je Kandidat, n=5 gegen n=10:

| Δt/s | Spreizung n=5 | Spreizung n=10 |
|---|---|---|
| 5 | 37.2 % | 93.6 % |
| 15 | 81.9 % | 100.6 % |
| 25 | 33.4 % | 198.6 % |
| 60 | 35.5 % | 39.3 % |

Bei drei von vier Kandidaten verdoppelt bis verdreifacht sich die Spreizung von n=5 auf
n=10, ohne dass sich Median oder die bei n=5 bereits sichtbaren Extremwerte grundlegend
ändern (Kontrollprobe an Δt=15 s: Minimum/Maximum bei n=5 und n=10 identisch, nur der Median
rutscht durch die fünf neuen Seeds -- die Spanne selbst war mit 5 Seeds schon vollständig
erfasst, nur ihr Verhältnis zum Median nicht). Das ist kein Δt=15-spezifischer
Instabilitätsbefund (frühere Zwischenvermutung, mit dem Kontrollversuch gegen 5 s/25 s/60 s
verworfen), sondern eine generelle Eigenschaft der Rollout-Metrik bei diesem
Trainingspool-Umfang.

**Zielgrößenabhängige Stabilität, nicht ein einzelner Δt-Mechanismus.** Bei `product_moisture`
ist Δt=60 s die stabilste Wahl (39.3 % Spreizung bei n=10) -- bei `energy_target` dagegen die
instabilste (104.3 %) UND zugleich die mit dem schlechtesten Median (0.2358 gegenüber 0.1957
bei 15 s). Eine univoke Empfehlung "größeres Δt ist stabiler" wäre falsch; eine frühere
Zwischenhypothese dazu (weniger Rollout-Schritte bei 60 s -> weniger Gelegenheit zur
Fehlerakkumulation, als genereller Mechanismus) ist damit widerlegt, nicht bestätigt. Für die
Δt-Empfehlung zählt daher weiterhin vorrangig `product_moisture` als maßgebliche Zielgröße
(Begründung siehe (d)); `energy_target` dient nur als Kontrollpunkt, an dem Δt=25 s (Median
0.2121, Spreizung 48.4 %) ein vernünftiger Mittelweg ist -- weder bester noch schlechtester
Kandidat -- während Δt=60 s dort schlecht abschneidet.

**Empfehlung:** für Rollout-Horizont-Metriken künftig **10 statt 5 Seeds als Standard**, nicht
nur für diese Studie. `scripts/study_sample_time.py --seeds`-Default entsprechend auf
`1..10` gesetzt.

**g) Vorbehalt.** Ergebnis basiert auf der unkalibrierten GAB-Isotherme (Befund 34). Laut (a)
ist τ1 gegenüber einer künftigen GAB-Neukalibrierung robust (GAB-Eigenanteil nur ×0.968,
der Sprung kommt fast vollständig aus der bereits bekannten Niveaukorrektur) -- τ2 dagegen
NICHT (echter GAB-Eigenanteil ×1.278). Eine TUHH-A-Antwort zur tatsächlichen Sorptionsisotherme
könnte τ2 also nochmals spürbar verschieben; dann ist diese Vorstudie zu wiederholen, nicht
die Architekturen neu zu trainieren.

### 36. Gewichte persistiert, GRU/LSTM auf dem Konstantlauf-Holdout ausgewertet -- Vorteil bleibt qualitativ, schrumpft aber deutlich

**a) Checkpoint-Schema und Round-Trip-Verifikation.** Neues Modul `utils/checkpoints.py`
(`CHECKPOINT_SCHEMA_VERSION = 1`): ein `torch.save`-Dict je `run_uid` unter
`artifacts/checkpoints/<run_uid>.pt` mit `schema_version`, `model_label`, `composite`
(Diskriminator `TorchSequenceModel` vs. `CompositeSequenceModel`), `config`, `target_names`,
`exogenous_names`, `n_exogenous`, `input_scaler`/`target_scaler` (Mittelwert/Skala),
`run_uid`, `process_fingerprint`, `data_spec`, `seed_bundle`, `git_commit`, sowie den
Gewichten selbst (`submodel_state_dicts` bei Composite-Modellen, `state_dict` sonst).

Erster Testversuch unzureichend: `predict()` arbeitet ausschließlich auf standardisierten
Werten -- ein Vergleich der rohen `predict()`-Ausgabe zweier Prozesse hätte einen kaputten
Skalierer (z. B. vertauschte `mean`/`scale`-Werte) nie gezeigt, weil ein falscher Skalierer
die standardisierte Vorhersage gar nicht verändert, nur ihre Rückrechnung in physikalische
Einheiten. Zweiter, schärferer Test: Prozess 1 speichert einen Checkpoint
(`roundtrip-test-0002.pt`) und einen Referenz-Array `_roundtrip_reference.npy`
(Shape `(544, 36, 3)`, physikalische Einheiten über `target_scaler.inverse()`). In dieser
Sitzung in einem dritten, unabhängigen Prozess wiederholt: Datensatz exakt aus den im
Checkpoint gespeicherten `data_spec`/`seed_bundle`/`process_fingerprint` rekonstruiert
(`master_seed=1` durch Rückwärtssuche über `SeedBundle.from_master` identifiziert,
`process_fingerprint=82f65f576a2e5253` bestätigt), dann `load_checkpoint()` + `predict()` +
`target_scaler.inverse()` gegen die gespeicherte Referenz geprüft: `max(|Vorhersage −
Referenz|) = 0.0`, `np.array_equal() = True` -- bitgleich in Ingenieurseinheiten.
Nebenbefund beim Datei-Diff: `roundtrip-test-0001.pt`, aus einem früheren Speicherstand,
fehlt das Feld `exogenous_names` vollständig (`load_checkpoint()` braucht es laut Docstring
nicht zum Laden, aber es belegt, dass sich das Schema während der Entwicklung noch geändert
hat, bevor es den aktuellen Stand erreichte).

**b) Konstantlauf-Holdout-Kernergebnis.** 20-Job-Sweep (GRU/LSTM × 10 Trainings-Seeds,
`scripts/run_arch_constant_holdout_sweep.py`/`_worker.py`),
`artifacts/runs/arch_gab_constant_holdout.jsonl` (100 Records). Auf der angeregten
Testmenge (Reproduktion von Befund 28 unter GAB-Isotherme/900-s-Horizont, Metrik
`nrmse_run_median`, h=36, Median über 10 Seeds):

| Zielgröße | Persistenz | GRU | LSTM | Reduktion GRU | Reduktion LSTM |
|---|---|---|---|---|---|
| `product_moisture_kg_per_kg` | 0.7799 | 0.0376 | 0.0364 | 95.18 % | 95.34 % |
| `energy_target` | 1.0830 | 0.0309 | 0.0301 | 97.15 % | 97.22 % |

Auf dem Konstantlauf-Holdout (`constant_run_fraction=1.0`, ein fixer Pool über alle 10
Trainings-Seeds je Ziehung) schrumpft das deutlich, bleibt aber durchweg positiv -- über
**drei unabhängig gezogene Holdout-Pools bestätigt** (999 aus dem ursprünglichen Sweep,
997/998 neu erzeugt und gegen dieselben 20 bereits gespeicherten Checkpoints ausgewertet,
kein Neutraining: `scripts/run_arch_multipool_eval.py`,
`artifacts/runs/arch_gab_constant_holdout_multipool.jsonl`, 80 Records):

- `product_moisture_kg_per_kg`: 30.5 % (Pool 998, GRU) bis 51.6 % (Pool 997, LSTM)
- `energy_target`: 47.9 % (Pool 999, LSTM) bis 55.1 % (Pool 997, LSTM)

Der qualitative Vorteil gegenüber Persistenz bleibt also bestehen und ist über drei
unabhängige Pools stabil -- die **Magnitude** aus Befund 28 (angeregte Testmenge, ~95-97 %)
überträgt sich aber NICHT direkt auf den realen Konstantlauf-Betrieb (~30-55 %). Für
Verbundberichte heißt das: nicht als "Befund 28 bestätigt" vereinfachen, sondern beide
Betriebsarten und ihre unterschiedliche Größenordnung explizit nennen.

**c) ARX bricht auf dem Konstantlauf-Holdout ein -- Ursache gemischt und
zielgrößenabhängig.** Auf Pool 999 ist ARX um Faktor 5.47 (`product_moisture`: Persistenz
0.5437 vs. ARX 2.9755) bzw. 2.87 (`energy_target`: 0.4668 vs. 1.3417) schlechter als
Persistenz. Verhältnis `nrmse_run_max`/`nrmse_run_median` je Trainings-Seed (32 Testläufe
im Pool), Spanne über die 10 Seeds:

| Zielgröße | ARX | GRU | LSTM |
|---|---|---|---|
| `product_moisture_kg_per_kg` | 34.9-119.8× | 11.1-69.7× | 8.5-36.3× |
| `energy_target` | 9.2-15.1× | 2.0-6.8× | 2.3-6.6× |

Bei `product_moisture` liegen GRU und LSTM in derselben Größenordnung wie ARX -- spricht
für einen gemeinsamen Normierungseffekt bei mindestens einem "ruhigen" Konstantlauf im
festen Pool, der alle drei Modelle trifft (ARX zeigt zusätzlich absolut größere
Spitzenfehler, 137-297 gegenüber 2.4-33.5 bei GRU/LSTM -- der gemeinsame Normierungseffekt
erklärt also nicht den gesamten ARX-Median, nur einen Teil davon). Bei `energy_target`
bleiben GRU/LSTM dagegen deutlich gedämpfter gegenüber ARX -- dort ist eine
ARX-spezifische Instabilität (autoregressive Rekursion, die bei bestimmten konstanten
Stellgrößen-Kombinationen über 36 Schritte aufschaukelt) die plausiblere Erklärung. Keine
der beiden Erklärungen pauschal auf beide Zielgrößen übertragen.

**d) Reale Sweep-Kosten vs. Schätzung.** Der 20-Job-Sweep (8-fache Prozessparallelität)
spannt laut Provenienz der geloggten Records eine Zeitspanne von ~42.5 Minuten (erster bis
letzter `created_unix_s`-Zeitstempel in `arch_gab_constant_holdout.jsonl`; berichtete
Gesamtdauer inklusive Prozessstart-Vorlauf vor dem ersten Log-Eintrag: 42.8 Minuten),
gegenüber ~27-30 Minuten, die aus der Einzelprozess-Thread-Skalierungsmessung geschätzt
worden waren. Reale Mehrprozess-Kontention war also teurer als der isolierte Benchmark
vermuten ließ; Ursache (Netzlaufwerk-E/A -- das Arbeitsverzeichnis liegt auf
`\\nap.rz.tu-harburg.de\...` -- gegenüber CPU-Kontention) nicht weiter untersucht.
Ergänzung zur bestehenden Thread-Skalierungsmessung, kein eigener Befund.

**e) Vorbehalt.** Alle Zahlen dieses Befunds beruhen auf der unkalibrierten GAB-Isotherme
(Befund 34) und Δt=25 s/900-s-Horizont (Befund 35), nicht auf den ursprünglichen
Befund-26/28-Zahlen unter der alten, konstanten Feuchte-Gleichgewichtsannahme. Ein
direkter Zahlenvergleich zu Befund 28 ist deshalb nur grobe Orientierung, keine exakte
Verbesserung/Verschlechterung.

### 37. Transformer-Architektur implementiert (GVB-Zusage TUHH-B-AP1) -- Korrektheit
bestätigt, aber 14-18x teurer als GRU/LSTM und in einem reduzierten Einzeldatenpunkt
schlechter

**a) Implementierung und Korrektheitsprüfungen.** `CompositeTransformerSequenceModel`
(`models/composite.py`), Registry-Slot `transformer` jetzt `implemented=True`. Encoder
über die volle Historie ungemaskiert, Decoder sequenziell/autoregressiv über den
Horizont: bei jedem der 36 Schritte wird die eigene, bereits generierte Ausgabe
(`last`) in die nächste Decoder-Eingabe zurückgeführt, `decoder_embeds` wächst
kumulativ, volle Selbstaufmerksamkeit über die bislang erzeugte Sequenz plus
Kreuz-Attention zur Historie -- keine explizite kausale Maske nötig, weil zu jedem
Zeitpunkt keine "zukünftigen" Decoder-Token existieren (siehe
`torch_sequence._build_transformer_network`). Fabrik-Standard `hidden_size=16,
num_heads=2, num_layers=1` -- bewusst klein, nicht auf GRUs `hidden_size=64` getrimmt:
ein einzelner Encoder-Layer bei `d_model=64` hätte allein rund 50000 Parameter, mehr
als das gesamte Drei-Ziele-GRU-Verbundmodell zusammen. Beim kleinen Standard: 23523
Parameter (Composite, alle drei Teilnetze) gegen 42819 beim GRU-Verbund (Befund 28/36).

Vier Prüfungen bestanden: Störungstest (`torch.equal`, kein Leck aus zukünftigen
Zeitschritten bei `teacher_forcing=0.0`, Gegenprobe mit `1.0` korrekt `False`),
Gradiententest (Autograd gegen finite Differenzen, relative Differenz 3.97e-9),
Round-Trip über zwei echte getrennte Prozesse (bitgleich), Verlust fällt über 15
Epochen ohne NaN/Inf.

**b) Trainingskosten -- real gemessen, nicht geschätzt.** Produktionslauf
(`n_train=32`, `max_epochs=200`, `patience=20`, `torch.set_num_threads(1)`):
Gesamtzeit 6240.03 s (104.0 Minuten). Epochen je Teilnetz: `energy_target` 200/200
(kein Early Stopping), `product_moisture_kg_per_kg` 141/200 (Early Stopping griff),
`d50_um` 200/200 (kein Early Stopping). 11.6 s/Epoche, konsistent über alle drei
Teilnetze (Gegenprobe: 6240.03/541 Gesamtepochen = 11.53 s/Epoche).

Korrekter Vergleichswert ist die reale **Einzelprozess**-Messung des GRU-Trainings aus
einem früheren Arbeitsstrang: 439.86 s bei `n_train=64` (größer als die hier
verwendeten 32). 104.0 Min / 7.33 Min = Faktor **~14.2** -- **nicht** Faktor 50 (ein
Vergleich gegen den 8-fach-parallelisierten Sweep-Durchschnitt aus Befund 36 wäre hier
irreführend, weil parallelisierte Wanduhrzeit und Einzelprozesszeit nicht vergleichbar
sind). Eine zweite, unabhängige Bestätigung liefert Teil (d) unten: im selben Lauf, auf
denselben Daten, unmittelbar nacheinander gemessen, liegt der Faktor bei 14.0-16.4x --
im selben Bereich, ohne die `n_train=32`-gegen-`64`-Asymmetrie der ersten Messung.

**c) Schritt-A-Diagnose: warum so teuer, und ist Inferenz genauso betroffen wie
Training?** Zwei günstige Messungen an einem frisch initialisierten (nicht
trainierten) Netz, kein neues Volltraining (`scripts/diag_transformer_cost.py`).

Fwd+Bwd-Aufschlüsselung (1 Batch, 256 Fenster, ein Teilnetz): Encoder (einmalig pro
Batch) 4.9 ms, Decoder-Loop gesamt (36 Schritte) 196.2 ms -- der Decoder-Loop
dominiert vollständig. Schrittzeit wächst mit der Schrittnummer: Schritt 1 = 2.11 ms,
Schritt 18 = 4.93 ms, Schritt 30 = 8.86 ms, Schritt 36 = 10.30 ms (Faktor 4.89x über 36
Schritte) -- das erwartete Bild für unkachierte Selbstaufmerksamkeit über eine
wachsende Sequenz (Kosten pro Schritt ≈ O(t), kumulativ O(H²)). Volles Fwd+Bwd (1
Batch): 478.6 ms, davon 201.1 ms reiner Forward (Encoder+Decoder-Loop), der Rest
Backward.

**Composite-Sequenzialität -- der eigentliche Befund dieser Diagnose-Runde, nicht nur
eine Randnotiz.** `CompositeSequenceModel.predict()` (identisch bei GRU/LSTM/
Transformer, Befund 26) ruft die drei Teilmodelle **sequenziell** auf, jedes mit
eigenem vollständigen Forward-Durchlauf. Direkt gemessen: ein Teilnetz allein 794.0
µs/Fenster, alle drei sequenziell 2347.6 µs/Fenster (Faktor 2.96x) -- deckt praktisch
den gesamten Unterschied zur Einzelnetz-Hochrechnung aus dem Fwd+Bwd-Test oben ab
(785.6 µs). Die naheliegende Gegenhypothese -- zu kleine Batches in
`RolloutEvaluator.evaluate()` amortisieren den Overhead nicht weg -- wurde geprüft und
verworfen: die tatsächlich verwendete Batchgröße war 1088 Fenster (ein einziger Batch,
`batch_size=4096`), **größer** als die 256 aus dem Fwd+Bwd-Test, und die
Pro-Fenster-Kosten ändern sich zwischen Batch=256 und Batch=1088 nur um ~1% (785.6 vs.
794.0 µs) -- Batching ist bei diesem winzigen `d_model=16` bereits bei 256 Fenstern
gesättigt. **Konsequenz:** mehr Parallelität (z. B. viele gleichzeitig bewertete
Szenarien in einer späteren RL-/Standortoptimierungs-Anwendung) würde den Nachteil
nicht auf einen kleineren Faktor drücken -- die Kosten stecken in zwei strukturell
sequenziellen Größen (36 Python-Ebenen-Decoder-Schritte × 3 sequenzielle
Teilnetz-Aufrufe), die kein Batching amortisiert.

**Für künftige Vergleiche festhalten** (z. B. bei einer späteren Mamba-Implementierung):
der geloggte GRU-Referenzwert (~135 µs/Fenster, `arch_gab_constant_holdout.jsonl`) ist
bereits eine Composite-Summe über drei sequenzielle Teilnetze, **kein**
Einzelnetz-Wert. Ein Vergleich "Einzelnetz X gegen Composite-GRU" wäre nicht mehr
wie-gegen-wie.

Inferenzvergleich (`RolloutEvaluator`, 2-Epochen-Fit nur zur Lauffähigkeit, nicht zur
Güte): Transformer 2394.8 µs/Fenster gegen GRU 111.1-179.7 µs/Fenster (10 Seeds,
Median ~135 µs, `arch_gab_constant_holdout.jsonl`) -- Verhältnis **~17.7x**. Reine
Inferenz (kein Backward) ist damit **mindestens so stark verlangsamt wie das
Training** (~14.2x) -- die Ursache liegt strukturell im Decoder-Loop selbst
(fehlendes KV-Caching), nicht im wiederholten Backward-Pass allein. Relevant auch für
eine spätere Nutzung als schnelles Umweltmodell in RL/Standortoptimierung, nicht nur
fürs Training.

**d) Schritt-B: ein reduzierter Einzeldatenpunkt.** 1 Seed, `n_train=16` (nächst
kleinere Stützstelle der etablierten Lernkurve {8,16,32,64}), Standard-Produktions-
konfiguration **unverändert** (`max_epochs=200`, `patience=20` -- keine Abkürzung nur
für den Transformer, dieselbe Konfiguration wie GRU/LSTM in Befund 28/36). `n_val=32`/
`n_test=32` fest (`n_train_max=64`-Pool, Präfix `n_train=16` daraus, derselbe Aufbau
wie im Konstantlauf-Holdout-Worker). Baselines + GRU + LSTM + Transformer auf
denselben Daten (`scripts/run_transformer_reduced_comparison.py`,
`artifacts/runs/transformer_reduced_comparison.jsonl`). Gesamtzeit 3333.5 s (55.6
Min.) -- weniger als die vorab geschätzten 67.5-74.9 Min., weil `d50_um` beim
Transformer bei `n_train=16` deutlich früher stoppte als bei `n_train=32`.

Trainingszeit je Modell, selber Lauf, unmittelbar nacheinander gemessen: GRU 209.4 s,
LSTM 178.9 s, Transformer 2933.6 s -- Faktor 14.0x (gegen GRU) bzw. 16.4x (gegen
LSTM), bestätigt den Kostenfaktor aus (b) unabhängig, ohne dessen `n_train`-Asymmetrie.

Epochen je Teilnetz:

| Modell | `energy_target` | `product_moisture_kg_per_kg` | `d50_um` |
|---|---|---|---|
| GRU | 200/200 | 172/200 | 103/200 |
| LSTM | 200/200 | 159/200 | 57/200 |
| Transformer | 200/200 | 172/200 | 25/200 |

NRMSE, laufweiser Median (Primärzahl, gegen Messung):

| Modell | `energy_target` h=1 | h=36 | `product_moisture_kg_per_kg` h=1 | h=36 | `d50_um` h=1 | h=36 |
|---|---|---|---|---|---|---|
| Persistenz | 0.2504 | 1.0776 | 0.0721 | 0.7687 | 0.0905 | 0.1153 |
| ARX | 0.0678 | 0.1859 | 0.0695 | 0.9245 | 0.0687 | 0.0888 |
| GRU | 0.0311 | 0.0360 | 0.0238 | 0.1179 | 0.0902 | 0.0949 |
| LSTM | 0.0351 | 0.0410 | 0.0246 | 0.0904 | 0.0903 | 0.0939 |
| Transformer | 0.0747 | 0.0615 | 0.0333 | 0.1749 | 0.0905 | 0.0973 |

**Ausdrücklich als vorläufig gekennzeichnet**, nicht auf derselben Stufe wie die
10-Seed-Aussagen für GRU/LSTM (Befund 28/36): ein einzelner, kostenbewusst reduzierter
Vergleichspunkt (1 Seed, `n_train=16`). Deutet darauf hin, dass der Transformer bei
dieser Kapazität sowohl bei h=1 als auch bei h=36 schlechter abschneidet als GRU/LSTM
-- bei gleichzeitig 14-18x höheren Kosten.

Zwei Beobachtungen entkräften die naheliegende Ausrede "nur zu wenig trainiert":

- Bereits bei h=1 (Ein-Schritt-Fehler, keine Rollout-Fehlerakkumulation über 36
  Schritte) liegt der Transformer hinter GRU/LSTM: `product_moisture_kg_per_kg` 0.0333
  gegen 0.0238 (GRU)/0.0246 (LSTM), `energy_target` 0.0747 gegen 0.0311/0.0351. Ein
  Modell, das "nur mehr Epochen bräuchte", würde man eher am Rollout-Fehler als am
  Ein-Schritt-Fehler erwarten -- letzterer ist hier bereits schlechter.
- Bei `product_moisture_kg_per_kg` liefen Transformer und GRU **exakt gleich viele**
  Epochen (172/200, identisches Early-Stopping-Verhalten) -- trotzdem bleibt der
  Transformer schlechter (h=36: 0.1749 gegen 0.1179). Gleiche Epochenzahl bei
  schlechterem Ergebnis spricht gegen "einfach länger trainieren würde reichen".

Die Kostenlage allein (14-18x, sowohl Training als auch Inferenz) ist bereits
entscheidungsrelevant und rechtfertigt keine weitere Seed-Verfeinerung in dieser
Studie.

**e) Offene, nicht behobene Ineffizienz.** Kein KV-Caching im Decoder-Loop
(`torch_sequence._build_transformer_network`): `self.decoder(decoder_sequence,
memory)` läuft bei jedem der `H` Schritte komplett neu über die wachsende Sequenz,
strukturell O(H²) an Neuberechnung. Die alte Registry-Notiz "quadratisch, aber
unkritisch" bezog sich auf den alten Horizont `H=24` (Befund 23), nicht auf den
aktuellen `H=36` (Befund 35) -- allein das Verhältnis 36²:24² = 2.25:1 erklärt den
gemessenen Faktor nicht, trägt aber bei. In dieser Sitzung bewusst **nicht** behoben,
nur diagnostiziert und dokumentiert -- keine Änderung an der Decoder-Architektur.

**f) Vorbehalt.** Wie Befund 35/36 beruhen alle Zahlen dieses Befunds auf der
unkalibrierten GAB-Isotherme (Befund 34) und Δt=25 s/900-s-Horizont (Befund 35).

### 38. Mamba-Architektur implementiert (selektiver State-Space-Kern, Gu & Dao 2023) --
O(1)-Kerneigenschaft bestätigt, aber drei Stabilitäts-Fix-Zyklen nötig, und im
reduzierten Einzeldatenpunkt bei `product_moisture_kg_per_kg` schlechter als reine
Persistenz

**a) Implementierung und Machbarkeit.** `CompositeMambaSequenceModel`
(`models/composite.py`), Registry-Slot `mamba` jetzt `implemented=True`
(`models/registry.py`). Geprüft, nicht vermutet: `pip install mamba-ssm` (offizielles
Paket) scheitert auf dieser CPU-only-Windows-Umgebung an fehlendem `nvcc`
(CUDA-Compiler) -- das Paket baut eigene CUDA-Kernel für den parallelen Scan und lässt
sich ohne CUDA-Toolkit nicht einmal installieren. Stattdessen reine PyTorch-Rekurrenz
(`models/mamba.py`, `MambaCore`): selektive SSM-Rekursion mit `A = -exp(A_log)`
(S4D-real-Initialisierung), eingabeabhängigem `Δ`/`B`/`C` über `x_proj`/`dt_proj`,
Zustandsupdate `h_t = Ā_t · h_{t-1} + B̄_t · x_t` -- echtes O(1) pro Schritt, kein
paralleler Scan, wie ausdrücklich beauftragt. Anders als der Transformer (der einen
eigenen `_build_transformer_network`-Zweig brauchte) passt die Zelle unverändert in
das bestehende `(input, hidden) -> (output, hidden)`-`cell_type`-Dict-Muster von
`torch_sequence.py`, weil sie wie GRU/LSTM/xLSTM einen fortgeschriebenen Zustand
fester Größe hat.

Fünf Prüfungen bestanden: Skalierungstest (Schrittzeit flach über 68 Schritte,
Verhältnis Schritt68/Schritt1 = 0.69x -- Gegenteil des Transformers, der dort 4.89x
Wachstum zeigte, Befund 37c), Störungstest (`torch.equal`, kein Leck bei
`teacher_forcing=0.0`), Gradiententest (Autograd gegen finite Differenzen, nach vollem
`eps`-Sweep 1e-9 bis 1e-3 sauberes monotones Verhalten, finale Toleranz 1e-4, maximale
relative Differenz 3.5e-7), Round-Trip über zwei echte getrennte Prozesse (bitgleich),
Verlust fällt über 15 Epochen ohne NaN/Inf (kleiner Datenumfang).

**b) Kostendiagnose -- zwei unabhängige Fragen.** (1) O(1) pro Schritt bestätigt: die
Schrittzeit ist über die Sequenz flach bis leicht fallend (Skalierungstest oben) --
strukturell das Gegenteil des Transformer-Decoders (Befund 37c, O(t) pro Schritt,
kumulativ O(H²)). (2) Absolute Kosten: zwei Messungen mit unterschiedlichem Ergebnis,
beide dokumentiert statt nur die günstigere:

- **b1** (GRU/LSTM künstlich ebenfalls einzeln geloopt, fairer reiner
  Pro-Schritt-Vergleich): Mamba/GRU 5.22x, Mamba/LSTM 6.34x.
- **b2** (echter Produktionspfad): Mamba/GRU 6.88x, Mamba/LSTM 9.70x -- schlechter als
  b1. Grund, am Code bestätigt (nicht nur vermutet): GRU/LSTM verarbeiten die
  32-Schritte-Historie im echten Trainerpfad über einen einzigen fusionierten
  `self.core(history)`-Aufruf (`nn.GRU`/`nn.LSTM` intern optimiert), während Mambas
  eigene `forward()` auch dort in einer Python-`for`-Schleife einzeln über die Zeit
  läuft (`models/mamba.py`, `MambaCore.forward`). b2 ist die praktisch relevante Zahl.

Composite-Inferenz (Summe über alle drei Teilnetze): 365.2 µs/Fenster -- deutlich
günstiger als der Transformer (2394.8/2422.3 µs/Fenster, Befund 37c), aber immer noch
2.7x teurer als der geloggte GRU-Referenzwert (~135 µs/Fenster,
`arch_gab_constant_holdout.jsonl`) und 5.2x-6.9x teurer je nach Vergleichsmethode
(b1/b2). Trainingszeit dieses Laufs: 392.4 s (6.5 Min., nur Mamba neu gerechnet;
GRU/LSTM/Transformer/Baselines aus `transformer_reduced_comparison.jsonl`
übernommen, Fingerprint/Split/Seeds vor Übernahme per Assertion verifiziert, siehe
`scripts/run_mamba_reduced_comparison.py`).

**c) Stabilitätsproblem -- in drei Stufen diagnostiziert und behoben.**

*Stufe 1 (erster Produktionslauf, `n_train=16`, Standardkonfig
`max_epochs=200`/`patience=20`):* Teilnetz `energy_target` zeigt `val_loss=NaN` ab
Epoche 12, Abbruch bei Epoche 31 (`best_state` von Epoche 11). Erste Hypothese
(`Δ` unbeschränkt) **getestet und verworfen**: ein `Δ`-Clamp (`max=50`) behebt das NaN
nicht, exakt derselbe Kollaps bei Epoche 12. Tiefere Diagnose zeigte: `x_t` selbst (der
rohe Zelleneingang, identisch mit `last` aus der autoregressiven Rückführung in
`torch_sequence.py`) explodiert bereits **vor** jeder Mamba-internen Rechnung --
Verzehnfachung über sechs Schritte (2 → 3.6 → 7 → 86 → 137838 → 2e19). Architektonischer
Befund: GRU/LSTM haben sättigende Nichtlinearitäten (tanh/sigmoid), die den Zellzustand
strukturell begrenzen -- die Mamba-Zelle (reine lineare Projektionen, `exp()` nur auf
den Zerfallsfaktor) hat nichts, was einen einmal leicht divergierenden Rollout-Schritt
abfängt.

*Stufe 2 (RMSNorm-Fix):* eine Normalisierung (`RMSNorm`) vor `in_proj` ergänzt --
fehlte in der ursprünglichen Implementierung, ist aber in der Referenzarchitektur immer
vorhanden (Pre-Norm-Residualblock, `x_out = x + Mixer(RMSNorm(x))`), analog zu den in
`nn.TransformerDecoderLayer` eingebauten `LayerNorm`-Schichten (die vermutlich
erklären, warum der Transformer nie dieselbe Instabilität zeigte, Befund 37). Ergebnis:
NaN behoben. `energy_target` trainiert jetzt vollständig durch (200/200 statt 32/200).

*Stufe 3 (Weight-Decay-Ausschluss für `A_log`/`D`):* am echten Referenzcode
(`state-spaces/mamba`, `mamba_simple.py`) verifiziert: `self.A_log._no_weight_decay =
True`, `self.D._no_weight_decay = True`. Generischer (nicht Mamba-spezifisch
benannter) Param-Groups-Mechanismus in `TorchSequenceModel.fit()`/`_param_groups()`
ergänzt: Parameter mit diesem Attribut-Marker bekommen `weight_decay=0`, alle anderen
den konfigurierten Wert -- bestätigt wirkungslos für GRU/LSTM/xLSTM/Transformer
(23/23 `test_composite.py`-Tests weiterhin grün). Ergebnis: **kein messbarer Effekt**
(Diff der beiden Läufe zeigt nur Gleitkomma-Rauschen in der 5. Nachkommastelle bei 7
von 263 Epochen-Einträgen, Epochenzahlen und NRMSE-Werte auf 4 Nachkommastellen
bitgleich). Erklärung: bei `weight_decay=1e-4` und AdamWs entkoppeltem Decay beträgt
der Faktor pro Gradientenschritt `1 - lr·weight_decay = 1 - 1e-3·1e-4 = 1 - 1e-7`. Über
~1800 Gradientenschritte (9 Batches × 200 Epochen) kumuliert das auf nur
`(1-1e-7)^1800 ≈ 0.99982` -- drei Größenordnungen zu klein, um bei `n_train=16`
messbar zu werden. Referenzkonvention korrekt übernommen (könnte
bei größerem `n_train` relevant werden), aber **nicht** die Erklärung der beobachteten
Schwäche aus Punkt (d). Direkt nachgeprüft (kein neues Training, nur Netzaufbau +
Optimizer-Konstruktion): die Param-Groups-Aufteilung tut tatsächlich, was sie soll --
`A_log`/`D` liegen isoliert in einer Gruppe mit `weight_decay=0.0`, alle übrigen
Mamba-Parameter (`in_proj`, `x_proj`, `dt_proj`, `out_proj`, `RMSNorm`-Gewicht) plus der
gemeinsame Kopf in der Gruppe mit dem konfigurierten Wert.

**d) Finale Schritt-2-Ergebnisse** (7-Modelle-Vergleich, 1 Seed, `n_train=16`,
Standard-Produktionskonfiguration; persistence/static/arx/gru/lstm/transformer
unverändert aus `artifacts/runs/transformer_reduced_comparison.jsonl` übernommen).

Epochen je Teilnetz (Mamba, nach RMSNorm+Weight-Decay-Fix, identisch zu RMSNorm-only):
`energy_target` 200/200, `product_moisture_kg_per_kg` 21/200, `d50_um` 42/200.

NRMSE, h=1/h=36 (Mamba, verifiziert gegen `artifacts/runs/mamba_reduced_comparison.jsonl`):
`energy_target` 0.1885/0.2262, `product_moisture_kg_per_kg` 0.0698/0.8004, `d50_um`
0.0903/0.0965.

| Modell | `product_moisture_kg_per_kg` h=36 | `energy_target` h=36 |
|---|---|---|
| Persistenz | 0.7687 | 1.0776 |
| ARX | 0.9245 | 0.1859 |
| GRU | 0.1179 | 0.0360 |
| LSTM | 0.0904 | 0.0410 |
| Transformer | 0.1749 | 0.0615 |
| Mamba | **0.8004** | 0.2262 |

**Unbequemer Kernbefund, nicht beschönigt:** Mamba ist bei `product_moisture_kg_per_kg`
(h=36) **schlechter als reine Persistenz** (0.8004 gegen 0.7687) -- und deutlich
schlechter als GRU/LSTM/Transformer. Bei `energy_target` (h=36) liegt Mamba bei 0.2262
-- besser als vor den Fixes (0.8727), aber immer noch schlechter als GRU
(0.0360)/LSTM (0.0410)/Transformer (0.0615).

**e) Diagnose der zwei schwachen Teilnetze** (Epochenverläufe geprüft).
`product_moisture_kg_per_kg` (21 Epochen): `train_loss` fällt glatt und ungebremst
weiter (-69 % über alle 21 Epochen, keine Abflachung) -- `val_loss` ist dagegen
zufällig bei Epoche 0 am besten und wird nie wieder unterboten. Diagnose: verrauschter,
verfrühter Early-Stopping-Abbruch, kein Zeichen einer echten Kapazitätsgrenze --
konsistent mit der bereits etablierten Lehre aus Befund 35(f) (kleine Stichproben
unterschätzen Streuung). `d50_um` (42 Epochen): `train_loss` flacht bereits ab Epoche
~9 fast vollständig ab (-6 % über die restlichen 32 Epochen). Diagnose: echte
Konvergenz an eine schwache Grenze, kein Abbruchartefakt -- konsistent mit der
projektweit bekannten Rauschgrenze von `d50_um` (bereits bei GRU/LSTM/Transformer
ähnlich schwach, 0.090-0.097).

**Offener Vorbehalt, ausdrücklich:** die `product_moisture`-Diagnose (verrauschter
Abbruch) wurde nur für Mamba in dieser Tiefe geprüft -- sie könnte genauso für die
bereits dokumentierten GRU/LSTM/Transformer-Ergebnisse bei `n_train=16` gelten, deren
Epoche-für-Epoche-Verläufe nie in gleicher Tiefe untersucht wurden. Das ist eine offene
Einschränkung des gesamten `n_train=16`-Einzeldatenpunkt-Vergleichs, nicht nur ein
Mamba-spezifisches Problem.

**f) Harte Grenze erreicht.** Drei Fix-und-Retest-Zyklen (Delta-Clamp verworfen,
RMSNorm erfolgreich, Weight-Decay-Ausschluss ohne messbaren Effekt) sind abgeschlossen.
Kein weiterer Tuning-Versuch (keine Lernraten-Änderung, kein weiterer
Normalisierungsort, keine Architekturvariante, keine weitere Seed-Verfeinerung) --
das Ergebnis wird dokumentiert wie es ist, inklusive des unbequemen Fakts aus (d).

**g) Vorbehalt.** Wie Befund 35/36/37 beruhen alle Zahlen dieses Befunds auf der
unkalibrierten GAB-Isotherme (Befund 34) und Δt=25 s/900-s-Horizont (Befund 35).

### 39. Splitvergleich (fenster-/punktweise gegen laufweise, Kräuter-Reproduktion) --
validiert nachträglich die projektweite Splitkonvention

Antwort auf Punkt 5 der Liste unten und auf den Vorschlag aus dem Abschnitt "Was das
Paper des Vorgängers ändert": Kräuter, Zhou & Heinrich (ABS23) erklären ihr berichtetes
R² = 0.994 zu einem erheblichen Teil methodisch (siehe dort). Dieser Befund misst den
Abstand statt ihn nur zu behaupten -- auf **demselben** Datensatz, mit **demselben**
Modell, nur der Split unterscheidet sich.

**a) Was übernommen wurde, was nicht.** Vier methodische Besonderheiten aus Kräuters
Aufbau, siehe "Was das Paper des Vorgängers ändert":

1. **Aufteilung zufällig über Beobachtungen statt Chargen** -- übernommen, das ist der
   ganze Gegenstand dieses Befunds (`data/point_split.py`, `PointSplitter`).
2. **Ein-Schritt-Aufgabe** (Zeitinformation über erste/zweite Ableitung, zwei
   Zeitschritte) -- übernommen (`models/extra_trees.py`, `history_length=3, horizon=1`,
   kein autoregressiver Rollout).
3. **Zentrierte Savitzky-Golay-Glättung der Zielgröße** -- **nicht** übernommen: nicht
   kausal (benutzt Zukunft), und das Sensormodell dieses Projekts
   (`simulation/greybox.py`, `SensorModel.measure`) ist bereits ein rein kausales
   PT1-Verzögerungsglied plus additives Rauschen ohne eigene Glättungsstufe -- eine
   zentrierte Glättung obendrauf hätte in dieser Pipeline keine sinnvolle Entsprechung.
4. **17-34 % Ausreißerentfernung je Charge** -- **nicht** übernommen: kein
   Sensorartefakt in der Simulation, dort ist jeder Messwert bereits das Ergebnis des
   oben genannten Sensormodells, nicht einer separaten Fehlerquelle.

PCA auf 12 Hauptkomponenten (Kräuters Antwort auf 74 Rohmerkmale) ebenfalls nicht
übernommen -- der hiesige Merkmalsraum (Kanäle × 3) ist von vornherein klein.

**b) Aufbau.** Beide Arme verwenden exakt denselben 96-Läufe-Pool
(`data.split.train + data.split.test`, kein `val`-Block -- Extra Trees hat kein Early
Stopping), dieselbe Fenstergeometrie (`history_length=3, horizon=1, stride=1` -- **nicht**
die 32-Schritte-Geometrie von GRU/LSTM/Transformer/Mamba, `stride=1` bewusst dicht, weil
gerade die überlappende Fensterung Kräuters Leck erzeugt) und dasselbe Modell
(`ExtraTreesModel`, sklearn-Standardwerte, 100 Bäume). Arm A (laufweise) nutzt die
bestehende `IndexBlockSplitter`/`build_datasets`-Maschinerie, Arm B (fenster-/punktweise)
`PointSplitter`/`windowed_point_split` im Verhältnis 64:32 Läufe entsprechend 2:1 --
dieselbe Proportion wie Arm A, nur auf Fenster- statt Lauf-Ebene angewendet. Fünf Seeds
(1-5, jeweils Daten-, Rausch- und Modell-Seed gemeinsam variiert), `scripts/run_extra_trees_split_comparison.py`.

**c) Robustheit -- zwei getrennte Prüfungen, keine davon nur behauptet.**
Trainings-Fensteranzahl beider Arme liegt über alle fünf Seeds innerhalb ±0.25 %
auseinander (73 536 laufweise, 73 414-73 706 punktweise je nach Seed) -- der gemessene
Unterschied ist damit sauber dem Splitmechanismus zuzuschreiben, keiner
Stichprobengrößen-Asymmetrie. Zusätzlich sind die drei Fenstermengen innerhalb des
punktweisen Arms nachweislich disjunkt und der Skalierer in beiden Armen nachweislich
identisch zwischen Train- und Testdatensatz (Identitätsprüfung, nicht nur Gleichheit --
`test_a.target_scaler is train_a.target_scaler` und symmetrisch für Arm B, direkt im
Treiberskript, siehe dort sowie `tests/test_point_split.py`).

**d) Ergebnis** (Median über 5 Seeds, `nrmse_run_median`, h=1):

| Zielgröße | laufweise | punktweise | Verhältnis (punktweise/laufweise) |
|---|---|---|---|
| `energy_target` | 0.366 | 0.261 | 0.71 |
| `product_moisture_kg_per_kg` | 0.262 | 0.084 | 0.32 |
| `d50_um` | 0.165 | 0.080 | 0.48 |

Der punktweise Split sieht in allen drei Zielgrößen systematisch besser aus als der
laufweise -- über den vollen Seed-Bereich, nicht nur im Median (`energy_target`
laufweise 0.296-0.446 gegen punktweise 0.242-0.264; `product_moisture_kg_per_kg`
laufweise 0.151-0.395 gegen punktweise 0.063-0.129; `d50_um` laufweise 0.116-0.176 gegen
punktweise 0.074-0.087). Bei allen drei Zielgrößen überschneiden sich die
Wertebereiche der fünf Seeds zwischen den Armen nicht (Maximum des punktweisen Arms
liegt jeweils unter dem Minimum des laufweisen) -- kein Einzelseed-Artefakt.

**e) Beobachtung, als Hypothese formuliert.** Der Effekt fällt bei der schwerer
vorherzusagenden Zielgröße `product_moisture_kg_per_kg` (Faktor 0.32) deutlich stärker
aus als bei der leichter vorherzusagenden `energy_target` (Faktor 0.71, die laut
`models/baselines.py`, `StaticRegressionModel`-Docstring, Befund 2, zu 98.7 % statisch
erklärbar ist -- **Nachtrag SA2/Befund 59b: 98.7 % ist die historische, laut
`02_prozess_setup.md` nicht reproduzierbare Zahl; die aktuell reproduzierte liegt bei
97.2 %. Beide sagen nichts darüber, wie groß die verbleibende Lücke in physikalischen
Einheiten ist -- Befund 59b zeigt, dass ein trainiertes Modell dort trotzdem Faktor
5-8 besser ist als die statische Regression. Die "gedächtnislos"-Hypothese unten betrifft
eine andere Eigenschaft (Splitsensitivität) und bleibt davon unberührt.**). **Nicht
bewiesen, aber naheliegend:** `energy_target` ist nahezu
gedächtnislos -- ein Modell, das die aktuellen Stellgrößen kennt, braucht kaum Kontext
aus benachbarten Fenstern desselben Laufs, weshalb ein punktweiser Split dort wenig
zusätzliche Information gegenüber einem laufweisen preisgibt. `product_moisture_kg_per_kg`
dagegen ist stark laufabhängig und langsam (Zeitkonstante ~370 s nach der
GAB-Korrektur, Befund 34) -- schon die deutliche Diskrepanz zwischen gepoolter und
laufweiser NRMSE-Normierung dieser Größe (`evaluation/evaluator.py`-Moduldocstring:
0.043 gepoolt gegen 1.354 pro Lauf, gegen 0.133/0.154 bei `energy_target`) zeigt, dass
ihr Fehler stark vom Laufniveau abhängt. Ein punktweiser Split lässt für jeden Testpunkt
Nachbarfenster **desselben Laufs** im Training -- das Modell bekommt damit implizit das
Laufniveau geschenkt, das ein laufweise gehaltener Testlauf nicht hat. Diese Erklärung
ist plausibel und mit bereits dokumentierten Fakten konsistent, aber an dieser Stelle
nicht direkt geprüft (keine Zerlegung des Fehlers nach Niveau- gegen Formanteil
durchgeführt) -- als Hypothese markiert, nicht als Mechanismus behauptet.

**f) Rückbezug -- keine neue Erkenntnis für die bisherigen Architekturvergleiche, sondern
deren nachträgliche Rechtfertigung.** Dieser Befund ändert nichts an Befund 26-38: alle
GRU-/LSTM-/xLSTM-/Transformer-/Mamba-Vergleiche in diesem Projekt haben durchgehend
laufweise gesplittet (`IndexBlockSplitter`/`HashRunSplitter`, nie fenster- oder
punktweise). Was dieser Befund liefert, ist die konkrete Zahl dafür, wie stark die
NRMSE-Werte dieser Vergleiche ausgesehen hätten, wäre stattdessen fensterweise
gesplittet worden -- bei `product_moisture_kg_per_kg` um Faktor ~3, bei `d50_um` um
Faktor ~2, bei `energy_target` um Faktor ~1.4 optimistischer. Die von Anfang an
getroffene Entscheidung für laufweise Splits (Modul-Docstring `data/splits.py`: "Läufe
zuordnen ... nie auf Fensterebene") war also nicht nur methodisch vorsichtig, sondern
verhindert nachweislich genau die Größenordnung an optimistischer Verzerrung, die
Kräuters eigene Zahl erklärt.

**g) Vorbehalt.** Wie Befund 34-38 beruhen alle Zahlen dieses Befunds auf der
unkalibrierten GAB-Isotherme (Befund 34).

### 40. Mamba bei voller Stichprobe (`n_train=64`, 10 Seeds, beide Testmengen) --
Befund 38 nicht entkräftet, beim Konstantlauf sogar verschärft

Vollwertige Auswertung der bereits in Befund 38 stabilisierten Mamba-Konfiguration
(`hidden_size=16`, `state_size=16`, `num_layers=1`, RMSNorm-Fix + Weight-Decay-Ausschluss
für `A_log`/`D`, unverändert) -- derselbe Maßstab wie GRU/LSTM in Befund 36
(`n_train=64`, 10 Seeds, angeregte Testmenge UND Konstantlauf-Holdout,
`HOLDOUT_SEED=999`), statt der kostensparenden Einzelseed-Messung aus Befund 38
(`n_train=16`, 1 Seed, nur angeregt). GRU/LSTM/Persistenz/static/ARX wurden **nicht**
neu gerechnet -- die 100 bereits vorliegenden Records aus
`artifacts/runs/arch_gab_constant_holdout.jsonl` (Befund 36) wurden per
Fingerprint/Split/Seed-Assertion je Seed gegen die neu generierten Mamba-Daten verifiziert
(bitgleich, alle 10 Seeds) und direkt wiederverwendet. Neue Records:
`artifacts/runs/mamba_gab_constant_holdout.jsonl` (20 Records, 10 Seeds x 2 Testmengen,
bewusst separate Datei, keine Duplizierung der Befund-36-Daten), 10 Checkpoints
(`artifacts/checkpoints/mamba_seed{1..10}.pt`). Neue Skripte:
`scripts/run_mamba_constant_holdout_worker.py` (Ein-Seed-Job, importiert DataSpec/
Holdout-Konstanten direkt aus `run_arch_constant_holdout_worker.py`, keine Neucodierung),
`scripts/run_mamba_constant_holdout_sweep.py` (Orchestrator, 9 Jobs parallel, gestaffelter
Start -- Struktur identisch zu Befund 36, nur auf eine Architektur reduziert). Primärmetrik
durchgehend `nrmse_run_median` (verifiziert identisch zur Befund-36/38-Konvention, siehe
(g)), h=36 (900 s, Befund 35), sofern nicht anders angegeben.

**a) Angeregte Testmenge -- absolute Verbesserung, aber die relative Lücke zu GRU/LSTM
schließt sich nicht.** `product_moisture_kg_per_kg` verbessert sich deutlich gegenüber
Befund 38 (`n_train=16`, 1 Seed): 0.8004 → 0.3109 (Median über 10 Seeds), Faktor 2.57.
Absolut also ein klarer Fortschritt mit mehr Daten. Der Abstand zu GRU/LSTM wird dabei
aber **nicht kleiner, sondern minimal größer**: bei `n_train=16` war Mamba 6.79x
schlechter als GRU (0.8004/0.1179); bei `n_train=64` sind es 8.27x (0.3109/0.0376, GRU)
bzw. 8.55x (LSTM). Mehr Trainingsdaten helfen Mamba also ungefähr im selben Verhältnis
wie GRU/LSTM, nicht überproportional -- kein Hinweis darauf, dass die Kapazitätslücke aus
Befund 38 mit wachsendem `n_train` verschwindet.

**b) Konstantlauf-Holdout -- der Kernbefund: Mamba schlechter als reine Persistenz, bei
BEIDEN maßgeblichen Zielgrößen, robust über alle 10 Seeds.** Das ist die praktisch
relevante Testmenge (RL-/Standortoptimierungs-Anwendung) und wurde für Mamba in dieser
Sitzung zum ersten Mal überhaupt ausgewertet (Befund 38 prüfte nur die angeregte
Testmenge -- verifiziert: kein `test_set`-Parameter in
`mamba_reduced_comparison.jsonl`/`..._rmsnorm_only.jsonl`).

| Zielgröße | Persistenz | Mamba (Median) | Mamba schlechter in | Verhältnis (Spanne über 10 Seeds) |
|---|---|---|---|---|
| `energy_target` | 0.4668 | 0.6601 | **10/10 Seeds** | 1.10x-1.88x |
| `product_moisture_kg_per_kg` | 0.5437 | 2.2224 | **10/10 Seeds** | 3.03x-7.60x |

Kein Einzelseed-Ausreißer (der zunächst befürchtete Fall aus Befund 35(f)) -- in jedem der
10 Seeds liegt Mamba bei beiden Zielgrößen über der Persistenz-NRMSE. Gesonderte
Streuung `product_moisture_kg_per_kg` x Konstantlauf-Holdout x h=36 (die Kombination mit
der größten verbleibenden Unsicherheit): Median 2.2224, Min 1.6456 (Seed 10), Max 4.1338
(Seed 8) -- Faktor 2.51 zwischen kleinstem und größtem Seed-Wert, aber **alle** zehn Werte
oberhalb 1.0 (d. h. schlechter als Persistenz) und oberhalb der jeweiligen GRU/LSTM-Werte
für dieselbe Kombination (GRU-Median 0.3544, LSTM-Median 0.2723).

**c) h=1 gegen h=36 -- zwei unterschiedliche, unbewiesene Fehlermechanismen, nicht
einer.** `nrmse_run_median`, Median über 10 Seeds, angeregt/Konstantlauf:

| Zielgröße | h=1, angeregt | h=36, angeregt | Wachstum | h=1, Konstantlauf | h=36, Konstantlauf | Wachstum |
|---|---|---|---|---|---|---|
| `product_moisture_kg_per_kg` | 0.0412 | 0.3109 | **7.5x** | 0.2171 | 2.2224 | **10.2x** |
| `energy_target` | 0.1217 | 0.1255 | 1.03x | 0.5695 | 0.6601 | 1.16x |

Zwei getrennte Hypothesen, nicht als gemeinsamer Mechanismus behauptet:

- **`product_moisture_kg_per_kg`: Rollout-Drift-Hypothese.** Bei h=1 ordentlich (0.0412
  angeregt, sogar besser als Persistenz 0.0707) -- der Fehler entsteht überwiegend erst
  in der 36-Schritte-Akkumulation, und zwar auf dem Konstantlauf-Holdout (10.2x) noch
  stärker als auf der angeregten Testmenge (7.5x). Naheliegend, aber hier nicht direkt
  geprüft (keine Zerlegung des Fehlers nach Akkumulationsschritt): die Mamba-Zelle hat
  keine sättigende Nichtlinearität (Befund 38c) -- unter einer über 900 s wirklich
  unveränderten Eingabe (per Definition des Konstantlaufs) könnte ein einmal leicht
  abweichender Zustand ungebremster akkumulieren als bei GRU/LSTM.
- **`energy_target`: Repräsentationsschwäche-Hypothese.** Schon bei h=1 klar hinter
  GRU/LSTM (0.1217 gegen 0.0320/0.0311 angeregt -- Faktor ~3.9x), und h=36 wächst
  gegenüber h=1 kaum (Faktor 1.03-1.16x). Kein Rollout-Drift-Muster -- das Problem sitzt
  bereits im Einzelschritt, nicht in der Akkumulation. Ebenfalls nicht mechanistisch
  geprüft, nur an den Zahlen so charakterisiert.

**d) Epochenzahlen -- Befund 38(e) teilweise bestätigt, nicht vollständig aufgelöst.**
Median über 10 Seeds: `energy_target` 169/200, `product_moisture_kg_per_kg` 65/200,
`d50_um` 35.5/200 (Einzelwerte je Seed siehe Trainingsprotokoll dieser Sitzung).

- `product_moisture_kg_per_kg`: Befund 38(e) diagnostizierte bei `n_train=16` einen
  verrauschten, verfrühten Abbruch (21/200) statt einer echten Kapazitätsgrenze. Bei
  `n_train=64` verdreifacht sich die mediane Epochenzahl (21 → 65) -- die Richtung der
  Diagnose bestätigt sich. Aber: 65 bleibt weit unter den 169 von `energy_target` bei
  derselben Konfiguration, und die Streuung über die Seeds ist groß (26-121 Epochen,
  Faktor 4.7) -- **nicht** vollständig aufgelöst, nur abgeschwächt. Ob der verbleibende
  Rückstand noch denselben verrauschten-Abbruch-Mechanismus hat oder eine andere Ursache,
  ist mit dieser Messung nicht unterschieden.
- `d50_um`: Median 35.5/200 gegenüber 42/200 bei `n_train=16` (Befund 38) -- im Rahmen der
  Seed-Streuung praktisch unverändert (26-74 über die 10 Seeds). Bestätigt die
  Rauschgrenzen-Diagnose aus Befund 38(e) (echte Konvergenz an eine projektweit bekannte
  Grenze, kein Abbruchartefakt) -- mehr Trainingsdaten ändern daran nichts, wie bei einer
  echten Grenze zu erwarten.
- `energy_target`: nur 3 von 10 Seeds erreichen die vollen 200/200 Epochen (Seeds 1, 5, 7)
  -- bei `n_train=16` (Befund 38, 1 Seed) lief es noch vollständig durch. Neue
  Beobachtung dieser Sitzung, in Befund 38 an nur einem Seed nicht sichtbar; nicht weiter
  diagnostiziert (kein vierter Fix-Zyklus, siehe Auftragsbeschränkung dieser Aufgabe).

**e) Gesamtempfehlung.** Die Einordnung aus Befund 38(f) ("kein weiterer Tuning-Versuch,
Ergebnis wird dokumentiert wie es ist") bleibt bestehen und wird durch die vollständige
10-Seed-Stichprobe **nicht entkräftet** -- auf der angeregten Testmenge bestätigt sich
sogar die grundsätzliche Größenordnung des Rückstands zu GRU/LSTM (Punkt a). Für die
Konstantlauf-Eignung, die in Befund 38 mangels Auswertung noch offen war, verschärft sich
das Bild: Mamba unterbietet in dieser Konfiguration bei beiden praktisch relevanten
Zielgrößen durchgehend die reine Persistenz-Baseline (Punkt b). Für den Projektzweck in
dieser Form nicht empfohlen -- dieselbe Schlussfolgerung wie Befund 38, jetzt mit
vollständiger statt einzelseed-basierter Evidenz und um den Konstantlauf-Befund ergänzt.

**f) Methodik-Nebenbefund: Sweep-Kosten bei langlaufenden Einzeljobs.** Kostensonde
(1 Seed, Einzelprozess): 1401.8 s. Hochrechnung für die restlichen 9 Seeds: naiv 46.7 Min.
(perfekte Parallelität, 8 parallele Prozesse), mit dem aus Befund 36(d) bekannten
Mehrprozess-Kontentions-Aufschlag (~1.5x, dort durch Netzlaufwerk-E/A) 70 Min. Real
gemessen: 2747.9 s = 45.8 Min. -- **unter beiden Schätzungen**, der Kontentions-Aufschlag
trat hier nicht auf. Naheliegende, nicht weiter geprüfte Erklärung: die Einzeljobs sind
hier 15-35x länger (1295-2085 s) als im ursprünglichen GRU/LSTM-Sweep, sodass die
Netzlaufwerk-E/A (Checkpoint-/JSONL-Schreiben je Job) einen kleineren Anteil der
Gesamtzeit ausmacht -- das Kontentionsverhältnis skaliert vermutlich nicht linear mit der
Jobdauer, hier einmal gemessen statt nur vermutet. Zusätzlich beobachtet: die
Batch-Logik (identisch zu Befund 36, aus `run_arch_constant_holdout_sweep.py` übernommen)
besetzt freie Slots nicht dynamisch nach -- Seed 10 (Batch 2, einzeln, 621.2 s) wartete
auf den langsamsten Job aus Batch 1 (Seed 3, 2084.8 s), obwohl zwei Slots ab Sekunde 1295
frei gewesen wären; bei dynamischer Nachbesetzung wäre der Sweep ca. 13 Min. früher fertig
gewesen. Für künftige Sweeps mit stark streuenden Einzeljob-Laufzeiten eine mögliche
Verbesserung, hier nicht umgesetzt (Auftrag war die Auswertung, keine Sweep-Infrastruktur).

**g) Metrik-Klarstellung.** In dieser Sitzung zunächst fälschlich vermutet, dann direkt an
`artifacts/runs/mamba_reduced_comparison.jsonl` widerlegt: Befund 38(d) verwendet
durchgehend `nrmse_run_median`, nicht die gepoolte `nrmse` -- dieselbe Konvention wie
Befund 36. Kein neuer Befund, nur Richtigstellung, damit sich der Irrtum nicht in
künftige Dokumentation fortpflanzt.

**h) Vorbehalt.** Wie Befund 34-39 beruhen alle Zahlen dieses Befunds auf der
unkalibrierten GAB-Isotherme (Befund 34) und Δt=25 s/900-s-Horizont (Befund 35).

### 41. Abluft-Austrittsgrößen (T_gas_out, Y_gas_out) algebraisch aus T_bed ableitbar --
Modul `simulation/exhaust.py`, vierte Zielgröße als Voraussetzung, noch nicht aktiviert

Motivation: eine spätere Standort-/RL-Anwendung (Wärmeaustausch zwischen mehreren
Trocknerlinien) braucht Temperatur und Feuchte der Abluft als "Ausgangsseite" derselben
Schnittstelle, deren "Eingangsseite" (`T_preheat`) mangels echter Anwendungspartner-Werte
zurückgestellt bleibt. Befund 31 hat Massenstrom/Feuchte/Temperatur der Abluft bereits
einmal berechnet -- dort als **einmalige diagnostische Rechnung** direkt aus
`solve_steady_state()`, nicht als etwas, das ein Sequenzmodell selbst lernt und im
Rollout fortschreibt. Diese Bestandsaufnahme klärt, ob das nötig ist.

**a) T_gas_out = T_bed -- algebraisch, am Code belegt, keine neue ODE nötig.** Zwei
unabhängige Stellen setzen das bereits gleich: der Docstring von
`thermo.evaporation_rate_kg_per_s` ("Annahme: ideal durchmischtes Bett, Abluft auf
Betttemperatur") und `steady_state.solve_steady_state`, wo `exhaust_sensible_kw` mit
derselben bisektierten Variablen `temperature` gebildet wird, die auch
`bed_temperature_celsius` liefert -- keine zweite Temperaturgröße existiert im Modell.
`T_bed` ist einer der vier ODE-Zustände (`IDX_BED_TEMPERATURE`, Befund 30) und steht
bereits in `GreyBoxFluidBedGranulator.output_names` (Position 2 von 4: `energy_target`,
`bed_temperature_celsius`, `product_moisture_kg_per_kg`, `d50_um`) -- durchläuft also
bereits dasselbe Sensormodell wie die drei gelernten Zielgrößen. Sie ist aber **nicht**
Teil von `target_names`: `DataSpec.target_names` (`evaluation/experiments.py`) trägt
weiterhin nur die drei ursprünglichen Zielgrößen, ebenso jedes geprüfte
Produktions-Sweep-Skript, das `DataSpec(...)` ohne expliziten `target_names`-Override
konstruiert (`study_architectures.py`, `run_arch_constant_holdout_worker.py`,
`run_mamba_constant_holdout_worker.py` u. a.). Die Sequenzmodelle sehen `T_bed` aktuell
nie, weder als Eingang noch als Ziel.

**b) Y_gas_out ist algebraisch ableitbar, ABER NUR WENN T_bed als vierte Zielgröße
vorhergesagt wird -- keine grundsätzlich andere, fehlende Information, sondern genau
diese eine.** `thermo.evaporation_rate_kg_per_s` liefert bereits intern die
Massenbilanz-Form `outlet_humidity = inlet_humidity + evaporation / dry_air_mass_flow`;
`evaporation` selbst hängt aber über die Sättigungsfeuchte `Y_s(T_bed)` und die relative
Feuchte zwingend von `T_bed` ab, nicht nur von der Partikelfeuchte -- mit den beiden
bereits vorhandenen drei Zielgrößen allein (ohne T_bed) ist Y_gas_out **nicht**
rekonstruierbar. Ein Alternativweg über die Wasserbilanz (`evaporation` aus
`d(water_mass)/dt` rückrechnen, ohne `T_bed`) wurde geprüft und verworfen: er bräuchte
stattdessen `solid_mass`/`product_mass_flow` -- ebenfalls interne, nicht vorhergesagte
Zustände, kein Gewinn gegenüber `T_bed`. Fazit: sobald T_bed als vierte Zielgröße
vorhergesagt wird (siehe (c)), ist Y_gas_out mit der bestehenden Funktion exakt
ableitbar -- vorher nicht.

**c) Günstiger Korrektheitslauf, kein Produktionslauf** (`n_train=8`, `hidden_size=8`,
15 Epochen, `patience=0`, `CompositeSequenceModel` mit vier statt drei Zielen,
`bed_temperature_celsius` ergänzt): das vierte Teilnetz baut ohne Fehler, sein
Trainingsverlust fällt über alle 15 Epochen endlich und monoton (0.06438 → 0.05837) --
vom Verlaufsmuster ähnlich glatt wie `energy_target` im selben Lauf (0.26598 → 0.25209),
ein erstes, noch unbewiesenes Indiz für die in Befund 2/39 dokumentierte
"quasi-statische" Charakteristik (`energy_target` zu 98.7 % statisch erklärbar) --
**Nachtrag SA2/Befund 59b: glatte Verlustkurven zeigen nur, dass die statische Regression
eine grobe Richtung trifft, nicht, dass ein trainiertes Modell wenig zusätzliches Signal
findet; Befund 59b beziffert Letzteres auf Faktor 5-8 im RMSE.** Die
**wichtigste** Aussage dieses Laufs: die drei bestehenden Teilnetze sind zwischen einem
Drei- und einem Vier-Ziele-Composite **bitgenau unverändert**
(`max|Trainingsverlust-Differenz| = 0.0`, alle 15 Epochen, alle drei Zielgrößen) -- die
Befund-26-Unabhängigkeit der Teilnetze (`_submodel_seed`, je-Ziel-Skalierer) hält auch
bei vier statt drei Zielen, kein neuer negativer Transfer durch die Erweiterung selbst.
Jetzt Regressionstest:
`tests/test_composite.py::test_composite_with_fourth_target_leaves_existing_submodels_bit_identical`.

**d) Kostenschätzung, in dieser Sitzung selbst korrigiert.** Ein erster Anlauf hatte
fälschlich die Transformer-Kostenmessung aus Befund 37(b) (`6240.03` s Gesamtzeit,
`11.53` s/Epoche, `n_train=32`) als GRU/LSTM-Referenz verwendet -- Fundstellenprüfung
ergab: diese Zahlen stehen unter der Überschrift "37. Transformer-Architektur", nicht
unter Befund 36 (GRU/LSTM). Die tatsächliche GRU/LSTM-Kostenreferenz steht in Befund
37(d) (`n_train=16`, gleicher Lauf, unmittelbar nacheinander gemessen): GRU 209.4 s/475
Epochen (0.441 s/Epoche), LSTM 178.9 s/416 Epochen (0.430 s/Epoche) -- rund Faktor 26
günstiger pro Epoche als der Transformer (7.39 s/Epoche im selben Lauf). Ein viertes,
mutmaßlich quasi-statisches Teilnetz (200 volle Epochen wie `energy_target`) würde damit
grob **1,5-3 Minuten** zusätzliche Trainingszeit kosten (`n_train=16`: ~87 s; hochgerechnet
auf `n_train=32`: ~170-175 s), nicht die ursprünglich (falsch) geschätzten 35-40 Minuten.
Transformer/Mamba bleiben mit Faktor ~14-17x teurer pro Teilnetz korrekt eingeordnet --
das betraf den in Befund 37 selbst gemessenen Architekturvergleich, nicht die falsch
zugeordnete Referenzzahl.

**e) Neues Modul `simulation/exhaust.py`.** `reconstruct_exhaust_state()` ruft
`thermo.evaporation_rate_kg_per_s()` unverändert auf -- keine neue Physik, nur eine
andere Eintrittsseite derselben Formel (Argumente statt Nebenprodukt einer
ODE-Integration/Bisektion), analog zu `thermo.py`/`sorption.py`/`steady_state.py`: reine
Funktion über `GasConfig`/`MaterialConfig`/`SorptionIsotherm` + Arrays, kein
Simulator-Zustand. Bewusst **nicht** in `simulation/__init__.py` re-exportiert -- geprüft,
nicht angenommen: `thermo.py`, `sorption.py`, `fluidization.py`, `energy.py`,
`diagnostics.py`, `steady_state.py` werden dort ebenfalls nicht re-exportiert, jeder
Aufrufer importiert direkt aus dem Submodul (`tests/test_steady_state.py`,
`tests/test_sorption.py` u. a. bestätigen das Muster). Gegen `solve_steady_state()` am
Nominalpunkt aus Befund 31 geprüft: bitgleiche Übereinstimmung
(`T_gas_out`: 62.310862 °C, `Y_gas_out`: 0.01711927 kg/kg, beide Wege rufen dieselbe
Funktion mit denselben Eingaben auf), zusätzlich gegen den dokumentierten
Befund-31-Tabellenwert (`Y_out = 0.0171` kg/kg) abgeglichen. Jetzt Regressionstests:
`tests/test_exhaust.py::test_reconstruction_matches_steady_state_reference_bitgenau`,
`::test_reconstruction_matches_documented_befund_31_values`.

**f) Entscheidung, hier festgehalten.** `DataSpec.target_names` bleibt bei den drei
ursprünglichen Zielgrößen -- explizites Opt-in pro Aufruf (wie in den Tests dieses
Befunds), kein geänderter Default. Grund: der Default wird ohne Override von praktisch
jedem Produktions-Sweep-Skript verwendet (siehe (a)); eine Umstellung hätte damit
produktionsweite Wirkung auf jeden künftigen Sweep, nicht nur lokale -- eine eigene,
spätere Entscheidung, wenn ein echter Produktionslauf mit vier Zielen ansteht.

**g) Was dieser Befund NICHT zeigt.** Keine Aussage zur tatsächlichen Vorhersagegüte von
`bed_temperature_celsius` auf echten Trainingsdaten (kein Produktionssweep, keine
Lernkurve) -- nur, dass die Infrastruktur (Composite-Erweiterung, algebraische
Rückrechnung) korrekt verdrahtet ist. Die Vermutung aus (c) (quasi-statische
Charakteristik ähnlich `energy_target`) ist damit plausibel gemacht, nicht bewiesen.

### 42. T_bed bei Produktionsgröße ausgewertet, End-zu-End-Abluft-Rückrechnung erstmals
geprüft -- klar über Persistenz (Gegensatz zu Mamba/Befund 40), dabei ein stiller
Fehlpaarungs-Fehler in der neuen Metrik gefunden und behoben

Befund 41 hat die Infrastruktur geschaffen (`bed_temperature_celsius` als vierte
Zielgröße, `simulation/exhaust.py::reconstruct_exhaust_state()` zur algebraischen
Ableitung von T_gas_out/Y_gas_out), aber ausdrücklich offengelassen (41g), ob T_bed bei
echter Trainingsgröße gut genug vorhergesagt wird, um diese Rückrechnung tatsächlich
nutzbar zu machen. Dieser Befund liefert genau diese Messung -- und, wichtiger als eine
isolierte T_bed-NRMSE, die **End-zu-End-Güte der Rückrechnung selbst**: wie gut sind
T_gas_out/Y_gas_out, wenn `reconstruct_exhaust_state()` mit vorhergesagtem statt wahrem
T_bed/Feuchte gefüttert wird. Neues Modul `evaluation/exhaust_reconstruction.py`, neue
Skripte `scripts/run_tbed_exhaust_cost_probe.py` (Einzellauf-Kostenprobe) und
`scripts/run_tbed_exhaust_sweep.py`/`run_tbed_exhaust_sweep_worker.py` (voller Sweep),
Log `artifacts/runs/befund42_tbed_exhaust_sweep.jsonl` (40 Records: GRU+LSTM × 10 Seeds ×
2 Testmengen). `exhaust.py`/`greybox.py`/`thermo.py` selbst unverändert -- nur lesend/
aufrufend genutzt, wie in Befund 41.

**a) Massenstrom-Leck vor dem Sweep erkannt und korrigiert, bevor Rechenzeit investiert
wurde.** `dry_air_mass_flow_kg_per_s` hängt entgegen dem ersten Anschein konstruktions-
bedingt von der Betttemperatur ab (`greybox._rhs` wertet die Luftdichte bei
`bed_temperature` aus, nicht bei Einlasstemperatur, `gas_density_kg_per_m3(gas,
bed_temperature, ...)`). Für die Rekonstruktions-Seite (aus Vorhersage) wird der
Massenstrom deshalb über `rescale_dry_air_mass_flow_kg_per_s()` auf die vorhergesagte
Betttemperatur umgerechnet (Dichteverhältnis, `thermo.gas_density_kg_per_m3` unverändert
importiert -- Gasgeschwindigkeit/Querschnittsfläche kürzen sich im Verhältnis heraus,
keine neue Physik). `inlet_absolute_humidity_kg_per_kg` bleibt in beiden Aufrufen (Referenz
und Rekonstruktion) wahr -- eine reine Störgröße (Umgebungstemperatur/-feuchte), kein
Leck. Regressionstests:
`tests/test_exhaust_reconstruction.py::test_rescaled_dry_air_mass_flow_matches_true_value_when_prediction_is_exact`,
`::test_rescaled_dry_air_mass_flow_decreases_for_overpredicted_bed_temperature`.

**b) Der stille Fehler -- eigenständiger methodischer Fund.** Die erste Fassung von
`evaluate_exhaust_reconstruction()` prüfte nicht, ob das übergebene
`(GeneratedData, SequenceDataset)`-Paar tatsächlich zusammengehört -- `test` muss aus
genau `data.trajectory`/`data.split` gebaut worden sein, sonst zeigen `run_index`/
`window_start` zwar formal gültig, aber auf die **falsche** Trajektorie. Bei zufällig
passender Fensterzahl (hier: Trainingspool und Konstantlauf-Holdout-Pool mit gleichem
`n_test=32`) lief eine Verwechslung **klaglos durch** und hätte still falsche Zahlen
geliefert -- kein Absturz, keine Warnung, nur eine plausibel aussehende, aber aus dem
falschen Pool berechnete Diagnostics-Zuordnung. Gefunden nicht durch einen Fehlerbericht,
sondern durch eine gezielte Gegenprobe (`(excited, test_constant)` absichtlich als
falsches Paar übergeben, vor dem eigentlichen Sweep, am Rande der Kostenschätzung für die
Konstantlauf-Wiederverwendung). Behoben durch eine explizite Konsistenzprüfung: pro Batch
wird `data.trajectory.outputs` an den berechneten (Trajektorie-Zeile, Zeit)-Indizes gegen
`future_targets_truth` (über den Fensterindex bezogen) für `bed_temperature_celsius`/
`product_moisture_kg_per_kg` verglichen -- beide referenzieren denselben Wert auf zwei
unabhängigen Wegen, eine Abweichung kann nur aus einem falschen Paar kommen. Bei
Abweichung wirft die Funktion jetzt einen `ValueError` statt stillschweigend
weiterzurechnen. Das ist die Art Fehler, die am gefährlichsten ist, weil sie sich nicht
als Absturz zeigt, sondern als plausible, aber falsche Zahl. Regressionstest:
`tests/test_exhaust_reconstruction.py::test_evaluate_exhaust_reconstruction_rejects_mismatched_data_and_test`
(prüft sowohl, dass korrekte Paarungen weiterhin funktionieren, als auch, dass die
ursprünglich unentdeckte Verwechslung jetzt zuverlässig scheitert).

**c) Konsistenzbeleg über den vollen Sweep, nicht nur den ursprünglichen Einzelfall.**
T_gas_out = T_bed exakt (Befund 41a) -- damit muss
`exhaust_nrmse_run_median/T_gas_out_celsius/h{1,36}` bitgleich mit
`nrmse_run_median` gegen die rauschfreie Referenz (`nrmse_truth_run_median/
bed_temperature_celsius/h{1,36}`) übereinstimmen, für **jeden** Record. Programmatisch
geprüft über alle 40 Records × 2 Horizonte = **80 Vergleiche** (exakter `==`-Vergleich,
nicht nur `np.isclose`): **0 Abweichungen.** Der Fix aus (b) hält über den gesamten Sweep
(beide Architekturen, beide Testmengen, alle 10 Seeds), nicht nur an den drei
ursprünglich geprüften Testfällen.

**d) Hauptergebnis: T_bed-NRMSE, `nrmse_run_median` (Median über 10 Seeds).**

| | GRU h=1 | GRU h=36 | LSTM h=1 | LSTM h=36 |
|---|---|---|---|---|
| Angeregte Testmenge | 0.0590 | 0.0561 | 0.0617 | 0.0583 |
| Konstantlauf-Holdout | 0.6562 | 0.6506 | 0.6832 | 0.6942 |
| Verhältnis (Holdout/angeregt) | 11.13x | 11.59x | 11.08x | 11.92x |

Auf der angeregten Testmenge liegt T_bed zwischen `energy_target` (0.0301-0.0320) und
`d50_um` (0.0803-0.0903) -- wie in Befund 41c vermutet, kein Ausreißer in beide
Richtungen. Auf dem Konstantlauf-Holdout verschlechtert sich T_bed dagegen um Faktor
**11.1-11.9x** gegenüber der angeregten Testmenge -- deutlich stärker als `energy_target`
(7.1-8.1x) und `product_moisture_kg_per_kg` (7.4-9.4x) im selben Sweep, `d50_um`
ausgenommen (0.99-1.17x, praktisch unverändert). **Trotzdem, und das ist die zentrale
Einordnung dieses Befunds:** T_bed bleibt auf dem Konstantlauf-Holdout absolut klar
**besser als die Persistenz-Baseline** (0.7364 bei h=1, 1.0915 bei h=36 -- günstig
nachgerechnet auf demselben festen Pool, `PersistenceModel`, keine neue Sweep-Dimension):

| | GRU | LSTM |
|---|---|---|
| Besser als Persistenz, h=1 | 10.9 % | 7.2 % |
| Besser als Persistenz, h=36 | 40.4 % | 36.4 % |

**Das ist der Gegensatz zu Befund 40, nicht dieselbe Situation unter anderem Namen.**
Mamba unterbot dort auf demselben Konstantlauf-Holdout-Muster die Persistenz bei
`energy_target`/`product_moisture_kg_per_kg` in 10/10 Seeds -- hier ist trotz der großen
relativen Degradation (Faktor ~11x) die absolute Aussage die gegenteilige: GRU/LSTM
bleiben bei T_bed auch im ungünstigsten getesteten Betriebspunkt klar über der
parameterfreien Baseline. Die große Verhältniszahl in der obigen Tabelle darf nicht mit
einem Mamba-artigen Versagen verwechselt werden.

**e) End-zu-End-Exhaust-Metrik -- die eigentliche Nutzlast dieses Befunds**
(`nrmse_run_median`, Median über 10 Seeds, Rekonstruktion aus Vorhersage gegen
Rekonstruktion aus Wahrheit, Massenstrom-Leck korrigiert siehe (a)):

| | GRU h=1 | GRU h=36 | LSTM h=1 | LSTM h=36 |
|---|---|---|---|---|
| T_gas_out, angeregt | 0.0418 | 0.0296 | 0.0441 | 0.0319 |
| T_gas_out, Konstantlauf-Holdout | 0.4868 | 0.2847 | 0.5413 | 0.3964 |
| Y_gas_out, angeregt | 0.0830 | 0.1067 | 0.0923 | 0.1158 |
| Y_gas_out, Konstantlauf-Holdout | 0.2621 | 0.3727 | 0.2675 | 0.3327 |

Y_gas_out degradiert durchgehend stärker als T_gas_out (bereits an Seed 1 in der
Kostenprobe beobachtet, jetzt über 10 Seeds bestätigt) -- plausibel, da Y_gas_out sowohl
von T_bed als auch von `product_moisture_kg_per_kg` abhängt (kombinierter Fehler aus
beiden Vorhersagen), während T_gas_out ausschließlich von T_bed abhängt. Bei GRU liegt
Y_gas_out bei h=36 auf dem Konstantlauf-Holdout über T_gas_out (0.3727 gegen 0.2847) --
konsistent mit der Erklärung (kombinierter Fehler aus T_bed und `product_moisture`). Bei
LSTM ist es umgekehrt: T_gas_out (0.3964) liegt über Y_gas_out (0.3327) -- die Erklärung
trägt hier nicht einheitlich, das Muster ist architekturabhängig, nicht durchgängig.
Nicht weiter mechanistisch geprüft.

**f) h=1-vs-h=36-Umkehr, jetzt über 10 Seeds bestätigt statt nur Seed 1 -- kleinerer,
aber weiterhin vorhandener Effekt.** T_bed, angeregte Testmenge: GRU-Median 0.0590 →
0.0561 (10/10 Seeds mit h=36 < h=1), LSTM-Median 0.0617 → 0.0583 (9/10 Seeds). Zum
Vergleich, im selben Sweep: `energy_target` 8/10 Seeds (beide Architekturen),
`product_moisture_kg_per_kg` 0/10 Seeds (durchgehend h=36 schlechter, wie in den meisten
bisherigen Befunden erwartet), `d50_um` 8/10 (GRU) bzw. 2/10 (LSTM) Seeds. Die Umkehr ist
damit kein Artefakt der neuen Zielgröße oder des neuen Exhaust-Codes (der ohnehin nur
Y_gas_out/T_gas_out betrifft, nicht die direkte T_bed-NRMSE), sondern ein Muster, das bei
mehreren Zielgrößen in diesem `n_train=64`-Sweep auftritt, bei `product_moisture_kg_per_kg`
aber durchgehend fehlt.

Präzisierung, wortwörtlich aus der vorherigen Klärung übernommen, nicht neu formuliert:

> h=36 ist eine kumulative NRMSE über die Schritte 1..36 (Docstring
> pooled_nrmse/per_run_nrmse, evaluator.py), keine Punktschätzung am 36. Schritt -- eine
> bewusste Designentscheidung, die verhindert, dass ein zwischenzeitlich divergierendes
> Modell durch zufällige späte Annäherung gut aussieht. Für die überwiegende Mehrheit der
> bisherigen Vergleiche (h=36 schlechter als h=1) ändert das nichts an der Interpretation
> -- eine schlechtere Zahl unter dieser bewusst strengen Metrik ist ein robuster, nicht
> abgeschwächter Befund. Nur für den seltenen, hier erstmals beobachteten umgekehrten Fall
> (h=36 besser als h=1, bei drei von vier Zielgrößen in diesem einen n_train=64-Lauf) ist
> die Präzisierung entscheidend: das bedeutet NICHT, dass das Modell bei späten
> Rollout-Schritten präziser vorhersagt, sondern dass die Referenz-Streuung sigma_y über
> die längere 1..36-Spanne stärker wächst als der akkumulierte absolute Fehler -- eine
> Eigenschaft der Metrik-Normierung über diesen speziellen Lauf/n_train=64, keine Aussage
> über verbesserte Spätvorhersage.

**g) Reale Sweep-Kosten: Faktor ~3.7x über der Schätzung, weiterer Beleg für die
Kontentions-Reihe aus Befund 36d/40f.** Geschätzt (basierend auf dem unbelasteten
Einzellauf, 1223.3 s, plus ~1.5x Kontentions-Aufschlag wie in Befund 36d/40f): ~61 Min.
Real: **224.0 Min** (13439.1 s, 20 Jobs, `--parallel 8`, gestaffelter Start) -- der
tatsächliche Kontentions-Aufschlag lag bei 8 parallelen Prozessen bei Faktor ~3.6-4.9x
gegenüber dem unbelasteten Einzellauf, nicht bei den übertragenen ~1.5x. Kein Job
fehlgeschlagen, nur die Zeitschätzung war falsch -- Ursache nicht weiter untersucht
(plausibelste, nicht verifizierte Vermutung: `torch.set_num_threads(1)` begrenzt nur
Torchs eigene Intraop-Threads, nicht notwendigerweise darunterliegende BLAS/OpenMP-
Bibliotheken).

**h) Vorbehalt.** Wie Befund 34-41 beruhen alle Zahlen dieses Befunds auf der
unkalibrierten GAB-Isotherme (Befund 34) und Δt=25 s/900-s-Horizont (Befund 35).

### 43. Trainingsbudget bei `n_train=8` unter aktuellem Standard geprüft -- kein
Mamba-artiges Versagen, aber ein echter, kreuz-architektur-konsistenter Restrückgang bei
zwei von drei Zielgrößen; dabei eine eigene Kennzahl an Rohdaten widerlegt, bevor sie in
die Dokumentation gelangte

**a) Ausgangslücke.** Befund 28, das GRU/LSTM-Lernkurven-Fundament dieses Projekts
(Stützstellen 8/16/32/64, `artifacts/runs/arch_composite.jsonl`/`arch_gru.jsonl` --
Provenienzvorbehalt zu `arch_composite.jsonl` siehe Befund 33), loggt
für `n_train=8` keine tatsächliche Epochen- oder Gradientenschrittzahl -- nur den
Konfigurations-Deckel (`max_epochs=200`, `batch_size=256`). Befund 24s oft zitierte
"40 Gradientenschritte insgesamt bei `n=8`" ist **kein** übertragbarer Referenzwert dafür:
geprüft, nicht aus dem Gedächtnis übernommen, stammt die Zahl aus einem isoliert roten
Unit-Test (`test_gru_trains_and_beats_persistence_on_energy`, seither korrigiert) mit einer
völlig anderen Fenstergeometrie -- ≈18 Fenster/Lauf dort gegen 136-138 Fenster/Lauf in
Befund 26-42 (Formel `floor((n_steps - history_length - horizon)/stride) + 1`, gegen
`metrics.n_windows` in mehreren Sweeps exakt verifiziert: 4416=32×138 bei
`horizon=24`-Konfiguration, 4352=32×136 bei `horizon=36`-Konfiguration). Ein direkter
Zahlenvergleich zwischen Befund 24 und Befund 28/37/38/40 wäre deshalb unzulässig.

**b) Neue, generische Logging-Infrastruktur.** `training_history` (Trainingsverlust je
Epoche) existierte bereits vor diesem Befund in `TorchSequenceModel`, wurde aber von
`describe()` nie ins `RunRecord`/JSONL übernommen -- nur `scripts/run_tbed_exhaust_
sweep_worker.py` (Befund 41/42) loggte lokal, außerhalb des Kernpfads, eine Epochenzahl
(`training/epochs/{target}`). Neuer, additiver Hook `SequenceModel.training_diagnostics()`
(`models/base.py`, Default `{}`, analog zu `n_parameters`), implementiert in
`TorchSequenceModel` (`epochs_run`, `gradient_steps`, `train_loss_ratio_last10`) und
`CompositeSequenceModel` (Aggregation je Teilmodell mit Zielgrößen-Suffix), eingehängt in
`evaluation/experiments.py::_record()` -- der einzige Ort, an dem `RunRecord`s im
Kernpfad entstehen (Befund 27: alle Studien-Schleifen dort konsolidiert), also automatisch
für jeden künftigen Sweep aktiv, nicht nur für Einzelskripte. `gradient_steps = epochs_run
× batches_per_epoche`, mit `batches_per_epoche = ceil(n_train_windows / batch_size)` --
bewusst die Ceiling-, nicht die Floor-Formel, weil `SequenceDataset.iter_batches()`
(`data/windows.py`) den letzten, kleineren Batch mitzählt. An einem günstigen Testlauf
(`batch_size=50` statt 256, absichtlich nicht teilerfremd zur Fensterzahl 144) gegen
Floor/Ceiling unterscheidbar verifiziert: geloggt 45 Gradientenschritte bei 15 Epochen,
exakt `15 × ceil(144/50) = 15 × 3 = 45` -- Floor hätte `15 × 2 = 30` ergeben. Volle
Testsuite vor und nach der Implementierung grün (297 bestanden, 1 übersprungen, 0
Fehler/Fehlschläge), inklusive der eigens dafür geprüften Stelle `tests/test_protocol.py`
(`_assert_records_equal`, exakter `params`-Vergleich zweier `RunRecord`-Sätze) -- unkritisch,
weil beide verglichenen Sätze über denselben `_record()`-Pfad laufen und die neuen Felder
deshalb symmetrisch auf beiden Seiten erscheinen.

**c) Die Kennzahl-Kalibrierung an zwei Beispielen war nicht genug -- ein Lehrstück, nicht
kaschiert.** Die erste Fassung von `train_loss_ratio_last10` verglich einen einzelnen
Epochenwert gegen den zehntletzten. Kalibriert wurde sie ausschließlich an zwei bereits von
Hand diagnostizierten Mamba-Fällen aus Befund 38(e) (0.31 für einen erkannt verrauschten
Abbruch, 0.94 für echte Konvergenz) -- beide Kalibrierpunkte bestanden, die Kennzahl wurde
versehentlich schon als tragfähig behandelt. Am ersten echten neuen Datensatz (LSTM,
`product_moisture_kg_per_kg`, `n_train=8`) widerlegte sie sich selbst: die 20 letzten
rohen Trainingsverlust-Werte (Epoche 115-134) sprangen ohne erkennbaren Trend zwischen
0.000075 und 0.000105 (~±20 % relative Amplitude) --

```
96 105  83  81  75  76  78  76  76  76  80  87  87  88  86  80  75  80  81  82   (×10⁻⁶)
```

-- und der Zwei-Punkt-Quotient (1.074, aus Epoche 134 gegen Epoche 124) hing sichtbar
vom Zufall ab, auf welche Stelle dieser Schwankung die beiden verglichenen Epochen gerade
fielen (Epoche 131 gegen Epoche 121 hätte 0.96 ergeben, nicht 1.07). Die daraus zunächst
gezogene Schlussfolgerung ("`product_moisture` zeigt klar keinen Budget-Engpass, ein
sauberer Stopp") war zu stark formuliert und wurde direkt danach zurückgenommen, **bevor**
sie in dieses Dokument gelangte. Ersetzt durch einen Fenstermittelwert (Mittel der letzten
5 Epochen gegen Mittel der 5 Epochen davor) -- robuster gegen genau dieses
Epoche-zu-Epoche-Rauschen, ohne die Komplexität einer Regressionssteigung für eine reine
Ja/Nein-Diagnose. **Lehre:** eine Kennzahl, die nur an den Fällen kalibriert wird, für die
man bereits die Antwort kennt, kann an echten neuen Rohdaten trotzdem versagen -- zwei
kalibrierende Beispiele reichen nicht als Nachweis der Robustheit, nur als Plausibilitäts-
check der Richtung.

**d) Endergebnis, GRU/LSTM, `n_train=8`, aktueller Standard.** Fingerprint/Split exakt aus
Befund 37/38/40 wiederverwendet (`process/fingerprint=82f65f576a2e5253`,
`split/digest=5b9628a151344f37`, `master_seed=1`, bitgleich geprüft) -- der `n=8`-Punkt
sitzt damit auf derselben Achse wie die bereits vorliegenden `n=16/32/64`-Punkte. 1088
Trainingsfenster (8×136), `batches_per_epoche=5`.

| Teilnetz | GRU `epochs_run`/`gradient_steps` | GRU ratio | LSTM `epochs_run`/`gradient_steps` | LSTM ratio |
|---|---|---|---|---|
| `energy_target` | 200/200, 1000 (Budget-Deckel, kein Early Stopping) | 0.9463 | 200/200, 1000 (Budget-Deckel, kein Early Stopping) | 0.9892 |
| `product_moisture_kg_per_kg` | 134/200, 670 | 0.9341 | 135/200, 675 | 0.9322 |
| `d50_um` | 60/200, 300 | 0.9634 | 43/200, 215 | 1.0056 |

Alle sechs Werte (3 Zielgrößen × 2 Architekturen) liegen in einem engen Band von
0.93-1.01 -- kein Mamba-artiges, eindeutig-noch-fallendes Versagen wie Befund 38(e)s 0.31
bei `n_train=16`. Die Kreuz-Architektur-Konsistenz ist dabei selbst der stärkste Beleg,
dass die Restwerte echtes Signal sind, nicht Kennzahl-Rauschen: `product_moisture` liegt
bei GRU (0.9341) und LSTM (0.9322) nahezu identisch, `d50_um` bei GRU (0.9634) knapp
unter und LSTM (1.0056) knapp über 1.0 -- zwei unabhängig trainierte Architekturen, die
an denselben acht Trainingsläufen dasselbe Restmuster zeigen, ist unwahrscheinlich, wenn
die Kennzahl nur Rauschen abbildete. `d50_um` liest sich damit als flach/konvergiert
(konsistent mit der bereits etablierten, architekturunabhängigen Rauschgrenze aus Befund
38e/40d), `product_moisture` und in geringerem Maß `energy_target` zeigen dagegen einen
echten, moderaten Restrückgang von ~5-7 % über die letzten 10 Epochen -- kein
dramatischer Befund, aber auch keine saubere, vollständig ausgereizte Konvergenz.
`energy_target` bleibt zusätzlich dadurch auffällig, dass es bei beiden Architekturen bis
zum vollen 200-Epochen-Deckel läuft, ohne dass Early Stopping je griff -- dasselbe Muster
wie bei etlichen Seeds in Befund 40(d)/42 bei `n_train=64`.

**e) Einordnung/Konsequenz.** Ausdrücklich **keine** retroaktive Neubewertung von Befund
28 selbst -- andere Physik (alte, konstante Gleichgewichtsfeuchte statt GAB-Isotherme,
Befund 34) und andere Fenstergeometrie (`horizon=24` statt 36), ein Zahlenvergleich wäre
nicht zulässig (siehe (a) und bereits Befund 36e). Für **künftige** kleine-`n`-Arbeit --
die noch offene ARX/NARX-Lernkurven-Erweiterung bis 128/256, oder jede künftige
`n=8`/`16`-Architekturprüfung -- eine leichte, durch das gefundene Ausmaß begründete
Empfehlung: `patience` bei sehr kleinem `n_train` großzügiger wählen. Kein vollständiges
Redesign der Budget-Logik (die in der Design-Diskussion vor diesem Befund erwogenen
Optionen -- `max_epochs` umgekehrt proportional zu `n_train`, oder ein
Gradientenschritt-Zähler statt Epochenzähler als Abbruchkriterium) ist beim hier
gefundenen Ausmaß (0.93-1.01, kein Extremfall) derzeit gerechtfertigt.

**f) Vorbehalt.** Wie Befund 34-42 beruhen alle Zahlen dieses Befunds auf der
unkalibrierten GAB-Isotherme (Befund 34) und Δt=25 s/900-s-Horizont (Befund 35).

### 44. ARX/NARX-Lernkurve bis `n=128/256` unter aktueller Konfiguration -- Befund 16s
scharfe Sättigungsgrenze bei `n=16` zu grob, NARX schließt die Lücke zu ARX drastisch,
überholt ihn aber zu keinem Zeitpunkt

**a) Ausgangslücke.** `docs/05_arbeitsplan_ap1.md`, Schritt 6 Teil 2, Punkt 2: „Lernkurven
bis 128 und 256 für ARX und NARX. Die NARX-Kurve läuft bei n = 64 noch; ohne diese Punkte
wird eine Sättigung berichtet, die nur die lineare Baseline betrifft." -- offen seit der
ursprünglichen Formulierung in Befund 16 (dort: „der ARX ist bei 16 Läufen fertig … das
NARX fällt über den ganzen Bereich monoton"), zuletzt in Befund 43(e) erneut als offener
Punkt benannt. Wie bei Befund 24 (Befund 43a) gilt: eine unter anderer Konfiguration
entstandene Zahl ist kein automatisch übertragbarer Referenzwert. Befund 16 lief unter der
Platzhalter-Config (`greybox_placeholder.yaml`, keine GAB-Isotherme, `horizon=24` = 600 s)
-- also **vor** Befund 34 (GAB-Isotherme) und Befund 35 (Δt=25 s/900-s-Horizont). Diese
Prüfung wiederholt Befund 16 deshalb nicht als Bestätigung, sondern als eigenständige
Neumessung unter aktuellem Standard.

**Versuchsaufbau.** `scripts/study_learning_curve.py`, erweitert (Config-Default auf
`greybox_gab_placeholder.yaml`, `horizon`-Default auf 36, `sizes`/`n-train-max` bis 256,
`narx` in `FACTORIES` ergänzt -- vorher fehlte NARX dort vollständig, das Skript konnte
Befund 16 in seiner damaligen Form nicht reproduzieren). 10 Seeds (1-10), Stützstellen
8/16/32/64/128/256, Pool 512 Läufe (`n_train_max=256`, `n_val=128`, `n_test=128`),
`process/fingerprint=82f65f576a2e5253` -- identisch mit Befund 37/38/40/43, aktueller
Standard bestätigt, nicht nur behauptet. Log: `artifacts/runs/
arx_narx_learning_curve_gab_900s.jsonl` (240 Records: 6 Stützstellen × 4 Modelle × 10
Seeds). Realer Kostenpunkt vorab gemessen, nicht nur hochgerechnet: 15 s für die teuerste
Einzelstützstelle (`n_train_max=256`, 1 Seed, alle vier Modelle); voller Sweep 274 s.

**b) Korrigierte ARX-Sättigungsaussage.** Primärmetrik `nrmse_run_median`, Median über 10
Seeds, `product_moisture_kg_per_kg`, h=36 (Rollout, 900 s):

| n_train | ARX Median | Δ zur vorigen Stützstelle |
|---|---|---|
| 8 | 0.5460 | -- |
| 16 | 0.5154 | -5.6 % |
| 32 | 0.4694 | **-8.9 %** |
| 64 | 0.4328 | -7.8 % |
| 128 | 0.4227 | -2.3 % |
| 256 | 0.4281 | +1.3 % (Stützstellen-Rauschen, keine echte Verschlechterung) |

Die größte relative Einzelverbesserung liegt bei `n=32` (-8.9 %), nicht bei `n=16`
(-5.6 %), wie Befund 16 unter alter Konfiguration nahelegte. ARX verbessert sich messbar
noch bis `n=64` (-7.8 %), erst ab `n=128` wird der Zugewinn klein (-2.3 %), bei `n=256`
liegt die Kurve im Rauschen. Bei `energy_target`, h=36, dasselbe Bild in der Form (größtes
Delta bei `n=16` mit -12.5 %, aber `n=32` mit -6.2 % noch deutlich vor dem Abklingen auf
-2.2 % bei `n=128` und +0.5 % bei `n=256`) -- in beiden Zielgrößen eine graduell
auslaufende Kurve über mehrere Stützstellen, kein scharfer Abbruch bei einer einzigen.

**Korrektur zu Befund 16.** Die frühere Formulierung „der ARX ist bei 16 Läufen fertig"
beschreibt eine reale, aber unter alter Physik (keine GAB-Isotherme, 600-s- statt
900-s-Horizont) gemessene Eigenschaft. Unter aktueller Konfiguration ist die Sättigung
eine sanfte, über `n=16→32→64→128` auslaufende Kurve. Analog zu Befund 43(a)/Befund 24:
die alte Zahl war kein direkt übertragbarer Referenzpunkt, sondern selbst erst unter
aktueller Konfiguration zu verifizieren -- was hiermit geschehen ist, mit korrigiertem
Ergebnis.

**c) NARX vs. ARX -- Hauptergebnis.** Verhältnis NARX zu ARX (`nrmse_run_median`, Median
über 10 Seeds, h=36 = Rollout-Primärhorizont), NARX relativ schlechter als ARX in %:

| n_train | `product_moisture_kg_per_kg` | `energy_target` |
|---|---|---|
| 8 | +69.1 % | +79.1 % |
| 16 | +39.6 % | +67.2 % |
| 32 | +39.2 % | +43.5 % |
| 64 | +31.2 % | +31.4 % |
| 128 | +25.7 % | +19.1 % |
| 256 | **+12.4 %** | **+8.9 %** |

NARX schließt die relative Lücke zu ARX über den gesamten Bereich drastisch
(`product_moisture`: 69 % → 12 %; `energy_target`: 79 % → 9 %), **überholt ARX aber zu
keinem Zeitpunkt** -- an keiner der sechs Stützstellen, bei keiner der beiden Zielgrößen,
liegt der NARX-Median unter dem ARX-Median.

**Schlussfolgerung zur Nichtlinearitätsfrage.** `RandomFeatureNARX` unterscheidet sich von
`LinearARXModel` konstruktionsbedingt in genau einer Sache -- der Basisfunktion
(nichtlinear statt linear) --, beide werden mit identischer Ridge-Regression gefittet
(`_ridge_solve`, kein Epochen-/Gradientenschritt-Konzept, vgl. Befund 43c). Reine
Nichtlinearität bringt bei dieser Aufgabe zu keiner geprüften Stützstellengröße einen
Vorteil gegenüber Linearität -- die Lücke wird mit mehr Daten kleiner, kehrt sich aber
nicht um. Ein etwaiger Gewinn von GRU/LSTM gegenüber ARX kann damit nicht aus der
Basisfunktion allein stammen, sondern müsste aus gelerntem Zustand und/oder
Rollout-Training kommen. **Ausdrücklich offengelassen:** eine zahlenmäßige Einordnung
gegen GRU/LSTM ist hier nicht Teil des Befunds -- Befund 28 (die bestehende GRU/LSTM-Kurve)
lief unter dem gleichen alten Standard wie Befund 16 (siehe Befund 43e) und ist mit dieser
neuen, aktuellen ARX/NARX-Kurve deshalb nicht direkt vergleichbar. Ein echter neuronaler
Vergleich bei `n=128/256` unter aktueller Konfiguration wäre ein eigener Arbeitsstrang
(echtes Training statt geschlossener Form) und ist nicht Teil dieser Aufgabe.

**Abgrenzung (nachträglich ergänzt).** Diese Schlussfolgerung gilt für `RandomFeatureNARX`
-- einen statischen nichtlinearen Regressor mit verzögerten Eingängen, angepasst per
geschlossener Ridge-Regression, ohne Epochen-/Gradientenschritt-Konzept und ohne
Rollout-Training. Sie überträgt sich **nicht** auf GRU/LSTM/Mamba/xLSTM: diese lernen einen
rekurrenten Zustand und werden im Mehrschritt-Rollout trainiert, nicht nur einschrittig
gefittet. Gegenzahlen aus dem Bestand, die diese Grenze konkret machen: Befund 28, `INLINE`,
`n_train=8`, `nrmse_run_median`: ARX 0.2319, GRU 0.0690, LSTM 0.0820 (`arch_composite.jsonl`,
Provenienzvorbehalt siehe Befund 33) -- GRU/LSTM liegen hier klar **unter** ARX, nicht in
dessen Nähe. Befund 36c, `energy_target`, Konstantlauf-Holdout Pool 999, `nrmse_run_median`:
Persistenz 0.4668 gegen ARX 1.3417 -- linearer ARX ist im Mehrschritt-Rollout instabil und
liegt dort schlechter als die reine Persistenz-Baseline. Als Referenz für Rollout-Metriken
sagt ARX deshalb wenig aus; bindende Baseline für Rollout-Vergleiche ist Persistenz, nicht
ARX.

**d) `static`-Ausreißer, Seed 5, `n=8`.** `product_moisture_kg_per_kg`, h=1: Seed 5
liefert 9.2894 gegen sonst 1.2-2.4 im selben `n_train=8`-Block (h=36 analog: 9.2650 gegen
1.2-2.5). Nur benannt, nicht tiefer untersucht -- `static` (quadratische Regression ohne
Historie) ist ohnehin nie die maßgebliche Baseline in diesem Projekt (Persistenz/ARX/NARX
sind die relevanten Vergleichspunkte), und der Ausreißer verschwindet bereits bei `n=16`
(1.2-2.1, wieder im normalen Band). Keine Auswirkung auf (b) oder (c), die beide nur
ARX/NARX/Persistenz betreffen.

**e) Vorbehalt.** Wie Befund 34-43 beruhen alle Zahlen dieses Befunds auf der
unkalibrierten GAB-Isotherme (Befund 34) und Δt=25 s/900-s-Horizont (Befund 35).

### 45. Anregungsbandbreite -- die schnelle Mode ist eine dokumentierte Grenze, kein Mangel

Realisierte Umschaltzeiten der Anregung (`excitation_diagnostics.excitation_report()`,
unverändert, gegen V0/GAB, τ₁=30.23 s, Δt=25 s): volle Spanne der Zulufttemperatur
(60 K bei 0.5 K/s Ratenlimit) braucht 120 s (= 3.97·τ₁), die volle Spanne des
Sprühmassenstroms 130 s (= 4.30·τ₁) -- Mittel 125 s ≈ 4.1·τ₁. Aus dem Ratenlimit folgt
direkt: jede Zulufttemperaturänderung > 15.1 K (= 0.5 K/s · τ₁) ist per Ratenlimit
langsamer als τ₁. Zusätzlich limitiert die Abtastzeit: 2·Δt = 50 s > τ₁ = 30.2 s -- die
schnellste Mode ist damit auch bei perfekter Aktordynamik nicht mit zwei Abtastpunkten
auflösbar (Nyquist-artiges Argument, vgl. `recommended_sequence_length_fast` in
`diagnostics.py`).

Quasistationärer Anteil (ratenbasiertes Maß, |u(t)-u(t-3τ)| < 0.05·Spanne, Median über
die vier Stellgrößen, `artifacts/runs/probe_excitation_bandwidth.json`, Abschnitt
`c1_base`): bei τ₁ 0.805-0.848, bei τ₂ 0.393-0.445 -- die Anregung verhält sich gegenüber
der schnellen Mode zu 80-85 % quasistationär. Von den geprüften Varianten bewegen
V1 (PRBS-Takt 50 s statt 120 s), V2 (Ratenlimit gelockert) und V3 (beides) diesen Anteil
nur innerhalb der Seed-Streuung; einzig V4 (Diagnosevariante, signal_weights nur PRBS,
Ratenlimit gelockert, keine Konstantläufe -- **absichtlich unrealistisch, kein
Designvorschlag**) verschiebt ihn sichtbar.

**Schlussfolgerung:** die schwache Anregung der schnellen Mode ist eine dokumentierte
Entwurfsgrenze (Ratenlimit + Δt), keine zufällige Schwäche des aktuellen Designs --
realistische Varianten der Anregung selbst ändern daran wenig. Offene Frage an Anwendungspartner:
reale Stellraten der Anlage, die die hier angesetzten Platzhalter-Ratenlimits ersetzen
müssten.

### 46. Verlassen des validierten Fluidisierungsfensters -- kein Austragsterm im Modell

Gegen `u_t(d50)` (der im Modell tatsächlich verwendete Bezugsdurchmesser, `greybox.py:393-395`
und `greybox.py:477-482`, unverändert): 26.6 % der 128 Läufe und 6.0 % der Abtastpunkte
verletzen `u > u_terminal`; 59.4 % dieser Verletzungen liegen im ersten Laufviertel, Median
des ersten Verletzungszeitpunkts 25 s -- also praktisch ab dem zweiten Abtastschritt
(`scripts/probe_f1_reference_diameter.py`, `scripts/probe_sweep_contamination.py`). Gegen
`u_t(d_Keim)` (nicht implementiert, nur zur Einordnung nachgerechnet) ist die Grenze dagegen
trivial **immer** verletzt (100 % der Läufe und Punkte) -- ein degenerierter, uninformativer
Bezugspunkt und Beleg dafür, dass `u_t(d50)` die einzig sinnvolle der beiden Varianten ist.

Im Zustandsvektor `(m_s, m_w, N, T_bed)` und den Bilanzgleichungen (`greybox.py:430-454`)
existiert kein Austrags-/Entrainment-Term; `u_mf`/`u_terminal` werden erst danach
(`greybox.py:477-482`) als reine Diagnosegrößen berechnet und wirken nicht auf die
Integration zurück. Dokumentiert war das bereits für den Fall "Anregung oberhalb
u_terminal" im Moduldocstring (`greybox.py:56-59`, "Der Austrag ist nicht modelliert, `u_t`
wird aber als Diagnose mitgeführt") -- der Docstring wurde in dieser Sitzung um den
d50-spezifischen Befund ergänzt (26.6 %/6.0 % s.o., Bezugsdurchmesser explizit benannt).

**Ergänzung (L1, `scripts/probe_l1_clip_sensitivity.py`): `max_driving_humidity_kg_per_kg`
ist selbst ein ungestützter Parameter.** Der Wert (Default 0.50 kg/kg,
`greybox_gab_placeholder.yaml`) ist ein Feld von `config.py:MaterialConfig`
(config.py:82-103), deren Klassendocstring selbst sagt "Stoffsystem ist noch TODO" --
keine Quellenangabe, keine Kalibrierung. Die Begrenzung wirkt in der rechten Seite der
Zustands-ODE (`thermo.evaporation_rate_kg_per_s`, `y_effective = minimum(y_saturation,
max_driving_humidity)`, thermo.py:199/272) und ist an der Knickstelle nicht
differenzierbar.

Über drei Werte {0.50, 1.00, 5.00 kg/kg} (0.50 = aktueller Default, 5.00 = praktisch nie
bindend) verstärkt die Klippung die Spitzentemperatur einzelner Läufe deutlich, aber nur
bei einer Minderheit der Läufe: gepoolt über die drei Trainingspool-Seeds (384 Läufe je
Konfiguration, jeweils gegen die 5.00-kg/kg-Referenz auf demselben Anregungsschema)
sind bei 0.50 kg/kg 219 Läufe betroffen (Median der Verschiebung 0.85 K, Maximum 13.3 K),
bei 1.00 kg/kg noch 135 Läufe (Median 2.4 K, Maximum 13.8 K). Der Anteil der Punkte mit
`bed_temperature_celsius` > 78.8 °C bleibt davon nahezu unberührt: 12.83 % bei 0.50 kg/kg
gegen 13.03 %/13.05 % bei 1.00/5.00 kg/kg -- die Klippung verstärkt bestehende
Temperaturspitzen, verursacht das Überschreiten der Schwelle aber nicht. Der
Klippungsanteil selbst (Anteil Punkte mit `Y_sat(T_bed) > max_driving_humidity`) fällt
dagegen stark mit dem Parameter: 12.83 % / 4.08 % / 0.00 %.

Schrittweitenkonvergenz (`numerics.substeps` 5→10→20, L2,
`scripts/probe_l2_step_convergence.py`) ist an drei Läufen mit hohen
`bed_temperature_celsius`-Spitzen nachweisbar schlechter als an drei Kontrollläufen
(maximale Abweichung 0.00029-0.00115 K gegen 0.0000016-0.00024 K, Verfeinerung nicht
monoton -- Ordnungsverlust durch die nicht differenzierbare Klippung), absolut aber
vernachlässigbar gegenüber dem 0.2-K-Sensorrauschen des Messmodells.

**Ergänzung: Projektion von `gas_velocity_m_per_s` auf `[u_mf(d50), u_terminal(d50)]`
implementiert, Default AUS.** `NumericsConfig.enforce_gas_velocity_projection`
(`config.py`) schaltet eine Sicherheitsprojektion ein, die die Gasgeschwindigkeit vor jedem
Integrationsschritt (`ProcessSimulator.rollout`, generischer Aufrufpunkt;
`GreyBoxFluidBedGranulator.project_action`/`_fluidization_margins`, stoffspezifische
Rechnung) auf das am WAHREN, unverrauschten Zustand berechnete Fenster clippt -- kein
Sicherheitsabstand (Faktor 1.0), kein anderer Kanal betroffen. Die
Zulufttemperatur (`inlet_temperature_celsius`) bleibt bewusst unbeschränkt: ihre
Betriebspunktvariation ist Gegenstand von Befund 49 und darf nicht mitgekappt werden.

Vorabmessung ohne Codeänderung (`scripts/probe_gas_velocity_projection.py`, 3
Trainingspool-Seeds + fixer Konstantlauf-Holdout, 479 232 Punkte gepoolt): die Projektion
senkt die Verletzung von 26.6 %/6.0 % (Läufe/Punkte, s. o.) auf eine Restverletzung von
0.16 % der Punkte, deren relative Überschreitung von `u_terminal` bei p95 = 0.17 % liegt
(Maximum 1.56 %) -- deutlich unter der Genauigkeit der Haider-&-Levenspiel-Korrelation
selbst (10-20 %, `fluidization.py`), also innerhalb von deren Modellunsicherheit statt ein
eigener physikalischer Sachverhalt. Ursache der Restverletzung ist keine falsch platzierte
Projektion, sondern eine strukturelle Grenze jedes Ein-Schritt-Vorgriffs unter
Zero-Order-Hold: die Projektion clippt auf `u_terminal` am Zustand VOR dem Schritt, `d50`
kann aber innerhalb der 25 s (5 RK4-Teilschritte) leicht sinken, wenn die Keimbildung die
Partikelzahl schneller erhöht als die Feststoffmasse wächst -- dann liegt die am
Schrittanfang korrekt geclippte Geschwindigkeit am Schrittende minimal über der (kleineren)
Grenze. Kürzung im Mittel 0.298 m/s (Maximum 1.161 m/s) an den 7.04 % betroffenen Punkten,
mit deutlicher Konzentration im ersten Laufviertel (48-59 % der Eingriffe je Gruppe) --
konsistent mit dem oben berichteten Median-Verletzungszeitpunkt von 25 s. Die Wirkung auf
nicht projizierte Kanäle bleibt klein: der Anteil der Punkte mit `bed_temperature_celsius`
> 78.8 °C fällt nur von 11.14 % auf 10.68 % (4.1 %, gepoolt) -- die Projektion schneidet die
Betriebspunktvariation aus Befund 49 nicht spürbar weg.

Der Schalter geht automatisch in `ProcessConfig.fingerprint` ein (dieser hasht die gesamte
Config): schon das Hinzufügen des Feldes ändert den Fingerprint der Standardkonfiguration
von 82f65f576a2e5253 auf 40aae2b4526a5ed7 (Schalter AUS, `sample_time_s=25 s`), bei
aktiviertem Schalter auf eb70e9c6d6590cf7. Bitgleichheit von altem und neuem Fingerprint
bei ausgeschalteter Projektion ist durch
`test_gas_velocity_projection_disabled_matches_legacy_baseline`
(`tests/test_gas_velocity_projection.py`) belegt. `Trajectory` trägt bei aktivierter
Projektion zusätzlich `requested_actions` (Aktion vor der Projektion), `projection_active`
und `n_projected_steps` je Lauf -- `actions` bleibt die realisierte, integrierte Aktion.

### 47. FIR-Lag-Tiefe der Stellgrößen-Historie -- Zielgröße bestimmt die nötige Sequenzlänge

Lag-Tiefen-Kurve (FIR: u(t),...,u(t-k), linear+quadratisch, kein Zielgrößen-Lag,
k∈{1,2,4,8,16,32,43}, In-Sample auf dem Trainingspool, Holdout auf dem fixen
Konstantlauf-Holdout, 3 Seeds, `scripts/probe_e2_e3_e4_lag_depth.py`,
`artifacts/runs/probe_e2_e3_e4_lag_depth.json`; konfigurierte `data/history_length` in
`arch_gab_constant_holdout.jsonl` und im `DataSpec`-Default, `experiments.py`: beide 32):

| Zielgröße | R²_holdout k=1 | k=32 | k=43 | Bild |
|---|---|---|---|---|
| `energy_target` | 0.873 | 0.859 | 0.857 | fällt leicht mit k -- kein Gedächtnis nötig, u(t) allein reicht |
| `bed_temperature_celsius` | 0.875 | 0.964 | 0.965 | sättigt bei k≈32 |
| `product_moisture_kg_per_kg` | 0.236 | 0.346 | 0.354 | steigt bei jedem k weiter, noch kein sichtbares Abflachen bis k=43 |
| `d50_um` | −0.042 | −0.106 | −0.123 | negativ und mit k fallend, siehe Befund 48/G5 |

**Nachtrag (AB2/AC1): Kelvin-Einordnung für `bed_temperature_celsius`, Quelle jetzt
verifiziert.** R²-Werte allein unterschätzen die praktische Bedeutung dieses Kanals, weil
Var(`bed_temperature_celsius`) über den Betriebsbereich groß ist -- das implizite Residuum
`sigma_resid = √Residuenvarianz` in Kelvin macht das sichtbar. Die ursprünglich fehlende
Quelle (siehe Commit-Historie: `scripts/probe_l1_clip_sensitivity.py` berichtete bisher nur
`product_moisture_kg_per_kg`) ist geschlossen -- das Skript ist auf alle vier
Simulator-Kanäle erweitert (unveränderte Split-Mechanik/Merkmalssatz/Punktauswahl,
`product_moisture_kg_per_kg`-Werte bitgleich zum vorherigen Stand, siehe Commit
"L1-Skript auf alle vier Kanäle erweitert"). Eigenes k-Raster ({1, 8, 43}, aus der
E3-Maskierungslogik übernommen) -- **k=8 statt k=32**, nicht direkt die Spalten der
R²-Tabelle oben, aber dieselbe Konstantlauf-Holdout-Population/Seeds/Merkmalslogik
(unmaskiert):

| k | Var(y) [K²] | Residuenvarianz [K²], Median [Min, Max] | Residuum [K], Median [Min, Max] | kW-Äquivalent (×8,35 kW/K) |
|---|---|---|---|---|
| 1 | 376,39 | 46,99 [41,95, 62,84] | 6,85 [6,48, 7,93] | 57,2 kW |
| 8 | 377,22 | 23,73 [20,81, 28,89] | 4,87 [4,56, 5,38] | 40,7 kW |
| 43 | 378,91 | 13,12 [12,76, 15,30] | 3,62 [3,57, 3,91] | 30,2 kW |

(`scripts/probe_l1_clip_sensitivity.py`, `artifacts/runs/probe_l1_clip_sensitivity.json`,
`results[0]` (`max_driving_humidity_kg_per_kg=0,50`, Default), `channels.
bed_temperature_celsius.unmasked_by_k`. Var(y) hat keine Spannweite über die Seeds -- sie
wird auf der Holdout-Zielgröße berechnet, die unabhängig vom Trainings-Seed ist; nur die
Residuenvarianz hängt vom angepassten Modell (und damit vom Seed) ab.) Gegenprobe: k=1
trifft die in X2 unabhängig **gemessene** statische Baseline nahezu exakt (RMSE ≈
6,85-6,87 K, Befund 50e) -- k=1 der FIR-Kurve und die quadratische statische Regression
ohne Historie liegen in derselben Größenordnung, wie erwartet. `1 − 46,99/376,39 = 0,8752`
trifft das geloggte `R²_holdout` bei k=1 (0,875) exakt.

Die formale Schwelle "kleinstes k mit |ΔR²_holdout| < 0.01 gegenüber k=43" liegt bei
`energy_target` 16, bei `bed_temperature_celsius` und `product_moisture_kg_per_kg` beide
bei 32. Für die Feuchte ist das eine Artefakt-Schwelle, kein echtes Plateau: die Kurve
steigt von k=1 bis k=43 durchgehend monoton (0.236→0.250→0.272→0.296→0.323→0.346→0.354),
ohne sichtbares Abflachen -- der 0.01-Schwellwert wird nur erreicht, weil die
Einzelschritte klein werden, nicht weil die Kurve konvergiert. `d50_um` erreicht
formal erst bei k=43 die Schwelle, aber nur weil R²_holdout durchgehend negativ bleibt
(Mittelwertsprädiktor schlägt jedes FIR-Modell).

**Ergänzung (M1): die zitierten Zeitkonstanten gelten für einen einzelnen
Linearisierungspunkt.** Die oben verlinkten `tau`-Werte (30.23/359.93/600.00/16000.00 s,
Tabelle weiter oben in dieser Datei) sind am exakten Beharrungszustand
(`solve_steady_state`) bei `bed_temperature_celsius` = 62.55 °C gemessen -- deutlich
unterhalb der Klippungsschwelle von Befund 46 (78.8 °C). An zwei weiteren
Beharrungszuständen (nur `inlet_temperature_celsius` variiert, sonst Nominalpunkt: 60 °C
-> 37.38 °C, feasible; 120 °C -> 87.79 °C, feasible) zeigt sich: `tau_3` (=600 s,
Feststoff-Regelkreis, `mass_control_time_constant_s`) und `tau_4` (=16000 s, geschlossene
Form `m_s,soll / (Sprührate*w_solid)`, Feststoff-Verweilzeit) bleiben exakt gleich --
beide sind Konfigurationskonstanten, unabhängig von der Betttemperatur, solange
Sprührate/Feststoffanteil/Bettmasse unverändert bleiben. Nur das aus (Wassermasse,
Betttemperatur) gekoppelte Teilsystem ändert sich mit dem Betriebspunkt: bei 37.38 °C
liegen die beiden übrigen Zeitkonstanten bei 37.00/1194.88 s, bei 87.79 °C bei
38.81/133.49 s.

Das Paar "30 s/360 s" aus dem Referenzpunkt existiert oberhalb der Klippungsschwelle also
**nicht mehr als Paar**: die schnelle Mode bleibt in derselben Größenordnung (30-39 s),
ihre Begleitmode fällt oberhalb der Schwelle von 360 s auf 133 s (dort ist die
Verdampfungsrate an der Bettemperatur konstant geklippt, die thermische
Selbstbegrenzung durchtrennt) und steigt umgekehrt deutlich unterhalb der Schwelle auf
1195 s. Die Modenstruktur -- nicht nur die Absolutwerte -- hängt damit vom Betriebspunkt
ab; eine Kalibrierung oder Validierung der Zeitkonstanten gegen Dyssol an nur einem
Betriebspunkt deckt das nicht ab.

**Ergänzung (N1/N2): die 3τ-Verankerung der Sequenzlänge gilt nur punktweise, nicht über
den Betriebsbereich.** Bei Δt=25 s entspricht die langsame Mode des gekoppelten
Teilsystems (Wassermasse, Betttemperatur) aus der M1-Ergänzung oben
`ceil(3τ/Δt)` = 144 Schritten bei 37.38 °C Betttemperatur (60 °C Eintritt, τ=1194.88 s),
44 Schritten am Referenzpunkt (62.55 °C, τ=359.93 s) und 17 Schritten bei 87.79 °C
(120 °C Eintritt, τ=133.49 s) -- exakte Werte aus `recommended_sequence_length_coupled_slow`
(`diagnostics.py:time_constant_report`, Befund N2 unten, per Wertevergleich als die zwei
Konfigurationskonstanten `mass_control`/`residence` ausgeschlossen und der verbleibende
gekoppelte Modenpaar-Anteil verwendet). Die konfigurierte `data/history_length` von 32
(`arch_gab_constant_holdout.jsonl`, `DataSpec`-Default, siehe oben) deckt damit das heiße
Ende ab, trifft den Referenzpunkt knapp (32 gegen nötige 44) und verfehlt das kalte Ende
um mehr als das Vierfache (32 gegen nötige 144). Das durchgehend fehlende Abflachen der
FIR-Lag-Kurve bei `product_moisture_kg_per_kg` bis k=43 (Tabelle oben, 0.236→0.250→0.272→
0.296→0.323→0.346→0.354, kein sichtbares Plateau) ist damit erklärbar: das
Anregungsschema deckt den gesamten Betriebsbereich ab (Populationsanteile Befund 49
unten), ein erheblicher Teil
der Trainings- und Holdout-Punkte liegt im kalten Regime, dessen relevante Zeitkonstante
mit keinem in dieser Studie getesteten k (bis 43) erreicht wird.

### 48. Feuchte-Dynamik konzentriert sich in der Betttemperaturgrenze, nicht in der Fluidisierungsverletzung -- revidiert durch K1 (CA2)

**Revision (CA2), Stand nach K1 (Befund 51).** Die ursprüngliche Fassung dieses Befunds
(unten, unverändert als historischer Stand belassen) unterstellte, die verletzende Region
zerfalle etwa hälftig in zwei UNABHÄNGIGE Teile, `u > u_terminal` und
`bed_temperature > 80 °C`. K1 (Befund 51) widerlegte diese Annahme bereits an den
Aggregatzahlen (Union sinkt nur 34 % statt der erwarteten 50 %, wenn `u > u_terminal`
fast vollständig verschwindet); dieser Nachtrag quantifiziert die Überlappung
komponentenweise (`scripts/probe_k1_violation_overlap.py`,
`artifacts/runs/probe_k1_violation_overlap.json`, dieselbe Punktpopulation wie K1: 3
Trainingspool-Seeds, Block `[:64]`, 221 184 Punkte je Konfiguration):

| Kriterium | unprojiziert | projiziert |
|---|---|---|
| `u > u_terminal` | 6,50 % | 0,21 % |
| `u < u_mf` | 0,00 % | 0,00 % |
| `bed_temperature_celsius > 80 °C` | 10,15 % | 9,73 % |
| Feuchte > kritisch | 1,05 % | 1,22 % |
| Union (irgendeines) | 16,61 % | 11,14 % |
| **>= 2 Kriterien gleichzeitig** | **1,10 %** | **0,02 %** |
| **NUR `bed_temperature_celsius > 80 °C`** | **9,06 %** | **9,71 %** |

(`u < u_mf` verletzt an keinem der 221 184 Punkte in beiden Konfigurationen -- von den
vier Kriterien aus `_violation_mask` trägt in der Praxis nur eines von dreien nennenswert
neben `bed_temperature_celsius`.) Unter Projektion (der für RL/Regelung relevante Fall)
sind **87,2 %** der verbleibenden Union-Verletzung (9,71 von 11,14 Prozentpunkten)
AUSSCHLIESSLICH `bed_temperature_celsius > 80 °C` -- keine andere Nebenbedingung verletzt
gleichzeitig. Das gleichzeitige Verletzen mehrerer Kriterien, das die
Halbierungserwartung stillschweigend voraussetzte, verschwindet unter Projektion fast
vollständig (1,10 % -> 0,02 %): `u > u_terminal` überlappte, statt eine unabhängige
zweite Hälfte zu sein, praktisch mit genau den Punkten, die ohnehin schon über
`bed_temperature_celsius` erfasst waren.

**Reformulierung:** der dynamische Spielraum der Feuchte-Dynamik (R²_holdout 0.354 -> 0.785
nach Maskierung, s. u.) konzentriert sich im WESENTLICHEN in einer Region, die durch die
Betttemperaturgrenze definiert ist, nicht durch die Verletzung des validierten
Fluidisierungsfensters.

**Das ist eine Entlastung, keine Verschärfung.** Die 80-°C-Grenze ist ein frei gesetzter
Platzhalter ohne Quelle (Befund 46; Literalkonstante `80.0` in `_violation_mask`, kein
Config-Feld, keine Kalibrierung) -- ihr Überschreiten ist kein Modellversagen, sondern
eine noch nicht belegte Zahl. `u > u_terminal` dagegen ist strukturell anders: im
Zustandsvektor fehlt ein Austrags-/Entrainment-Term (Befund 46), das Modell ist dort
PHYSIKALISCH UNVOLLSTÄNDIG, nicht nur unter einem unbelegten Schwellwert betrieben. Der an
mehreren Stellen dieses Dokuments mitgeführte Vorbehalt, der gemessene Architekturvorteil
bei `product_moisture_kg_per_kg` beruhe auf einer physikalisch ungültigen Datenregion,
trägt damit in dieser Form nicht mehr: die Region, in der das Signal sitzt, ist zu 87,2 %
eine Grenze ohne physikalischen Geltungsanspruch, nicht eine Region mit fehlender Physik.
Was bleibt: eine unbelegte Zahl (80 °C) bestimmt, welcher Teil der Feuchte-Dynamik als
"gültig" zählt -- das ist ein Kalibrierungs-Vorbehalt (die Grenze könnte höher oder
niedriger liegen), kein struktureller Modellmangel mehr.

**K1s Halbierungsvorhersage: widerlegt.** Das Prüfkriterium K1 (Befund 51) erwartete unter
Projektion eine Halbierung von dR2(k=43) auf `product_moisture_kg_per_kg` (~+0,2 gegenüber
+0,431), weil die Verletzungsregion als etwa hälftig `u > u_terminal`/`bed_temperature >
80 °C` angenommen wurde. Begründung des Fehlschlags: die Annahme setzte disjunkte
(unabhängige) Regionen voraus; tatsächlich überlappen sie stark (s. Tabelle oben) -- die
Projektion entfernt praktisch nur Punkte, die ohnehin schon über `bed_temperature_celsius`
erfasst waren, und lässt die Union entsprechend fast unverändert (dR2 bleibt bei +0,4205
gegenüber +0,431, keine Halbierung).

---

**Ursprüngliche Fassung (Stand vor K1/CA2), unverändert als historischer Befund belassen:**

FIR k=43, `product_moisture_kg_per_kg`: R²_holdout 0.354 unmaskiert gegen 0.785 nach
punktweisem Ausschluss der Verletzungspunkte (Kriterien wie Befund 46, d50-Basis,
`scripts/probe_e2_e3_e4_lag_depth.py`, Abschnitt `e3`). **Vorbehalt, der diese Zahl
begrenzt:** der Vergleich läuft über zwei verschiedene Stichproben -- 37.8 % der
Holdout-Punkte entfallen (13420 von 35488), und mit ihnen ändert sich auch Var(y), gegen
die R² normiert. Die Differenz ist ein Hinweis darauf, dass ein relevanter Teil des
lernbaren Signals in der Feuchte-Dynamik aus der Region kommt, die außerhalb des
validierten Fluidisierungsfensters liegt (Befund 46) -- **als offene Frage markiert, nicht
als quantifiziertes Ergebnis.**

**Ergänzung (L1): der Effekt ist unabhängig von `max_driving_humidity_kg_per_kg`.**
Dieselbe Maskierung (Kriterien wie oben) unter drei Werten des Klippungsparameters
{0.50, 1.00, 5.00 kg/kg} ergibt bei `product_moisture_kg_per_kg`, k=43 praktisch dieselbe
Differenz maskiert minus unmaskiert: +0.431 / +0.428 / +0.427
(`scripts/probe_l1_clip_sensitivity.py`, `artifacts/runs/probe_l1_clip_sensitivity.json`).
Der R²-Sprung ist also kein Artefakt der Klippung selbst -- er bleibt praktisch
unverändert bestehen, auch wenn diese kaum noch bindet (0.00 % Klippungsanteil bei
5.00 kg/kg, siehe Befund 46).

In physikalischen Einheiten (Default 0.50 kg/kg, Holdout, k=43): Var(y) fällt durch die
Maskierung von 0.0228 auf 0.00252 kg²/kg² (Faktor 9), die Residuenvarianz von 0.01473 auf
0.000542 kg²/kg² (Faktor 27). Der Gewinn ist damit nicht nur ein Normierungseffekt --
die Referenzvarianz schrumpft zwar auch, aber die Residuenvarianz fällt dreimal so stark.
**Vorbehalt, unverändert:** der Vergleich läuft über zwei verschiedene Stichproben
(maskiert entfernt 37.8 % der Holdout-Punkte, s.o.); Var(y) und Residuenvarianz sind
deshalb nicht direkt zwischen den beiden Stichproben vergleichbar, nur die jeweiligen
R²-Werte innerhalb ihrer eigenen Stichprobe.

**Ergänzung: Verweis auf Befund 49.** Befund 49 liefert eine zweite, von der
Verletzungsmaske unabhängige Erklärung für den hier beschriebenen R²-Sprung: der
Betriebspunkt selbst spaltet die Feuchte-Dynamik in zwei Regime mit unterschiedlicher
Modenstruktur auf. **Präzisierung:** die Region hoher Betttemperatur (HEISS, Befund 49)
und die Region der Verletzung des validierten Fluidisierungsfensters (Befund 46) fallen
überwiegend zusammen, ohne in dieser Untersuchung gemeinsam quantifiziert worden zu sein
-- sind aber verschiedene Sachverhalte: Befund 46 maskiert punktweise nach einer
physikalischen Zulässigkeitsbedingung (Fluidisierungsgrenze, Feuchtegrenze,
`bed_temperature_celsius` > 80 °C), Befund 49 klassifiziert punktweise ausschließlich
nach `bed_temperature_celsius` (> 70 °C / < 55 °C), ohne Bezug zu einer Gültigkeitsgrenze
und mit anderen Schwellwerten. Die beiden Masken sind nicht deckungsgleich (Befund 46:
6.0 % der Punkte, mehrere Kriterien; Befund 49: 36.7 %/44.5 % der Holdout-Punkte
heiß/kalt, 18.7 % Übergang, ein einziges Schwellenpaar) und sollten nicht gegeneinander
ausgetauscht werden.

### 49. Betriebspunktabhängigkeit der Modenstruktur als mechanistische Begründung für Sequenzmodelle

Befund 47/M1 zeigt an drei Beharrungszuständen (Eintrittstemperatur 60/90/120 °C ->
Betttemperatur 37.38/62.55/87.79 °C), dass die Modenstruktur des gekoppelten
Teilsystems (Wassermasse, Betttemperatur) vom Betriebspunkt abhängt: 37.00 s/1194.88 s,
30.23 s/359.93 s, 38.81 s/133.49 s -- die schnelle Mode bleibt bei 30-39 s, die
langsame variiert um Faktor 9 (133 bis 1195 s). Hypothese: ein einzelnes lineares
zeitinvariantes (LZI) Modell kann diese beiden Regime nicht gleichzeitig abbilden.
Geprüft an der Kreuztabelle unten.

Regime-Kreuztabelle (`scripts/probe_n1_regime_crosstab.py`,
`artifacts/runs/probe_n1_regime_crosstab.json`, fp `9c05902fa1363984` -- weicht von der
in mehreren älteren Skript-Docstrings zitierten `82f65f576a2e5253` ab, Konfigurationsdrift
zwischen Skript-Ständen, nicht Gegenstand dieser Untersuchung; Δt=25 s, 3
Trainingspool-Seeds, fixer Konstantlauf-Holdout wie Befund 47/48): punktweise
Regimezugehörigkeit über `bed_temperature_celsius`, getrennt von der Verletzungsmaske aus
Befund 46 -- HEISS > 70 °C, KALT < 55 °C, dazwischen verworfen. Populationsanteile:
Trainingspool 25.55 % heiß / 39.36 % kalt / 35.08 % Übergang, Holdout 36.73 % heiß /
44.54 % kalt / 18.74 % Übergang.

FIR (linear+quadratisch, k=43), drei Modelle (auf HEISS, auf KALT, auf HEISS+KALT
gepoolt gefittet) gegen zwei Auswertemengen (HEISS-Holdout, KALT-Holdout), Median über
3 Seeds, R² und Residuenvarianz (physikalische Einheiten):

| Zielgröße | HEISS-Holdout: lokal / gepoolt -- ResidVar (Faktor) | KALT-Holdout: lokal / gepoolt -- ResidVar (Faktor) |
|---|---|---|
| `energy_target` | 0.913 / 0.867 -- 0.0913 / 0.1389 (lokal 1.52×) | 0.812 / 0.788 -- 0.0884 / 0.0994 (lokal 1.13×) |
| `bed_temperature_celsius` | 0.941 / 0.943 -- 4.79 / 4.59 (gepoolt 1.04×) | 0.823 / 0.705 -- 11.50 / 19.16 (lokal 1.67×) |
| `product_moisture_kg_per_kg` | 0.630 / −23.46 -- 1.075e-5 / 7.108e-4 (lokal 66×) | 0.267 / 0.214 -- 0.02848 / 0.03054 (lokal 1.07×) |
| `d50_um` | −0.041 / −0.221 (beide negativ) | −0.300 / −0.088 (beide negativ, gepoolt weniger schlecht) |

Kernfrage in einem Satz -- schlägt das lokal gefittete Modell das gepoolte auf beiden
Auswertemengen, und um wie viel in Residuenvarianz: **uneinheitlich über die Kanäle.**
Für `product_moisture_kg_per_kg` -- die Zielgröße aus Befund 47/48 -- ist der Effekt
eindeutig und an einer Stelle drastisch: auf HEISS-Holdout senkt das lokale Modell die
Residuenvarianz um Faktor 66 gegenüber dem gepoolten. In Gegenrichtung bricht ein auf
KALT gefittetes Modell auf HEISS-Holdout vollständig zusammen (R² Median −50.7, Spanne
−931.7 bis −38.9 über die 3 Seeds; Residuenvarianz 0.0015 kg²/kg² gegenüber einer
Zielvarianz von nur 2.9e-5 kg²/kg² -- 51-fach über der Referenzvarianz). Für
`bed_temperature_celsius` ist der lokale Vorteil auf KALT deutlich (1.67×), auf HEISS
dagegen nicht vorhanden (das gepoolte Modell liegt dort minimal vorn, 1.04×). Für
`energy_target` ist der lokale Vorteil in beide Richtungen vorhanden, aber moderat
(1.13-1.52×). Für `d50_um` bleibt R² in allen sechs Zellen negativ -- kein belastbares
Signal, konsistent mit Befund 47/48.

Zusatz bei k=143 (≈ 3 × 1194.9 s / 25 s ≈ 143.4, als runder Wert getestet; die exakte
`ceil(3τ/Δt)` liegt bei 144, s. N2 unten -- Prüfung auf Sättigung des kalten Regimes bei
ausreichend langem Fenster), nur Zelle KALT gefittet/KALT ausgewertet:
`energy_target` 0.769 (gegen 0.812 bei k=43 -- kein Sättigen, eher Rückgang),
`bed_temperature_celsius` 0.816 (gegen 0.823 -- innerhalb der Seed-Spannweite, praktisch
gesättigt), `product_moisture_kg_per_kg` 0.296 (gegen 0.267, Spannweite [0.236, 0.557]
über die Seeds -- kein klares Sättigen erkennbar), `d50_um` −0.689 (gegen −0.300 --
weiterhin negativ, wird schlechter).

**Vorbehalt:** die HEISS- und KALT-Auswertemengen unterscheiden sich in Var(y) (s. Tabelle
im JSON-Artefakt), R² ist deshalb nur innerhalb derselben Auswertemenge zwischen den drei
Fit-Varianten vergleichbar, nicht zwischen HEISS- und KALT-Spalte. Aus demselben Grund ist
die Kreuzvorhersage-Zelle (KALT gefittet, HEISS ausgewertet, `product_moisture_kg_per_kg`)
mit Vorsicht zu lesen: die extrem negativen R²-Werte kommen auch daher, dass die
Zielvarianz im HEISS-Regime sehr klein ist (2.9e-5 kg²/kg²) -- ein an sich moderater
Absolutfehler wird dadurch relativ riesig. Die Residuenvarianz selbst (0.0015 kg²/kg²)
bleibt aber auch in physikalischen Einheiten mehr als eine Größenordnung über der
ResidVar des lokalen HEISS-Modells (1.075e-5) und über der Zielvarianz beider Regime --
der Kollaps ist kein reines Normierungsartefakt.

**Ergänzung P1/P2: arbeitspunktgeplantes FIR-Modell (GEPLANT) gegen eine
Merkmalszahl-Kontrolle (CONTROL).** Offene Frage nach dem N1-Befund oben:
Arbeitspunktabhängigkeit der Parameter (dann genügt ein arbeitspunktgeplantes lineares
Modell) oder echte Zustandsabhängigkeit der Dynamik (dann Begründung für
Sequenzmodelle)? Geprüft mit zwei zusätzlichen, gepoolt gefitteten FIR-Modellen
(`scripts/probe_p1_p2_scheduled_fir.py`, `artifacts/runs/probe_p1_p2_scheduled_fir.json`,
gleicher fp `9c05902fa1363984`, gleiche Split-Mechanik/Konstantlauf-Holdout/3 Seeds wie
N1; POOLED und LOKAL sind aus N1 unverändert wiederverwendet, bitgleich mit der Tabelle
oben):

- **GEPLANT** (k=43): wie POOLED, aber der Merkmalssatz ist zusätzlich um die
  Wechselwirkung jedes linearen/quadratischen FIR-Merkmals mit der gemessenen
  `bed_temperature_celsius(t)` sowie um `bed_temperature_celsius(t)` selbst als
  Haupteffekt erweitert -- die Merkmalszahl verdoppelt sich dadurch exakt (353 -> 706).
- **CONTROL** (k=87, nächstliegend zur GEPLANT-Merkmalszahl: 705 Merkmale statt 706):
  POOLED-Modell ohne jeden Betriebspunktbezug, aber mit vergleichbarer Merkmalszahl --
  prüft, ob ein GEPLANT-Gewinn reine Merkmalszahl statt Arbeitspunktbezug ist. (n_points
  bei CONTROL geringfügig kleiner als bei k=43, 12780/14910 statt 13287/15591 -- Folge
  des größeren k, das pro Lauf mehr Anfangspunkte für die Lag-Fenster verbraucht, kein
  Fehler.)

**Vorbehalt zu GEPLANT:** `bed_temperature_celsius(t)` ist für `energy_target`,
`product_moisture_kg_per_kg` und `d50_um` ein anderer Ausgangskanal als das jeweilige
Ziel -- keine autoregressive Rückführung des Zielkanals. Für das Ziel
`bed_temperature_celsius` selbst ist die Konstruktion dagegen autoregressiv (das Modell
bekommt den Zielwert zum Vorhersagezeitpunkt direkt als Merkmal); die Zeile ist deshalb
unten markiert und nicht interpretierbar (R²=1.000, ResidVar ≈ 1e-22 auf beiden
Auswertemengen -- Leckage-Artefakt, kein Arbeitspunktbefund).

R² (Median über 3 Seeds):

| Zielgröße | HEISS: GEPLANT/POOLED/LOKAL/CONTROL | KALT: GEPLANT/POOLED/LOKAL/CONTROL |
|---|---|---|
| `energy_target` | 0.908 / 0.867 / 0.913 / 0.867 | 0.849 / 0.788 / 0.812 / 0.773 |
| `bed_temperature_celsius`¹ | 1.000 / 0.943 / 0.941 / 0.941 | 1.000 / 0.705 / 0.823 / 0.693 |
| `product_moisture_kg_per_kg` | −14.767 / −23.464 / 0.630 / −26.692 | 0.355 / 0.214 / 0.267 / 0.227 |
| `d50_um` | −0.217 / −0.221 / −0.041 / −0.246 | −0.061 / −0.088 / −0.300 / −0.105 |

¹ autoregressiv/Leckage, s. Vorbehalt oben -- nicht vergleichbar mit den anderen Zeilen.

Residuenvarianz, physikalische Einheiten (Median über 3 Seeds):

| Zielgröße | HEISS: GEPLANT/POOLED/LOKAL/CONTROL | KALT: GEPLANT/POOLED/LOKAL/CONTROL |
|---|---|---|
| `energy_target` | 0.0964 / 0.1389 / 0.0913 / 0.1388 | 0.0711 / 0.0994 / 0.0884 / 0.1068 |
| `bed_temperature_celsius`¹ | 1.7e-22 / 4.587 / 4.792 / 4.774 | 1.3e-22 / 19.16 / 11.50 / 20.05 |
| `product_moisture_kg_per_kg` | 4.581e-4 / 7.108e-4 / 1.075e-5 / 8.022e-4 | 0.02504 / 0.03054 / 0.02848 / 0.03118 |
| `d50_um` | 21649 / 21719 / 18522 / 21617 | 21042 / 21567 / 25769 / 21288 |

**Kernfrage `product_moisture_kg_per_kg`, HEISS-Holdout (die Zelle mit dem Faktor-66-
Befund oben):** GEPLANT senkt die Residuenvarianz gegenüber POOLED von 7.108e-4 auf
4.581e-4 kg²/kg² -- ein Faktor 1.55× -- und schließt damit **36.1 %** der Lücke zu LOKAL
(1.075e-5) in Residuenvarianz. R² bleibt dabei mit −14.767 tief negativ (gegenüber LOKAL
+0.630, POOLED −23.464) -- GEPLANT ist auf dieser Zelle nach wie vor weit von einer
brauchbaren Vorhersage über die Regimegrenze entfernt. CONTROL reproduziert den
GEPLANT-Gewinn nicht: mit vergleichbarer Merkmalszahl (705) wird das Modell auf HEISS
sogar schlechter als POOLED (R² −26.692 gegenüber −23.464, ResidVar 8.022e-4 gegenüber
7.108e-4). Der GEPLANT-Gewinn ist damit keine reine Merkmalszahl-Überanpassung, sondern
an die Betriebspunkt-Interaktion gebunden -- aber er ist klein gegenüber der
verbleibenden Lücke.

Auf KALT-Holdout ist das Bild anders: GEPLANT übertrifft dort sogar LOKAL (R² 0.355
gegenüber 0.267), CONTROL bleibt nahe POOLED (0.227 gegenüber 0.214). Für `energy_target`
liegt GEPLANT auf HEISS knapp unter LOKAL (0.908 gegenüber 0.913) und auf KALT über LOKAL
(0.849 gegenüber 0.812); CONTROL liegt in beiden Fällen nahe POOLED -- kein
Merkmalszahl-Effekt erkennbar. Für `d50_um` bleibt R² in allen vier Modellen negativ,
kein belastbares Signal.

**Einordnung für `product_moisture_kg_per_kg`:** die Region mit dem drastischsten
POOLED-Versagen (HEISS-Holdout, Faktor 66 gegenüber LOKAL) zeigt ein reales, durch
CONTROL nicht erklärbares, aber kleines arbeitspunktlineares Signal (36 % der
Residuenvarianz-Lücke, s. Vorbehalt unten). Der weit überwiegende Teil der Lücke zu LOKAL
bleibt durch eine lineare Arbeitspunktplanung unerklärt.

**Ergänzung Q1/Q2: drei Präzisierungen zur GEPLANT/CONTROL-Auswertung oben.**

a) **Die 36,1-%-Zahl ist gegen eine Divergenz gemessen, nur eingeschränkt aussagekräftig.**
R² liegt bei POOLED (−23,464) und GEPLANT (−14,767) auf HEISS beide weit im divergenten
Bereich (schlechter als die Konstante Ø -- R²<0 bedeutet per Definition schlechter als der
Mittelwertschätzer). Ein Anteil "36 % der Lücke geschlossen" zwischen zwei so tief
negativen Werten beschreibt eine reale, aber kleine Verschiebung innerhalb des
Versagensbereichs, keine Bewegung in Richtung Brauchbarkeit. Belastbar ist nur der
Vergleichsbefund aus P2: GEPLANT ist gegenüber POOLED nachweislich besser als reine
Merkmalszahl (CONTROL reproduziert den Effekt nicht, wird sogar schlechter), erreicht
aber auf HEISS keine Brauchbarkeit (R² bleibt tief negativ).

b) **Regime-Asymmetrie -- offene Frage, kein erklärtes Ergebnis.** Auf KALT übertrifft
GEPLANT sogar LOKAL (R² 0,355 vs. 0,267), auf HEISS versagt es (R² −14,767). Die
betriebspunktabhängigen Zeitkonstanten der langsamen Mode (M1/N2 oben,
`recommended_sequence_length_coupled_slow`) liefern dazu einen möglichen, aber nicht
geprüften Erklärungsansatz: bei Δt=25 s und `ceil(3τ/Δt)` entspricht die langsame Mode
144 Schritten im kalten Referenzpunkt (37,38 °C, τ=1194,88 s) gegenüber 17 Schritten im
heißen (87,79 °C, τ=133,49 s). Bei k=43 ist das kalte Regime damit strukturell
unterparametrisiert (43 gegen benötigte 144 Schritte), während k=43 die heiße Zeitkonstante
mehr als abdeckt (43 gegen benötigte 17). Das eine Regime scheitert also möglicherweise an
zu wenig Lag-Tiefe, das andere trotz ausreichender Lag-Tiefe -- zwei verschiedene
Ursachen, nicht eine gemeinsame. Diese Zuordnung ist nicht geprüft (kein
k-Sweep pro Regime in dieser Untersuchung) und wird hier als offene Frage vermerkt, nicht
als Befund.

c) **Q1-Konditionierungskontrolle: Entwarnung.** Die Trainings-Entwurfsmatrizen sind
schlecht konditioniert (Konditionszahl POOLED/CONTROL ≈ 5–12·10⁶, GEPLANT ≈ 1,4–2,2·10⁹;
Verhältnis größte/kleinste Spaltennorm bis 7,2·10⁵ bei GEPLANT -- die quadratischen
FIR-Terme und deren Wechselwirkung mit `bed_temperature_celsius(t)` liegen um
Größenordnungen über den linearen Termen). Alle drei Matrizen haben dabei vollen Rang
(`np.linalg.matrix_rank` = von `lstsq` gemeldeter Rang = Merkmalszahl, je Seed, keine
Rangdefizienz). Eine Kontrollrechnung mit spaltenweiser Standardisierung (Mittel 0, Std 1
je Trainings-Seed, Bias-Spalte ausgenommen) senkt die Konditionszahl um 3–4
Größenordnungen (Beispiel POOLED/Seed 1: 8,1·10⁶ → 2,0·10³) und das Spaltennorm-Verhältnis
auf ≈1 -- ändert aber R² und Residuenvarianz auf beiden Holdouts und allen vier Kanälen
nicht messbar (Abweichung der Vorhersagen gegenüber unstandardisiert < 10⁻¹¹ relativ zur
Zielgrößen-Skala, Beispielrechnung `product_moisture_kg_per_kg`). Das ist der erwartete
Befund für eine exakte OLS-Lösung: Spaltenstandardisierung ist eine invertierbare
Reparametrisierung, die Vorhersagefunktion ändert sich nicht. Das tief negative R² auf
HEISS ist damit **kein Konditionierungsartefakt**, sondern eine strukturelle Eigenschaft
der angepassten Modelle -- Entwarnung für die Zahlen oben, kein Vorbehalt. (Skript/JSON:
`scripts/probe_p1_p2_scheduled_fir.py`, Schlüssel `conditioning` und
`standardized_control_check` in `artifacts/runs/probe_p1_p2_scheduled_fir.json`.)

### 50. Ablufttemperatur ist Anwendungsgröße, nicht Architekturvergleichskanal -- eigenes Protokoll mit statischer Pflichtbaseline und Kelvin-Primärmetrik

**a) Begründung, warum ein eigenes Protokoll.** Die Ablufttemperatur ist zu 76,8 %
statisch erklärbar (R²_stat = 0,7682, `bed_temperature_celsius`,
`c0/r2_stat_full_sample_reference/bed_temperature_celsius/median`,
`artifacts/runs/probe_excitation_bandwidth.json`) und eine reine FIR-Regression über die
Stellgrößen-Historie (kein Zielgrößen-Lag, `k=43`) erreicht bereits R²_holdout = 0,965
(Befund 47, Tabelle dort). Unter der in diesem Projekt sonst durchgehend verwendeten
Persistenz-Baseline sieht jede Architektur an diesem Kanal gut aus -- die eigentliche
Hürde ist die statische Regression, nicht Persistenz. Ein gemeinsamer
Architekturvergleichs-Sweep (wie Befund 28/36/40) würde diese Kanaleigenschaft mit
Modellgüte verwechseln. Deshalb: ein eigenes, schlankes Protokoll (eine Architektur, drei
Seeds), mit statischer Regression als Pflichtbaseline und absolutem Fehler in Kelvin als
Primärmetrik -- nicht NRMSE, weil die Verbundschnittstelle (Wärmeverbund) eine physikalische
Zahl in Kelvin braucht, keine normierte.

**b) Abgrenzung zu Befund 42 -- keine Vorhersagelücke, sondern fehlende Metrik/Baseline/
Einordnung.** Vor diesem Befund lag bereits ein voller Sweep vor: Befund 42,
`artifacts/runs/befund42_tbed_exhaust_sweep.jsonl`, GRU **und** LSTM × 10 Seeds × 2
Testmengen = 40 Records, `bed_temperature_celsius` als vorhergesagter Kanal, End-zu-End
gegen `T_gas_out_celsius`/`Y_gas_out_kg_per_kg` rekonstruiert, mit dem Ergebnis "GRU/LSTM
bleiben auch im ungünstigsten Fall klar über Persistenz" (Befund 42d). Befund 50 schließt
also **keine** Vorhersagelücke -- die gab es nicht. Was fehlte:

- **Kein benannter Abluftkanal.** `outlet_temperature_celsius`/`ExhaustConfig` gab es erst
  mit X1 (dieser Commit-Zweig) -- Befund 42 arbeitete mit `bed_temperature_celsius`/
  `T_gas_out_celsius`, ohne den Rohrleitungs-/Filterverlust-Platzhalter, ohne den
  Vertragsnamen, den der Wärmeverbund erwartet.
- **Keine statische Pflichtbaseline.** Befund 42 verglich ausschließlich gegen Persistenz.
  Bei R²_stat = 0,768 (a) ist das keine aussagekräftige Hürde -- ein Vergleich nur gegen
  Persistenz lässt jede Architektur an diesem Kanal besser aussehen, als sie gegenüber der
  eigentlich bindenden Referenz ist.
- **Keine Primärmetrik in Kelvin.** `EvaluationResult.rmse_run_median_physical` existiert
  erst seit G3 (`tests/test_pipeline.py::test_existing_nrmse_keys_unchanged_by_rmse_
  addition`) -- Befund 42 berichtete ausschließlich NRMSE, keine absolute Kelvin-Zahl.
- **Keine Einordnung in rückgewinnbare Leistung.** Kein Bezug zu kW oder zur
  446,6-kW-Referenzlinie der Wirbelschichtlinie.

Befund 50 ergänzt damit Metrik, Baseline und Verbundbezug -- er stellt die
Vorhersagegüte selbst nicht neu fest (das hat Befund 42 bereits getan) und ist insofern
eine schwächere, aber korrekte Aussage als ursprünglich in diesem Dokument entworfen.

**c) X1 (Voraussetzung): `ExhaustConfig`, `outlet_temperature_celsius`.** Neuer,
eigener Config-Abschnitt `ExhaustConfig.outlet_duct_heat_loss_k` (Kelvin, Default 0,0 --
Identität zu `bed_temperature_celsius`, Platzhalter ohne Quelle).
`simulation.exhaust.reconstruct_exhaust_state()` liefert seither zusätzlich
`outlet_temperature_celsius` als eigenes, korrekt benanntes Feld. Fingerprint von
`greybox_gab_placeholder.yaml` (Projektion AUS): `078572b3a7a0fb45` (siehe
Fingerprint-Historie oben).

**d) X2 -- Protokoll und Verdrahtungsprüfungen.** GRU, drei Seeds (1, 2, 3), `n_train_max=64`
(Konvention aus `arch_gab_constant_holdout.jsonl`, Befund 36), konstantes Holdout
(`HOLDOUT_SEED=999`), Pflichtbaselines Persistenz und statische Regression
(`scripts/run_outlet_temperature_gru_protocol.py`, `artifacts/runs/
outlet_temperature_gru_protocol.jsonl`, 18 Records). `outlet_temperature_celsius` kann
NICHT direkt als `DataSpec.target_names`-Eintrag verwendet werden (Simulator-Output-Menge
ist fest, vier Kanäle, siehe Skript-Docstring) -- trainiert wird gegen
`bed_temperature_celsius`, jede Vorhersage zusätzlich durch die (um `outlet_temperature_
celsius` erweiterte) `evaluate_exhaust_reconstruction()` geschickt und dort unter dem
Abluftnamen berichtet. Alle Verdrahtungsprüfungen bestanden: Fingerprint aller drei Seeds
== `078572b3a7a0fb45`; `torch/num_threads`/`env/mkl_num_threads`/`env/omp_num_threads`
(=1) und `model/net_actual_device` (`cpu`) im ersten Baseline- bzw. GRU-Record belegt;
`T_gas_out_celsius` == `outlet_temperature_celsius` bitgleich in allen 18 Records × 2
Horizonten (36 Vergleiche, exakter `==`-Vergleich) -- erwartungsgemäß bei
`outlet_duct_heat_loss_k=0,0`.

**e) X2 -- Ergebnisse, Median [Min,Max] über 3 Seeds.** `outlet_temperature_celsius`,
Primärmetrik RMSE in Kelvin, sekundär NRMSE, zusätzlich das kW-Äquivalent des absoluten
Fehlers (8,35 kW/K) und dessen Anteil an der 446,6-kW-Referenzlinie:

Testmenge "angeregt" (`excited`):

| Modell | h | RMSE [K] | NRMSE | kW-Äquivalent | Anteil 446,6 kW |
|---|---|---|---|---|---|
| Persistenz | 1 | 0,789 [0,726, 0,821] | 0,125 [0,116, 0,130] | 6,58 kW | 1,47 % |
| Statisch | 1 | 6,848 [6,327, 6,931] | 0,786 [0,755, 0,888] | 57,18 kW | 12,80 % |
| GRU | 1 | 0,302 [0,298, 0,304] | 0,041 [0,040, 0,044] | 2,53 kW | 0,57 % |
| Persistenz | 36 | 6,532 [6,093, 6,568] | 0,920 [0,835, 0,940] | 54,54 kW | 12,21 % |
| Statisch | 36 | 6,820 [6,327, 6,833] | 0,791 [0,760, 0,900] | 56,95 kW | 12,75 % |
| GRU | 36 | 0,183 [0,162, 0,208] | 0,026 [0,025, 0,032] | 1,53 kW | 0,34 % |

Testmenge Konstantlauf-Holdout:

| Modell | h | RMSE [K] | NRMSE | kW-Äquivalent | Anteil 446,6 kW |
|---|---|---|---|---|---|
| Persistenz | 1 | 0,351 [0,351, 0,351] | 0,631 [0,631, 0,631] | 2,93 kW | 0,66 % |
| Statisch | 1 | 5,940 [5,868, 6,262] | 9,527 [8,984, 9,556] | 49,60 kW | 11,11 % |
| GRU | 1 | 0,278 [0,271, 0,280] | 0,483 [0,479, 0,490] | 2,32 kW | 0,52 % |
| Persistenz | 36 | 0,487 [0,487, 0,487] | 1,158 [1,158, 1,158] | 4,07 kW | 0,91 % |
| Statisch | 36 | 5,884 [5,816, 6,175] | 10,697 [10,164, 11,038] | 49,13 kW | 11,00 % |
| GRU | 36 | 0,131 [0,126, 0,171] | 0,333 [0,276, 0,338] | 1,10 kW | 0,25 % |

GRU liegt in Kelvin absolut bei 0,13-0,30 K RMSE, gegenüber 5,9-6,8 K bei der statischen
Baseline -- die statische Regression ist hier, anders als bei `energy_target`
(R²_stat = 98,7 %, Befund 2), keine belastbare Referenz für diesen Kanal: ihr absoluter
Fehler bleibt über alle Horizonte/Testmengen etwa konstant bei 6 K (kein Lernen aus der
Historie), während Persistenz auf der angeregten Testmenge bei h=36 mit GRU vergleichbar
schlecht wird (6,5 K) und auf dem Konstantlauf-Holdout deutlich besser bleibt als
Statisch. Zum Vergleich `energy_target`/`product_moisture_kg_per_kg` (RMSE physikalisch,
Median über 3 Seeds, angeregte Testmenge, h=36): GRU 0,0149 kW / 0,00051 kg/kg gegenüber
Statisch 0,1163 kW / 0,0245 kg/kg -- GRU bleibt an allen drei Zielgrößen klar vor beiden
Baselines, konsistent mit Befund 42.

**f) Kostenvorbehalt.** `training/elapsed_s` (GRU, drei Zielköpfe): 506,7 s / 619,3 s /
544,4 s (Median 544,4 s), gemessen mit `MKL_NUM_THREADS`/`OMP_NUM_THREADS=1` als
Prozessumgebungsvariablen (Befund 43g). Kein kontrollierter Vergleich gegen einen
identisch konfigurierten Zwei-Zielkopf-Lauf in diesem Protokoll -- der vorab geschätzte
"~50 % mehr Trainingszeit"-Aufschlag für den dritten Zielkopf ist mit diesen Zahlen
weder bestätigt noch widerlegt, nur die absolute Kostenzahl ist belastbar.

**g) X3 -- rückgewinnbare Leistung Q_rec aus vorhergesagten Kanälen.**
:math:`\dot Q_{rec} = \dot m_L\,(T_{out} - (T_{dp}(Y_{out}) + 3\,\mathrm K))\,(c_{p,L} +
Y_{out}\,c_{p,D})`, ausgewertet aus den VORHERGESAGTEN `outlet_temperature_celsius`/
`Y_gas_out_kg_per_kg` gegen dieselben, aus der Wahrheit rekonstruierten Größen
(`evaluation/recoverable_power.py`, `scripts/probe_x3_recoverable_power.py`,
`artifacts/runs/probe_x3_recoverable_power.json`, 18 Records, GRU-Checkpoints aus X2
geladen, kein erneutes Training). Taupunkt über dieselbe Buck-1996-Korrelation wie überall
im Projekt (`thermo.saturation_pressure_pa`, vektorisiert in `thermo.dew_point_celsius`,
gegen den bestehenden skalaren Löser kreuzgeprüft). Median [Min,Max] über 3 Seeds:

Testmenge "angeregt":

| Modell | h | NRMSE | RMSE [kW] | Anteil 446,6 kW |
|---|---|---|---|---|
| Persistenz | 1 | 0,096 [0,086, 0,102] | 5,48 [4,88, 6,17] | 1,23 % |
| Statisch | 1 | 1,181 [1,031, 1,460] | 75,54 [74,62, 105,25] | 16,91 % |
| GRU | 1 | 0,020 [0,020, 0,024] | 1,37 [1,35, 1,48] | 0,31 % |
| Persistenz | 36 | 0,814 [0,783, 0,841] | 58,20 [50,45, 58,81] | 13,03 % |
| Statisch | 36 | 1,199 [1,049, 1,495] | 75,67 [75,00, 104,72] | 16,94 % |
| GRU | 36 | 0,027 [0,024, 0,031] | 1,80 [1,65, 1,94] | 0,40 % |

Testmenge Konstantlauf-Holdout:

| Modell | h | NRMSE | RMSE [kW] | Anteil 446,6 kW |
|---|---|---|---|---|
| Persistenz | 1 | 0,177 [0,177, 0,177] | 1,53 [1,53, 1,53] | 0,34 % |
| Statisch | 1 | 7,132 [7,038, 8,958] | 55,34 [54,55, 74,30] | 12,39 % |
| GRU | 1 | 0,138 [0,137, 0,140] | 1,20 [1,20, 1,21] | 0,27 % |
| Persistenz | 36 | 0,563 [0,563, 0,563] | 3,69 [3,69, 3,69] | 0,83 % |
| Statisch | 36 | 8,421 [8,151, 9,301] | 55,52 [55,46, 73,73] | 12,43 % |
| GRU | 36 | 0,250 [0,201, 0,256] | 1,49 [1,40, 1,64] | 0,33 % |

Q_rec bei GRU liegt bei 0,25-0,40 % der 446,6-kW-Referenzlinie -- in derselben
Größenordnung wie der lineare kW-Äquivalent-Fehler aus (e), keine erkennbare Verstärkung
durch die Nichtlinearität der Q_rec-Formel bei diesem Fehlerniveau.

**h) Massenstrom-Vorbehalt.** `DISTRICT_MASS_FLOW_KG_PER_S = 8,0` kg/s
(`evaluation/recoverable_power.py`) ist der Massenstrom des Wärmeverbundmodells, zwei
Größenordnungen über dem simulierten Labormassenstrom (`dry_air_mass_flow_kg_per_s`,
Größenordnung 10⁻²-10⁻¹ kg/s in dieser Konfiguration). Alle kW-Zahlen in (e) und (g) sind
eine Skalierungsrechnung -- sie beantworten "wie groß wäre der Fehler bei diesem
Temperatur-/Feuchte-Vorhersagefehler, wenn eine Anlage mit dem Verbund-Massenstrom liefe",
keine Aussage über die tatsächliche Laborwirbelschicht.

**i) Nachtrag (AB1): `h36 < h1` bei GRU bleibt auch gegen den rauschfreien Zustand
bestehen -- offene Frage, nicht erklärt.** Abschnitt (e) berichtet für GRU einen
kleineren RMSE bei h=36 als bei h=1 (0,30→0,18 K angeregt, 0,28→0,13 K Konstantlauf).
Geprüft gegen `nrmse_truth_run_median`/`exhaust_rmse_run_median_physical` (beide bereits
gegen die rauschfreie Wahrheit ausgewertet, nicht gegen die verrauschte Messung -- exakte
Übereinstimmung mit `exhaust_nrmse_run_median/T_gas_out_celsius` bestätigt, 0 Abweichungen
über alle 36 Vergleiche): das Muster bleibt bestehen (angeregt: 0,041→0,026 NRMSE,
0,30→0,18 K; Konstantlauf: 0,48→0,33 NRMSE, 0,28→0,13 K). Die eingangs vermutete
Filter-Erklärung (Modell koppelt bei h=1 an die verrauschte letzte Messung, filtert das
Rauschen später über den rekurrenten Zustand heraus) ist damit **widerlegt** -- der Effekt
tritt unabhängig vom Messrauschen auf.

Bei den Baselines fehlt das Muster: Persistenz zeigt durchgehend `h36 > h1` (physikalisch
erwartet, keine gelernte Dynamik), die statische Regression liegt nahezu flach
(`bed_temperature_celsius`, angeregt: 6,85→6,82 K -- RMSE minimal fallend, NRMSE dagegen
leicht steigend, 0,786→0,791 -- kein einheitliches Bild). Das Muster ist damit
**spezifisch für GRU**, nicht ein allgemeines Artefakt der Metrik.

`rmse_run_median_physical`/`nrmse_*_run_median` bei h=36 ist -- wie in Befund 42f für die
NRMSE bereits festgehalten -- eine kumulative Größe über die Schritte 1..36
(`per_run_rmse_median`/`per_run_nrmse`, `evaluator.py`, Slice `[:horizon]`), keine
Punktschätzung am 36. Schritt. Zusätzliche Horizonte aus dem bestehenden Log
(`exhaust_rmse_run_median_physical/T_gas_out_celsius`, h=1/9/18/36, GRU, angeregt):
0,3025 K → 0,2127 K → 0,1941 K → 0,1833 K -- monoton fallend, nicht nur ein
Zwei-Punkt-Effekt.

**Nachtrag (AC2): Fensterpopulation geprüft und als Ursache widerlegt.** Naheliegendste
Hypothese war, dass mit wachsendem Horizont Fenster am Laufende wegfallen (ein Fenster
braucht `history_length + horizon` Schritte) und der bei h=1 berichtete Fehler dadurch auf
einer größeren, potenziell schwereren Population sitzt als der bei h=36 automatisch
mitverwendete. Geprüft (kein Training -- GRU-Checkpoints aus X2 geladen, Persistenz/
statische Baseline frisch angepasst, `scripts/probe_ac2_h1_window_population.py`,
`artifacts/runs/probe_ac2_h1_window_population.json`, 18 Records): pro Testpool zwei
Fenstersets gebaut -- eines mit `horizon=36` (wie X2, 4352 Fenster: 32 Läufe × 136
Fensteranfänge) und eines mit `horizon=1` (4480 Fenster: 32 Läufe × 140
Fensteranfänge, +128 zusätzliche Fenster nahe Laufende, die bei `horizon=36` keinen
Platz mehr haben). Da beide Startmengen `arange(0, n_steps−span+1, stride)` mit
demselben Ursprung/Stride verwenden, ist die kleinere Menge exakt eine Teilmenge der
größeren -- Einschränkung über `window_start`, kein Matching nötig; Kontrollrechnung
bestätigt: die eingeschränkte Teilmenge liefert bitgleiche RMSE-Werte zum nativen
`horizon=36`-Datensatz bei h=1, für alle 18 Kombinationen.

GRU-Einschritt-RMSE auf der vollen h=1-Population gegen die h=36-Teilmenge (relative
Abweichung je Seed, Median über 3 Seeds): angeregt +0,37 %, Konstantlauf −0,45 % --
ebenso bei Persistenz (+0,16 %/−0,74 %) und Statisch (−0,53 %/+0,13 %), alle Abweichungen
unter 1,2 % und ohne einheitliches Vorzeichen über Modelle/Testmengen.
**Kernfrage beantwortet: der Effekt verschwindet nicht** -- die Fensterpopulation erklärt
weder die Größenordnung (< 1,2 % durch Populationswechsel gegenüber ~40 % relativem
Rückgang h1→h36 bei GRU) noch die Richtung des Musters. Die
Fensterpopulations-Hypothese ist damit **widerlegt**, nicht nur offen.

Bleibt als offene Frage, in diesem Auftrag NICHT geprüft: Rollout-Training. Ob GRU mit
einem Mehrschritt-/Rollout-Verlust trainiert wurde, der dem allerersten vorhergesagten
Schritt strukturell weniger Gewicht gibt als den späteren (z. B. weil der Verlust über
alle 36 Schritte gemittelt wird und der erste Schritt nur 1/36 der Trainingsgradienten
je Fenster beiträgt), ist eine offene, hier nicht geprüfte nächste Hypothese.

**Nachtrag (AD1): Exponentialansatz -- gestützte, aber nicht bewiesene
Zustandsschätzer-Erklärung.** Kein Training -- GRU-Checkpoints aus X2 geladen,
Persistenz/statische Baseline frisch angepasst, alle 36 Horizonte auf der (wie in AC2
konstant gehaltenen) h=36-Fensterpopulation ausgewertet
(`scripts/probe_ad1_exponential_decay.py`,
`artifacts/runs/probe_ad1_exponential_decay.json`, 18 Records). Modell `err(h) = floor +
A·exp(−h·dt/τ)`, `dt=25 s`, je Seed und Testmenge einzeln angepasst (nicht über Seeds
gemittelt).

GRU (6 Fits, 3 Seeds × 2 Testmengen): `floor` 0,124-0,206 K (Median 0,171 K), `A`
0,120-0,184 K (Median 0,141 K), `τ` 85,4-225,0 s (Median 157,9 s) -- durchgehend nahe am
Vier-Punkt-Vorab-Fit (`floor` 0,181 K, `A` 0,143 K, `τ` 143 s) und mit sehr kleinem
Residuum (`R²` 0,978-0,996, RMS-Residuum 0,0024-0,0046 K). **Der Vier-Punkt-Fit war nicht
überangepasst** -- die 36-Punkte-Fits reproduzieren ihn eng. **Gegenprobe bestanden:**
alle sechs `τ`-Werte (85,4-225,0 s) liegen innerhalb des Modenbands des gekoppelten
Wassermasse/Betttemperatur-Paars am Referenzpunkt (30,2-359,9 s, Befund 49); zwei Werte
(85,4/119,6 s) liegen leicht unterhalb der 133-1195-s-Spanne über den vollen
Betriebsbereich, aber innerhalb der engeren Referenzpunkt-Spanne.

Baselines zeigen die Form nicht: Persistenz fittet zwar mit hohem `R²` (0,975-0,9999),
aber mit **negativer** Amplitude (`A` −0,36 bis −9,55 K -- wachsender, nicht fallender
Fehler) und einem `floor`, der weit außerhalb des beobachteten Bereichs extrapoliert wird
(z. B. Seed 1 angeregt: `floor`=10,12 K gegenüber beobachtetem `err(h=36)`=6,57 K -- der
Fehler wächst bei h=36 noch ungebremst, erreicht keinen Boden). Statisch zeigt nahezu
keine Amplitude (`|A|` 0,01-0,83 K gegenüber `floor` 5,1-6,8 K, ein Fall exakt flach:
Seed 2 angeregt, `err(h=1)`=`err(h=36)`=6,327 K, `R²`=0) und ein instabiles `τ`
(0,3-4643 s -- bei verschwindender Amplitude nicht mehr identifizierbar). Keine der
beiden Baselines zeigt dieselbe fallende Exponentialform mit physikalisch plausiblem
`τ` wie GRU.

**Einordnung:** die Übereinstimmung von `τ` mit dem Modenband, die durchgehend hohe
Fitgüte bei GRU und die Abwesenheit derselben Form bei den Baselines **stützen** die
Zustandsschätzer-Erklärung (der rekurrente Zustand nähert sich mit wachsendem Horizont
der wahren Dynamik an) -- **bewiesen ist sie damit nicht**: ein Exponentialfit mit
plausiblem `τ` ist notwendig, aber nicht hinreichend für den kausalen Mechanismus, und
die Rollout-Training-Hypothese oben ist damit nicht ausgeschlossen. Der entscheidende
Test wäre eine Variation von `history_length` (eine längere Historie sollte den
h1-Fehler senken, den Boden bei h=36 aber nicht wesentlich verändern, wenn die
Zustandsschätzer-Erklärung trägt) -- das braucht Training und gehört nicht in diesen
Auftrag.

**Nachtrag (AE1): `τ`-Streuung nach Testmenge gruppiert -- schwacher, aber konsistenter
Testmengen-Hinweis, mit Seed-Effekt vermengt, keine neue Rechnung.** Dieselben sechs
GRU-Fits aus AD1, gruppiert:

| Testmenge | `τ` [s] (Seed 1/2/3) | Median | Spanne | `floor` [K] Median [Spanne] | `A` [K] Median [Spanne] |
|---|---|---|---|---|---|
| angeregt | 150,7 / 85,4 / 165,1 | 150,7 | [85,4, 165,1] | 0,1853 [0,1665, 0,2056] | 0,1372 [0,1201, 0,1454] |
| Konstantlauf | 179,4 / 119,6 / 225,0 | 179,4 | [119,6, 225,0] | 0,1270 [0,1236, 0,1748] | 0,1588 [0,1325, 0,1843] |

Je Seed gepaart (angeregt → Konstantlauf): Seed 1 150,7→179,4 s (+28,7), Seed 2
85,4→119,6 s (+34,2), Seed 3 165,1→225,0 s (+59,9) -- **alle drei Seeds in dieselbe
Richtung** (Konstantlauf-`τ` > angeregt-`τ`), konsistent mit Befund 49 (angeregte Läufe
durchlaufen einen weiten Betriebsbereich, Konstantläufe einen engen; die Begleitmode des
gekoppelten Paars variiert darüber von 133 auf 1195 s). Gleichzeitig ist die
Seed-Rangfolge (Seed 2 < Seed 1 < Seed 3) in beiden Testmengen identisch -- ein
seedseitiger Effekt liegt also gleichzeitig vor, nicht nur ein testmengenseitiger. Der
Median-Unterschied zwischen den Testmengen (28,7 s) ist kleiner als die Streuung
innerhalb jeder einzelnen Testmenge (79,7 s bzw. 105,4 s) -- aus den Randverteilungen
allein wäre keine Trennung erkennbar, nur aus der Seed-Paarung.

**Vorbehalt, ausdrücklich:** mit drei Seeds je Gruppe ist das keine belastbare
statistische Trennung (3/3 gleiche Richtung hat bei reinem Zufall ~25 % Wahrscheinlichkeit).
Zusätzlich bleiben selbst die höheren (Konstantlauf-)`τ`-Werte (bis 225,0 s) weit unter
dem kalten Ende des vollen Betriebsbereich-Bands (1195 s) -- die beobachtete Variation
deckt nur einen schmalen Ausschnitt des physikalisch möglichen Modenbands ab, kein
Hot-bis-Kalt-Spektrum. Das ist ein **schwacher, richtungskonsistenter Zusatzhinweis**,
der von der Fitform unabhängig ist, aber kein eigenständiger Beleg -- er verstärkt die
in (AD1) bereits als "gestützt, nicht bewiesen" eingeordnete Zustandsschätzer-Erklärung
graduell, ändert die Einordnung selbst nicht.

**j) Korrektur (AB2) zur Einordnung in (a): "reine Anwendungsgröße" trägt in Kelvin
nicht.** Die Einordnung der Ablufttemperatur als Anwendungsgröße und NICHT als
Architekturvergleichskanal (a) wurde über R²_stat = 0,768 und FIR k=43 mit R²_holdout =
0,965 begründet -- beides R²-Kennzahlen. In Kelvin liegt zwischen dem besten linearen
Modell (FIR k=43, 3,62 K Residuum, 30,2 kW-Äquivalent -- jetzt artefaktverifiziert, Befund
47 Nachtrag AB2/AC1) und GRU (0,18 K, 1,5 kW, Abschnitt e) Faktor **20**. Eine Einordnung,
die allein auf den hohen R²-Werten beruht, trägt diesen Unterschied nicht -- R² nahe 1 bei
großer Var(y) (Befund 47, Nachtrag AB2/AC1) verdeckt einen in absoluten/physikalischen
Einheiten weiterhin großen Architekturunterschied.
Die Entscheidung, ob die Ablufttemperatur (wieder) in einen gemeinsamen
Architekturvergleich aufgenommen werden sollte, ist damit **offen** -- hier nicht
entschieden, nur als offene Frage festgehalten.

Sachverhalt und Zahlen, keine Empfehlung.

### 51. Lernkurven-Ersatzdatensatz unter dokumentierter Konfiguration -- K1 widerlegt die angenommene Regionsaufteilung aus Befund 48

**Zweck.** `artifacts/runs/arch_composite.jsonl` (Befund 33: `git/commit="unknown"`,
Config nicht rekonstruierbar, entstanden vor der gesamten Git-Historie dieses Repos) wird
**nicht korrigiert**, sondern **ersetzt**: `artifacts/runs/arch_learning_curve_projected.jsonl`
(200 Records) unter git-verfolgter, vorab per Fingerprint verifizierter Konfiguration,
Commit `0b3eaaf` (Launcher-Skripte bereits in `19afa2a`, vor der Ausführung committet,
BA0b). **`arch_composite.jsonl` bleibt mit seinem Provenienzvorbehalt stehen** -- ein
zahlenmäßiger Vergleich zwischen beiden Dateien ist nicht zulässig, weil die Konfiguration
der alten Datei unbekannt ist.

**Konfiguration.** `process/fingerprint=2d2679eeed716577` (`enforce_gas_velocity_projection
=True`, `sample_time_s=25.0` angewendet, `outlet_duct_heat_loss_k=0.0` unverändert), vorab
gegen die zwölf `section_fingerprints` verifiziert (BA0c) -- keine Abweichung von der
dokumentierten Config seit deren Festlegung. Drei Zielkanäle explizit gesetzt
(`TARGET_NAMES` in `run_arch_learning_curve_worker.py`, BA0d):
`energy_target`/`product_moisture_kg_per_kg`/`bed_temperature_celsius` -- der
`DataSpec`-Default deckt nur zwei ab, `run_arch_constant_holdout_worker.py`
(`_excited_spec`, Zeile 71-82) demonstriert das Risiko: dort bleibt der Default
unverändert (zwei Kanäle), weil dieses Skript die Basis erzeugt und keine
`_expected_from_existing`-Prüfung hat, die den Fehler aufgedeckt hätte. Budgetpunkte
`n_train ∈ {8,16,32,64}` (Sobol-Verschachtelung, Zweierpotenzen), 5 Seeds,
Architekturen `persistence, static, arx, gru, lstm`, `history_length=32`,
`MKL_NUM_THREADS`/`OMP_NUM_THREADS=1` als Prozessumgebungsvariablen, `--parallel=4`.
Pilot (ein Seed, `n_train=8`) vorab bestanden: `git/commit` ein Hash, `git/dirty=False`,
`torch/num_threads`/`env/mkl_num_threads`/`env/omp_num_threads=1`,
`model/net_actual_device="cpu"`, `process/fingerprint` wie erwartet,
`data/target_names` mit allen drei Kanälen, `rmse_run_median_physical` belegt.

**Laufzeit.** Vorab geschätzt (2-Punkt-Fit aus den Pilotmessungen `n_train=8`/`16`,
linear mit Achsenabschnitt): ~2,9 h. Tatsächlich: **3,96 h** (14250,8 s, 5 Jobs in 2
Batches bei `--parallel=4`) -- unter der 4-h-Schwelle, aber näher daran als die
Vorabschätzung nahelegte (die `n_train=32/64`-Punkte liegen außerhalb des für den Fit
verwendeten Messbereichs).

**Zwischenfall (Provenienz).** Der erste Sweep-Versuch wurde nach ~20 Minuten
abgebrochen: ein parallel gestartetes Sondierungsskript (K1-Probe) schrieb eine neue,
nicht ignorierte Datei (`artifacts/runs/probe_l1_clip_sensitivity_projected.json`) in den
Arbeitsbaum, während zwei der fünf Seed-Jobs bereits liefen -- die beiden zuletzt
gestarteten Jobs (Seeds 3/4) protokollierten dadurch `git/dirty=True`, exakt der
Fehler, den dieser gesamte Befund beheben soll. Alle Prozesse wurden beendet, die
Probe-Datei committet (Commit `0b3eaaf`, stellt den sauberen Zustand wieder her), das
Arbeitsverzeichnis geleert, der komplette Sweep (alle fünf Seeds) auf dem jetzt sauberen
Arbeitsbaum neu gestartet. Alle 200 Records tragen denselben Commit (`0b3eaaf`) und
`git/dirty=False` (geprüft). Lehre: kein weiteres Skript, das neue getrackte Dateien
anlegt, während ein provenienzsensitiver Lauf aktiv ist.

**K1 -- widerlegt die angenommene Regionsaufteilung.** Erwartung: dR2(k=43) für
`product_moisture_kg_per_kg` sollte sich gegenüber der unprojizierten Referenz (+0,431)
etwa halbieren (~+0,2), weil die verletzende Region punktweise etwa hälftig aus
`u > u_terminal` (8,8 %) und `bed_temperature > 80 °C` (8,6 %) bestand. Gemessen (
`probe_k1_l1_clip_sensitivity_projected.py`, unverändert aufgerufen):
**dR2(k=43) = +0,4205** -- praktisch unverändert. Ursache (Zusatzmessung, `_violation_mask`
komponentenweise ausgewertet, 3 Seeds, `n_train_max=64`): die Projektion senkt
`u > u_terminal` von 6,71 % auf 0,17 % der Punkte (wie erwartet, siehe K2), aber die
UNION-Verletzungsmaske (alle vier Kriterien) sinkt nur von 17,10 % auf 11,26 % -- eine
Reduktion um 34 %, nicht 50 %. Die `u_terminal`-Verletzung ist **keine unabhängige
Hälfte** der Region, sondern überlappt stark mit den bereits durch
`bed_temperature > 80 °C`/Feuchte-Kriterium erfassten Punkten. **K1: nicht erfüllt** --
Befund 48s Annahme einer disjunkten 50/50-Aufteilung der Verletzungsregion ist zu
revidieren, nicht nur zu quantifizieren.

**K2 -- erfüllt.** Restverletzung `u > u_terminal`, gepoolt über alle 5 Seeds des
Hauptlaufs (737 280 Punkte): **0,237 %** (Erwartung "rund 0,16 %", frühere Baseline
6,0 %). `projection_active=True` in jeder Trajektorie; `n_projected_steps > 0` an
195 von 640 Läufen (5 Seeds × 128 Läufe).

**K3 -- erfüllt.** Temperaturkanal, unter Abluftnamen berichtet
(`exhaust_rmse_run_median_physical/outlet_temperature_celsius`, h=36, `n_train=64`,
Median über 5 Seeds): statisch 6,854 K gegenüber GRU 0,206 K -- **Faktor 33,3**
(Referenz aus dem schlanken Protokoll: Faktor 38; auf `bed_temperature_celsius` direkt
Faktor 16,3). Deutlich über der geforderten Schwelle (Faktor 5).

**K4 -- überwiegend erfüllt, mit benannten Ausnahmen.** GRU/LSTM fallen für
`energy_target` und `bed_temperature_celsius` über alle vier Budgetpunkte monoton
(Median über 5 Seeds, h=36); LSTM ebenso für `product_moisture_kg_per_kg`. GRU zeigt dort
einen kleinen Anstieg von `n=16` (9,06e-4) auf `n=32` (9,96e-4), rund 10 % relativ, bevor
der Wert bei `n=64` (6,09e-4) wieder fällt. Die Baselines `static`/`arx` sind für
`product_moisture_kg_per_kg` deutlicher nicht-monoton (`static`: 0,0317→0,0278→0,0550→
0,0407 -- Verdopplung bei `n=32` relativ zu `n=16`; `arx` analog schwächer) und `static`
schwach auch für `bed_temperature_celsius` (`n=32`→`n=64`: 6,916→6,956). Persistenz zeigt
konstruktionsbedingt über alle Budgetpunkte identische Werte (kein Training auf
`n_train`) -- keine echte Nicht-Monotonie, nur ein Artefakt der Prüfung auf `>=`.

**K5 -- Nachtrag (CA1): der Faktor-Sprung ist ein Testmengen-Wechsel, keine Verbesserung
und keine Verschlechterung.** Vor der ersten Fassung dieses Befunds stand hier ein Faktor
4,4-23 ohne Erklärung -- eine Größenordnung über der bisherigen Referenz (Faktor 1,5-2,
`artifacts/runs/arch_gab_constant_holdout.jsonl`, `constant_holdout`, 10 Seeds, Persistenz
0,5437/GRU 0,354/LSTM 0,272, `nrmse_run_median`). Ein Sprung dieser Größe braucht eine
Ursache, bevor er berichtet wird -- deshalb hier zuerst der Absolutwert-Vergleich, nicht
der Faktor.

*Zulässigkeitsprüfung.* `split/digest` (`5b9628a151344f37`), `data/horizon` (36),
`data/n_val`/`n_test` (32/32), `holdout/seed`/`n_runs` (999/32) und die
Anregungs-/Splitgeometrie sind zwischen altem und neuem Log **identisch** -- dieselbe
Testmenge in Konstruktion, nicht nur im Namen. Zwei Unterschiede bleiben ungeklärt, tragen
aber nachweislich nichts zur unten erklärten Größe des Sprungs bei (s. Kernfrage):
Zielkanalsatz (`d50_um` alt gegen `bed_temperature_celsius` neu -- ob das gemeinsame
Training mit dem jeweils dritten Kanal die Feuchte-Metrik selbst beeinflusst, ist mit den
vorhandenen Logs nicht trennbar, da keine Läufe mit sonst identischer Config und
getauschtem drittem Kanal existieren) und Seedzahl (10 alt gegen 5 neu). Zusätzlich:
altes Log auf Windows erzeugt, neues auf Linux (`platform`-Feld) -- nach dem Nebenbefund
weiter unten (Bit-Abweichungen über viele RK4-Schritte) ein Effekt in den letzten
Nachkommastellen, keiner dieser Größenordnung.

*Absolutwerte, `product_moisture_kg_per_kg`, h=36, Median über die Seeds je Zelle:*

| Testmenge | Modell | alt `nrmse_run_median` | neu `nrmse_run_median` (n=64) | alt `rmse_...physical` | neu `rmse_...physical` (n=64) |
|---|---|---|---|---|---|
| `constant_holdout` | Persistenz | 0,5437 | 0,4466 | -- | 0,001231 kg/kg |
| `constant_holdout` | GRU | 0,354 | 0,3409 | -- | 0,000651 kg/kg |
| `constant_holdout` | LSTM | 0,272 | 0,2623 | -- | 0,000543 kg/kg |
| `excited` | Persistenz | *(nicht referenziert)* | 0,7646 | -- | 0,011727 kg/kg |
| `excited` | GRU | *(nicht referenziert)* | 0,0383 | -- | 0,000609 kg/kg |
| `excited` | LSTM | *(nicht referenziert)* | 0,0315 | -- | 0,000511 kg/kg |

(Physikalischer RMSE des alten Logs nicht nachgetragen -- `rmse_run_median_physical`
existiert dort als Feld, wurde aber für diesen Nachtrag nicht neu ausgewertet, da die
`nrmse`-Spalte für die Fragestellung ausreicht; bei Bedarf nachrüstbar.)

**Kernfrage beantwortet: weder noch.** Auf der Testmenge, auf der die alte Referenz
gemessen wurde (`constant_holdout`), ist Persistenz nicht schlechter und sind GRU/LSTM
nicht auffällig besser geworden -- der neue Faktor bei `n_train=64` liegt bei 1,31 (GRU)
bzw. 1,70 (LSTM), leicht **unter** der alten Referenz (1,54/2,00), praktisch konsistent.
Der zitierte Faktor 4,4-23 stammt dagegen von der `excited`-Testmenge, für die es **keine**
alte Referenz gibt -- kein Sprung, sondern ein erstmals gemessener, andersartiger
Vergleich, in der ursprünglichen Fassung dieses Befunds ohne Testmengen-Angabe zitiert.
Der Mechanismus: auf `excited` (aktiv angeregte Läufe) ist die naive Persistenz-Vorhersage
strukturell schlecht (NRMSE 0,7646, physikalisch 0,0117 kg/kg) -- sie kann Sollwertsprünge
innerhalb des Horizonts nicht abbilden; auf `constant_holdout` (Betriebspunkt konstant)
ist sie dagegen trivial gut (0,4466 bzw. 0,0012 kg/kg). GRU/LSTMs absoluter Fehler
unterscheidet sich zwischen den beiden Testmengen dagegen kaum (physikalisch
0,0005-0,0006 kg/kg auf beiden) -- der große Faktor auf `excited` kommt ausschließlich aus
dem Nenner (einer schwachen Baseline dort), nicht aus einem besseren Modell.

**K5 -- erfüllt, mit korrigierter Referenz.** Kein Rückgang gegenüber der alten Referenz
(`constant_holdout`: 1,31/1,70 gegen 1,54/2,00, konsistent); der auf `excited` gemessene,
deutlich größere Faktor (4,4-24 über `n_train=8..64`, Median über 5 Seeds) ist ein
eigenständiges, erstmals erhobenes Ergebnis auf einer anspruchsvolleren Testmenge, keine
Verbesserung gegenüber einer nicht existierenden Referenz. Beide Zahlen bleiben gültig,
solange sie mit Testmenge zitiert werden.

**Nebenbefund (Testsuite).** `tests/test_gas_velocity_projection_disabled_matches_legacy_baseline`
schlägt auf dieser Maschine fehl (`np.array_equal` gegen eine eingefrorene `.npz`-Fixture
liefert `False` in Nachkommastellen jenseits der Anzeigegenauigkeit). Einzige
Fehlschlag der vollen Suite. Vermutlich plattformbedingte Gleitkomma-Abweichung
(Fixture auf anderer Maschine erzeugt, `libm`/BLAS-Implementierung kann sich über 1152
RK4-Schritte in den letzten Bits unterscheiden) -- konsistent mit dem unabhängigen
Kreuzvergleich Windows/Linux dieses Befunds (identischer Seed, identischer Worker,
NRMSE-Werte stimmen bis auf die letzten 2-3 Stellen überein). Nicht untersucht, ob dies
zutrifft; hier nur als offener Befund festgehalten, nicht repariert (keine Änderung an
`src/`, kein Ersetzen der Fixture ohne Rücksprache).

**Was NICHT Teil dieses Befunds ist.** xLSTM, Mamba, Transformer (explizit
ausgeschlossen). Kein Vergleich der absoluten Zahlen gegen `arch_composite.jsonl`
(Provenienzvorbehalt, s.o.).

**Nachtrag (Befund 58): die drei ausgeschlossenen Architekturen sind seither nachgezogen
worden.** `artifacts/runs/arch_learning_curve_projected_extended.jsonl` deckt
Transformer/xLSTM/Mamba unter **geprüft identischem Protokoll** ab -- derselbe
Fingerprint (`2d2679eeed716577`), je `master_seed` derselbe `split/digest` und dieselben
`seed/data`/`seed/noise`/`seed/model` wie hier. Die beiden Dateien sind damit
**protokollgleich und gemeinsam auswertbar**, nicht nur nebeneinander zitierbar -- siehe
Befund 58 für den vollständigen Achtergleich (inklusive der dortigen
Epochenobergrenzen-Konfundierung bei Transformer, die diesen Datensatz hier nicht
betrifft: GRU/LSTM binden zwar teils ebenfalls an `model/net_max_epochs`, aber mit
fallender statt steigender Tendenz über `n_train`).

---

Fingerprint-Historie (Ergänzung zur Tabelle oben): `2d2679eeed716577` trägt mit
`arch_learning_curve_projected.jsonl` (Commit `0b3eaaf`) zum ersten Mal ein committetes
Log -- vorher stand "Bislang kein Log" in der Tabelle. Skripte mit hartkodierten
`EXPECTED`-Werten, die auf diesen neuen, git-verfolgten Datensatz umgestellt werden
**könnten** (nicht umgesetzt, nur aufgeführt): `run_xlstm_reduced_comparison.py`
(intern widersprüchlich seit der `d50_um`-Entfernung), sowie die beiden
Constant-Holdout-Worker mit `_expected_from_existing`-Prüfung,
`run_mamba_constant_holdout_worker.py` und `run_xlstm_constant_holdout_worker.py`.
Sachverhalt, keine Empfehlung.

### 52. Verschiebt sich der energieoptimale Betriebspunkt über den Störgrößenbereich? -- Voruntersuchung zur RL-Eignung, kein Vorschlag

**a) Zweck und Abgrenzung.** Kein Reinforcement Learning, kein Netztraining, kein
Datensatz-Sweep -- nur `solve_steady_state` (`simulation/steady_state.py`) plus
klassische Optimierung je Gitterpunkt (`differential_evolution` + Nelder-Mead-
Verfeinerung), implementiert in `scripts/study_operating_point_regret.py` (vollständige
Problemdefinition im Skript-Docstring). Optimiert werden `inlet_temperature_celsius`,
`gas_velocity_m_per_s`, `nozzle_pressure_bar` (Grenzen aus `ActionSpaceConfig`);
`spray_rate_kg_per_h` bleibt am Nominalwert fest (bestimmt bei festem
Suspensionsfeststoffanteil den Durchsatz -- ohne Fixierung wäre das Energieminimum
Sprührate null, keine Produktion). Störgrößengitter 5×5 über
`ambient_temperature_celsius` × `ambient_relative_humidity` (Bereiche aus
`DisturbanceSpaceConfig`), `suspension_solid_fraction` fest am Nominalwert. Harte
Nebenbedingungen: `product_moisture_kg_per_kg <= critical_moisture_kg_per_kg`,
`u_mf(d50) <= gas_velocity <= u_terminal(d50)`, `bed_temperature_celsius <= 80`
(Platzhalter, Befund 46).

**b) Ergebnis: fester Betriebspunkt existiert, Regret liegt im Prozentbereich.** An
allen 25 Gitterpunkten zulässig, DE-Suchbudget planmäßig ausgeschöpft (kein
Fehlschlag, siehe c), Verfeinerung überall konvergiert. Fester Betriebspunkt (mittlere
Energie über das Gitter minimal, an jedem Punkt zulässig): `inlet_temperature_celsius`
= 90,41 °C, `gas_velocity_m_per_s` = 0,800 m/s, `nozzle_pressure_bar` = 1,003 bar.
Regret (n=25): Median 0,174 kW (16,0 %), P95 0,230 kW (22,2 %), Max 0,242 kW (25,8 %) --
klar im Prozentbereich, nicht im Promillebereich (Einordnungsmaßstab aus dem Auftrag: Promille
→ fester Betriebspunkt genügt, Prozent → Policy begründet; keine Empfehlung, nur
Einordnung). Spannweite der instanzweisen Optima über das Gitter: `inlet_temperature_celsius`
[62,36; 90,42] °C (Spanne 28,1 K, 46,8 % der Stellgrößenspanne), `gas_velocity_m_per_s`
[0,800; 1,016] (12,7 %), `nozzle_pressure_bar` [1,000; 1,011] (0,4 % -- praktisch
invariant, konsistent mit `energy.py`s eigener Feststellung, dass diese Stellgröße
reinen Kosten- ohne Nutzenanteil hat: der Optimierer fährt sie durchgehend ans Minimum).
`bed_temperature_celsius` liegt an allen 25 Punkten zwischen 30,0 °C und 45,0 °C und
**bindet nirgends** an der 80-°C-Platzhaltergrenze -- **BB4 (Sensitivität gegen diese
Grenze) entfällt damit**, seine Vorbedingung ist nicht erfüllt.

**c) Regimewechsel -- die mechanistische Erklärung des Regrets.** Je Gitterpunkt bindet
entweder die Feuchtegrenze (Optimum sitzt exakt auf `critical_moisture_kg_per_kg`) oder
keine der drei Nebenbedingungen (Optimum liegt an der unteren Box-Grenze von
`gas_velocity_m_per_s`, 0,8 m/s). Der Übergang verläuft nicht allein über
`ambient_temperature_celsius`, sondern ist eine Kurve in der
(`ambient_temperature_celsius`, `ambient_relative_humidity`)-Ebene -- höhere
Luftfeuchte verschiebt sie zu höherer Umgebungstemperatur, weil feuchtere Zuluft
weniger Trocknungskapazität mitbringt und deshalb bei sonst gleicher Umgebungstemperatur
mehr Heizleistung braucht, um dieselbe Verdampfungstriebkraft zu erreichen. Gitteraufgelöst
(5×5, Schrittweite 7,5 K × 0,15): bei `ambient_temperature_celsius` <= 20 °C bindet die
Feuchtegrenze bei allen fünf getesteten Luftfeuchten; bei 35 °C bindet sie bei keiner; die
Zeile bei 27,5 °C liegt genau im Übergang und zerfällt selbst nach Feuchte (bindet bei
`ambient_relative_humidity` >= 0,70, nicht bei <= 0,55). Die Übergangstemperatur liegt
demnach bei niedriger Feuchte zwischen 20 und 27,5 °C, bei hoher Feuchte zwischen 27,5 und
35 °C -- eine Verschiebung um mindestens 7,5 K je nach Feuchte, nicht feiner bestimmbar als
das Gitter erlaubt. Ein fester Betriebspunkt muss beide Regime gleichzeitig bedienen: im
kalten/feuchten Regime zwingt die Qualitätsgrenze zu einem bestimmten Mindestaufwand, im
warmen/trockenen Regime ist der energieoptimale Punkt unbeschränkt und läge -- ohne den
Kompromiss mit dem anderen Regime -- noch niedriger. Genau dieser Kompromiss ist der
gemessene Regret.

**d) Vorbehalte, vollständig, alle in Richtung Überschätzung des Regrets/der
erzielbaren Einsparung.**

- **Beharrungszustand statt Charge -- jetzt beziffert, siehe Befund 53.** `tau_4` liegt
  bei rund 16 000 s gegenüber einer Chargendauer von 28 800 s -- 83,5 % Annäherung an den
  Beharrungszustand. Befund 53 rechnet die hier vorausgesetzte Annahme nach: Der Regret
  bleibt unter transienten 28 800-s-Rollouts (chargengemittelte statt Fixpunkt-Energie,
  unveränderte Sollwerte) praktisch unverändert (Median 16,25 % statt 16,03 %, Max
  26,03 % statt 25,80 % -- also nicht kleiner, wie hier vermutet, sondern minimal
  größer). Der Vorbehalt in dieser Richtung ("tatsächlich erzielbarer Gewinn kleiner als
  berichteter Regret") trifft damit NICHT zu. Was sich transient tatsächlich zeigt: `d50`
  und die Partikelanzahl (nicht die Feststoffmasse selbst, die über den separaten,
  schnellen Regelkreis `tau_3 = 600` s bereits nach einem Bruchteil der Charge konvergiert
  ist) bleiben am Chargenende 19,6 % vom Fixpunkt entfernt, und die chargengemittelte
  rückgewinnbare Leistung `Q_rec` liegt median 36 % (bis zu 76 %) über dem
  Fixpunktwert -- beides ohne Wirkung auf `energy_target` oder die Zulässigkeit.
- **Perfekte Störgrößenkenntnis unterstellt.** Jeder Gitterpunkt wird mit exakter
  Kenntnis von `ambient_temperature_celsius`/`ambient_relative_humidity` gelöst -- eine
  reale Policy hätte nur Messungen oder Prognosen davon. Der berichtete Regret ist die
  Obergrenze einer Policy mit perfekter Störgrößenkenntnis, nicht einer realistischen.
- **Keine Stellratenbegrenzung berücksichtigt.** Die volle Spanne der optimalen
  `inlet_temperature_celsius` (28,1 K) entspräche bei der Platzhalterrate 0,5 K/s einem
  Übergang von 56 s -- kurz gegenüber typischen Wetteränderungszeitskalen, also
  vermutlich ein kleiner Beitrag zur Überschätzung, aber nicht Null und hier nicht
  gerechnet.
- **`critical_moisture_kg_per_kg` ist ein Platzhalter ohne Quelle** (0,200 kg/kg,
  Laborniveau, siehe Dateikopf `greybox_gab_placeholder.yaml`) und bindet genau im
  kalten/feuchten Regime (c). Der dort berechnete Regret hängt damit direkt an einem
  unbelegten Zahlenwert -- eine andere kritische Feuchte verschiebt Lage und Höhe des
  Optimums in diesem Regime. Befund 54 (DA1) misst diese Abhängigkeit explizit über
  0,10-0,30 kg/kg: die Prozentbereich-Einordnung bleibt stabil, die Regret-Größenordnung
  nicht (Faktor 2-3 unterhalb von 0,20 kg/kg) -- ebenso geprüft dort: `tau_4`
  (Bettmasse-Sollwert), gegen das Befund 52s Regret dagegen robust ist.
- **Ein Prozess, kein Verbund.** Kein Wärmenetz, keine Abwärmerückgewinnung zwischen
  Anlagen berücksichtigt (vgl. `exhaust_sensible_kw`/`exhaust_latent_kw` in
  `SteadyStateSolution`, hier nicht einbezogen) -- auf Verbundebene könnte ein Teil der
  hier als Regret gezählten Energie ohnehin zurückgewonnen werden, unabhängig vom
  gewählten Betriebspunkt. **Nachtrag, seit `heatnetwork/` existiert
  (`docs/06_waermenetz.md`):** die hier minimierte Zielfunktion ist der
  BRUTTOVERBRAUCH (`power_total`) OHNE Rückgewinnungsgutschrift. Das ist mehr als nur
  ein fehlender Term -- `gas_velocity_m_per_s` sitzt im Optimum an allen 25
  Gitterpunkten an ihrer unteren Box-Grenze (c), und `Q_rec`
  (`heatnetwork.balance`/`evaluation.recoverable_power`) steigt sowohl mit dem
  Massenstrom als auch mit der Ablufttemperatur. Mit einer Netto-Zielfunktion
  (Bruttoverbrauch minus Rückgewinnungsgutschrift) könnte sich deshalb nicht nur die
  Höhe des Regrets, sondern die LAGE des Optimums qualitativ ändern -- ein höherer
  Gasstrom, der mehr Heizleistung kostet, aber auch mehr rückgewinnbare Leistung
  liefert, wäre unter der Bruttozielfunktion nie im Optimum sichtbar. **Als offene
  Frage festgehalten, nicht als Ergebnis:** die Neuformulierung der Zielfunktion und
  ihre Auswirkung auf Befund 52s Optimum sind nicht Teil des `heatnetwork`-Auftrags und
  hier nicht gerechnet.

**e) Methodischer Fehlschlag im ersten Lauf, festgehalten statt gelöscht.** Der erste
volle Gitterlauf (`artifacts/runs/bb1_bb3_operating_point_regret_PREFIX.json`) fand 14
von 25 Gitterpunkten unzulässig und keinen überall zulässigen festen Betriebspunkt --
beides falsch. Ursache: die Zulässigkeitsprüfung verlangte exakte Gleichheit
(Toleranz 1e-9) gegen dieselbe Nebenbedingung, für die die "bindet sie?"-Prüfung
bereits eine sinnvolle Toleranz (1e-4 kg/kg) verwendete -- eine Toleranzasymmetrie
zwischen zwei Prüfungen derselben Größe, dieselbe Fehlerklasse wie an anderen Stellen in
diesem Projekt. 8 der 14 Punkte hingen ausschließlich daran (Optimierer traf die
Feuchtegrenze korrekt auf ~1e-7 kg/kg genau, das wurde als Verletzung gewertet); die
übrigen 6 waren ein Solver-Bracket-Fehlschlag durch unterschiedliche Bisektionspräzision
zwischen Such- und Report-Aufruf (behoben durch identische Präzision in beiden, siehe
Docstring von `evaluate_point` -- ein erster Korrekturversuch über ein verbreitertes
Bracket verschärfte das Problem stattdessen auf praktisch jeden Gitterpunkt, ebenfalls
dort dokumentiert). Die fehlerhafte Datei bleibt committet, damit der Vergleich (14/25
unzulässig, kein fester Punkt → 0/25 unzulässig, fester Punkt existiert) nachvollziehbar
bleibt.

**f) Provenienz nachgezogen, plattformübergreifend bitgleich reproduziert.**
`artifacts/runs/bb1_bb3_operating_point_regret.json` trägt als einziges Befund-52/53/54-
Artefakt kein `environment`-Feld -- `scripts/study_operating_point_regret.py` rief
`environment_fingerprint()` zum Zeitpunkt des ursprünglichen Laufs schlicht nicht auf
(anders als die später entstandenen `study_operating_point_transient.py`/`probe_*.py`).
Diese Lücke lässt sich nachträglich nicht mehr schließen -- welcher Commit/Dirty-Zustand
den ursprünglichen Lauf erzeugt hat, ist nicht rekonstruierbar.

Beim Nachziehen der Provenienz (Ergänzung von `environment_fingerprint()`, analog zu den
späteren Skripten) stellte sich heraus: das Skript war unter Windows nie ausführbar. Es
teilt die Config über eine modulglobale Variable (`_BASE_CONFIG`), die per `fork()` in die
`Pool`-Worker vererbt werden sollte -- funktioniert unter Linux (Fork-Default), bricht
unter Windows (spawn-Default) in jedem Worker mit `AssertionError` ab, da dort `_BASE_CONFIG`
nie gesetzt wird. Befund 52 kann demnach nur auf dem Linux-Laptop entstanden sein. Fix:
`Pool(..., initializer=_set_base_config, initargs=(base_config,))` setzt die Config in
jedem Worker explizit, unabhängig von der Startmethode -- Verhalten unverändert, belegt
durch einen Determinismusvergleich (gleicher Seed, 16 vs. 8 Worker, 0 Abweichungen außerhalb
reiner Timing-Felder).

Mit dem Fix (Commit `7103b73`, sauberer Baum, `git/dirty=False`) auf DIESER
Windows-Maschine neu gerechnet (5×5-Gitter, identisches Suchbudget), Ergebnis in
`artifacts/runs/study_operating_point_regret_win_repro.json` (neuer Name, das
ursprüngliche Artefakt bleibt unverändert): **bitgleich** zum ursprünglichen Linux-Lauf --
fester Betriebspunkt `inlet_temperature_celsius=90,4129 °C`, `gas_velocity_m_per_s=0,8001
m/s`, `nozzle_pressure_bar=1,0026 bar` (identisch bis auf die letzte gespeicherte
Nachkommastelle), Regret relativ Median 16,030 %/P95 22,199 %/Max 25,801 % (Differenz zum
Original: 0,0 in allen drei Kennzahlen), 0/25 unzulässig in beiden Läufen. Das ist damit
zugleich eine plattformübergreifende Reproduktion des Hauptergebnisses (Linux → Windows),
nicht nur eine Provenienz-Nachbesserung: **`study_operating_point_regret_win_repro.json`
ist die provenienzvollständige Fassung; `bb1_bb3_operating_point_regret.json` bleibt die
historische, in (b) zitierte Fassung ohne rekonstruierbare Provenienz -- beide committet,
beide stimmen in den berichteten Zahlen exakt überein.**

### 53. Überlebt der Befund-52-Regret den Übergang von Beharrungszustand zu Charge? -- ja, praktisch unverändert; die stationäre Zulässigkeit auch

**a) Zweck und Abgrenzung.** Befund 52 bewertet alle 25 Gitterpunkte (Umgebungstemperatur x
relative Luftfeuchte) ausschließlich im exakten Beharrungszustand (`solve_steady_state`).
`tau_4 ≈ 16 000 s` gegenüber einer Chargendauer von 28 800 s (83,5 % Annäherung) bedeutet:
der Fixpunkt wird während einer Charge nie erreicht. Dieser Befund prüft, ob Regret und
Zulässigkeit aus Befund 52 das übersteht -- transiente 28 800-s-Rollouts (`Delta_t = 25 s`,
1152 Schritte) des vorhandenen Grey-Box-Simulators (`GreyBoxFluidBedGranulator.rollout`)
mit den DORT gefundenen, konstant gehaltenen Sollwerten, ausgehend vom nominalen
Anfangszustand. Kein Netztraining, kein Reinforcement Learning, keine transiente
Reoptimierung außer im expliziten CB4-Abschnitt. Implementiert in `scripts/
study_operating_point_transient.py`, Ergebnis in `artifacts/runs/
study_operating_point_transient.json`.

Zwei Configs, bewusst unterschiedlich: die Beharrungszustands-Referenz verwendet
unverändert Befund 52s Config (Fingerprint `abbe8325f1326d80`, `sample_time_s = 5 s`
Dateidefault, Projektion AUS -- für `solve_steady_state` irrelevant, da algebraisch). Die
transienten Rollouts laufen unter `sample_time_s = 25 s`, `enforce_gas_velocity_
projection = True` (Fingerprint `2d2679eeed716577`) -- der Config, unter der im Projekt
sonst jede Chargen-/Anregungsstudie läuft (`DataSpec`-Default), NICHT Befund 52s
Beharrungszustands-Config. Grund für die Projektion: Befund 52 optimiert ohne sie
(`solve_steady_state` kennt keine Projektion) -- ihre Wirkung während der Charge ist Teil
der Fragestellung (CB1d), nicht vorentschieden.

**b) CB1 -- Fixpunkt gegen Chargenverlauf.** `d50` und die Feststoffmasse hängen an
unterschiedlichen Zeitkonstanten, nicht (wie in der Aufgabenstellung zunächst vermutet)
gemeinsam an `tau_4`: die Feststoff**masse** wird über den separaten, schnellen
Regelkreis `mass_control_time_constant_s = 600 s` (`tau_3`) geregelt und ist nach 28 800 s
(48 Zeitkonstanten) vollständig konvergiert; NUR `d50`/Partikelanzahl hängen an
`tau_4 = 16 000 s` und bleiben entsprechend zurück. `d50`-Dynamik ist zudem vollständig
entkoppelt von Stell-/Störgrößen (die Zahlenbilanz hängt nur von Sprührate/
Feststoffanteil ab, beide über das ganze Gitter am Nominalwert fest) -- die
Chargenend-Abweichung ist deshalb an allen 25 Gitterpunkten IDENTISCH.

| Kanal | Chargenende | Fixpunkt | Abweichung absolut | Abweichung relativ |
|---|---|---|---|---|
| `d50` [µm] | 602,98 | 750,00 | -147,02 | -19,60 % (an allen 25 Punkten identisch) |
| Feststoffmasse [kg] | 2,000 | 2,000 | 0,000 | 0,00 % (an allen 25 Punkten identisch) |
| `product_moisture_kg_per_kg` | -- | -- | Median -7,5e-6 (Spanne -3,4e-5 … +2,6e-8) | Median -0,0038 % (Spanne -0,0168 … +0,0000 %) |
| `bed_temperature_celsius` [°C] | -- | -- | Median +0,00049 (Spanne -0,0000037 … +0,0018) | Median +0,0015 % (Spanne -0,0000 … +0,0044 %) |

Feuchte und Betttemperatur -- beide über schnelle, gekoppelte Moden (30-1200 s je nach
Betriebspunkt, Befund 30/M1) -- sind nach 28 800 s praktisch am Fixpunkt; `d50` (und mit
ihm die Partikelanzahl) bleibt strukturell zurück. Die Erwartung aus der Aufgabenstellung
("`d50` und Feststoffmasse deutlich entfernt") trifft damit nur für `d50` zu, nicht für
die Feststoffmasse selbst.

`energy_target`: Chargenende praktisch bitgleich zum Fixpunkt (Median-Differenz
-2,0e-6 kW) -- erwartbar, da `energy_target` selbst nur von den schnellen Kanälen
abhängt. Chargen**mittel** liegt leicht darunter: Median -0,0037 kW (-0,37 %), Spanne
-0,0046 … -0,0016 kW (-0,38 … -0,17 %) -- klein, aber systematisch (die Anfahrphase
braucht durchgehend etwas weniger Leistung als der wärmere/feuchtere Fixpunkt in den
meisten Gitterpunkten).

`Q_rec` (rückgewinnbare Leistung, `twinlink_ml.evaluation.recoverable_power.
compute_recoverable_power_kw`, unverändert wiederverwendet -- Bezugsmassenstrom 8 kg/s,
Taupunktmarge 3 K, Moduldefaults): Chargenende ebenfalls praktisch am Fixpunkt
(Median +0,0056 kW). Das Chargen**mittel** dagegen liegt SYSTEMATISCH und DEUTLICH
darüber: Median +11,7 kW (+35,9 %), Spanne +5,9 … +13,1 kW (+5,6 … +76,3 %). Ursache:
der nominale Anfangszustand (`bed_temperature_celsius = 55 °C`) liegt bei den meisten
Gitterpunkten OBERHALB des Fixpunkts (30-45 °C, Befund 52b) -- die Anfahrphase liefert
eine wärmere, zunächst trockenere Abluft und damit ein größeres `T_out - T_dp`-Fenster
als der eingeschwungene Zustand. Anders als bei `energy_target` ist dieser Effekt weder
klein noch vernachlässigbar: eine Verbund-Wärmerückgewinnungsplanung, die mit dem
Fixpunkt-`Q_rec` rechnet, UNTERSCHÄTZT das chargengemittelte Potenzial an fast jedem
Gitterpunkt um zweistellige Prozentwerte.

CB1d -- Gasgeschwindigkeits-Projektion: `n_projected_steps = 0` an ALLEN 25 Gitterpunkten
über alle 1152 Schritte. Die Projektion greift nicht ein, obwohl Befund 52 ohne sie
optimiert hat -- die Sorge aus der Aufgabenstellung (Abweichung könnte aus fehlender
Projektion in Befund 52 statt aus der Transienz stammen) trifft für dieses Gitter nicht
zu.

**c) CB2 -- sind die stationären Optima als Chargenfahrweise zulässig? Ja, alle 25.**
Über den GESAMTEN Chargenverlauf (nicht nur das Ende): 0 von 25 Gitterpunkten verletzen
`product_moisture_kg_per_kg > critical_moisture_kg_per_kg`, das Fluidisierungsfenster
(oberes oder unteres) oder `bed_temperature_celsius > 80` zu irgendeinem der 1152
Zeitschritte.

Die Vorhersage der Aufgabenstellung ("ein Betriebspunkt, der stationär genau auf der
Feuchtegrenze liegt, verletzt sie transient mit hoher Wahrscheinlichkeit einen Teil der
Zeit") **trifft nicht zu**. Am schärfsten geprüften Punkt (`ambient_temperature_celsius
= 5 °C`, `ambient_relative_humidity = 0,25`, Feuchtegrenze bindend am Fixpunkt,
`X_p,fix = 0,19999...`) steigt die Partikelfeuchte streng monoton vom nominalen
Anfangswert (0,0513) auf 0,19844 (t = 12 500 s) und 0,19999 (t = 28 800 s, letzter
Schritt) -- kein Überschwingen, das Maximum der gesamten Trajektorie liegt exakt am
letzten Zeitschritt. Erklärung: die Feuchtemode nähert sich dem Fixpunkt in diesem Modell
als gedämpfte, nicht oszillierende Relaxation (keine Regelkreis-Dynamik mit Überschwingen
in den Bilanzen von `greybox._rhs`) -- ein stationär bindender Sollwert wird deshalb von
unten angenähert, nie überschritten. Kernfrage in einem Satz: **0 von 25 stationär
zulässigen Optima sind als Chargenfahrweise unzulässig** -- die vermutete
Sicherheitsmarge ist für dieses Modell (glatte, nicht-oszillierende Bilanzen) nicht
begründet; sie könnte es für ein Modell mit Regelkreis-Überschwingen (z. B. eine
kaskadierte Feuchte-/Temperaturregelung) sein, das hier nicht vorliegt.

**d) CB3 -- überlebt der Regret? Ja, nahezu unverändert.** Dieselbe Regret-Rechnung wie
Befund 52 BB3(b), aber mit der CHARGENGEMITTELTEN Energie statt des Fixpunktwerts, bei
UNVERÄNDERTEN Sollwerten (25 Rollouts mit den je Gitterpunkt optimalen Sollwerten aus
CB1, 25 Rollouts mit Befund 52s festem Betriebspunkt `inlet_temperature_celsius = 90,41
°C`, `gas_velocity_m_per_s = 0,800 m/s`, `nozzle_pressure_bar = 1,003 bar`, je Gitterpunkt
mit denselben 28 800-s-Rollouts):

| | Beharrungszustand (Befund 52) | Chargenmittel (CB3) |
|---|---|---|
| Median absolut [kW] | 0,174 | 0,176 |
| Median relativ | 16,03 % | 16,25 % |
| P95 absolut [kW] | 0,230 | 0,232 |
| P95 relativ | 22,20 % | 22,40 % |
| Max absolut [kW] | 0,241 | 0,243 |
| Max relativ | 25,80 % | 26,03 % |

Der Regret bleibt klar im Prozentbereich, in dieser Stichprobe sogar minimal (0,2-0,3
Prozentpunkte) höher als im Beharrungszustand -- der in Befund 52d vorausgesetzte
Vorbehalt ("realer Gewinn kleiner als berichteter Regret, weil der Fixpunkt nie erreicht
wird") trifft NICHT zu. Die Einordnung aus Befund 52 (Prozentbereich → Policy begründet,
keine Empfehlung) bleibt unter der Charge unverändert gültig.

**e) CB4 -- wandert das Optimum unter chargengemittelter Energie? Ja, aber nur im
unbeschränkten Regime.** Reoptimierung (`differential_evolution` + Nelder-Mead,
Zielfunktion = chargengemittelte Energie plus zeitlich gemittelte Straffunktion) an den
vier Eckpunkten des Störgrößengitters, `workers=-1` (Populations-Parallelität innerhalb
eines Punkts). Laufzeit deutlich über der ursprünglichen Schätzung (~6-10 s/Rollout wurde
zur Planungsgrundlage genommen; unter Last lagen einzelne Rollouts spürbar höher) --
32-47 min je Eckpunkt (Gesamt ca. 2,6 h), das 30-min-Abbruchkriterium bezog sich in der
Implementierung nur auf die DE-Phase (dort nie erreicht); die anschließende
Nelder-Mead-Verfeinerung hat ihr eigenes, aus der DE-Laufzeit geschätztes Zeitbudget in
drei von vier Fällen überzogen, weil die Kosten-je-Auswertung-Schätzung unter
Prozessorlast (paralleler `pytest`-Lauf während der ersten Ecke) zu optimistisch war --
kein Abbruch, aber ein Hinweis, dass diese Schätzung bei künftiger Wiederverwendung
lastfrei kalibriert werden sollte. Konvergenzstatus je Ecke unten ausgewiesen;
`de_converged=False` bei allen vieren ist planmäßig (siehe Befund 52, DE erschöpft ihr
Iterationsbudget konstruktionsbedingt).

| Ecke (`T_amb`, `phi`) | Stationär optimal (`T_in`/`u_gas`/`p_nozzle`) | Transient optimal (CB4) | Chargenmittel-Energie: stationär → CB4 | Konvergenz |
|---|---|---|---|---|
| 5 °C, 0,25 | 73,90 °C / 0,800 m/s / 1,008 bar | 71,19 °C / 0,835 m/s / 1,014 bar | 1,197 → 1,205 kW (**+0,6 %**) | DE nein, Verfeinerung nein |
| 5 °C, 0,85 | 76,48 °C / 0,800 m/s / 1,011 bar | 76,49 °C / 0,800 m/s / 1,000 bar | 1,234 → 1,234 kW (**±0,0 %**) | DE nein, Verfeinerung nein |
| 35 °C, 0,25 | 90,42 °C / 0,800 m/s / 1,006 bar | 60,00 °C / 1,091 m/s / 1,013 bar | 0,929 → 0,630 kW (**-32,2 %**) | DE nein, Verfeinerung JA |
| 35 °C, 0,85 | 89,02 °C / 0,800 m/s / 1,002 bar | 86,95 °C / 0,800 m/s / 1,013 bar | 0,912 → 0,882 kW (**-3,3 %**) | DE nein, Verfeinerung nein |

(Energiewerte sind der reine Chargenmittelwert von `energy_target`, ohne Straffunktion,
frisch aus dem jeweiligen Rollout nachgerechnet -- an allen vier Punkten `max(product_
moisture_kg_per_kg) <= 0,200002` und Fluidisierungsfenster/Betttemperatur unverletzt,
die Straffunktion trägt also praktisch nichts zu den berichteten Energiewerten bei.)

Klarer Regime-Unterschied: in den beiden **kalten** Ecken (Feuchtegrenze bindet am
Fixpunkt, Befund 52c) ändert sich praktisch nichts (±0,6 %) -- die Feuchtegrenze pinnt
den Betriebspunkt so eng, dass eine transiente Zielfunktion keinen nennenswerten
zusätzlichen Spielraum eröffnet, und in der kälteren/trockeneren der beiden Ecken findet
die (unkonvergierte) Suche innerhalb ihres Budgets nicht einmal eine Verbesserung
gegenüber dem bereits guten stationären Sollwert. In den beiden **warmen** Ecken
(unbeschränktes Regime, Befund 52c: Optimum an der unteren `gas_velocity`-Box-Grenze)
wandert das Optimum, an der trockenen Ecke drastisch: `inlet_temperature_celsius` fällt
von 90,4 °C auf die Box-UNTERGRENZE (60,0 °C), `gas_velocity_m_per_s` steigt von der
Box-UNTERGRENZE (0,800) auf 1,091 m/s -- eine chargengemittelte Energieeinsparung von
32 %, bei einer vollständig konvergierten Verfeinerung. Mechanismus: der nominale
Anfangszustand (55 °C, Befund 52b-Fixpunkte hier bei 30-45 °C) liegt in dieser Ecke weit
über dem stationären Optimum; eine niedrigere Zulufttemperatur kombiniert mit höherer
Gasgeschwindigkeit nutzt die überschüssige thermische/konvektive Kapazität der
Anfahrphase, statt sie (wie der stationäre Sollwert) ungenutzt zu lassen.

Antwort auf die CB4-Leitfrage (RL-Umgebung auf `solve_steady_state` oder transiente
Rollouts?): **regimeabhängig.** Im feuchtegrenzen-gebundenen (kalten) Regime ist
`solve_steady_state` als Grundlage ausreichend -- der transiente Zusatzaufwand
(Sekunden vs. Minuten je Auswertung) erkauft dort keine relevante Verbesserung. Im
unbeschränkten (warmen) Regime unterschätzt `solve_steady_state` das erzielbare
Potenzial um bis zu 32 % -- eine RL-Umgebung, die dort trainiert oder bewertet wird,
braucht transiente Rollouts. Bei nur einem von vier Eckpunkten vollständig konvergierter
Verfeinerung ist das ein Anhaltspunkt, kein belastbarer Zahlenwert je Ecke -- für eine
belastbare Aussage wäre eine lastfrei kalibrierte, vollständig konvergierte Wiederholung
nötig.

**f) Einordnung.** Von den vier in Befund 52d aufgeführten Vorbehalten in Richtung
Regret-Überschätzung entfällt der erste ("Beharrungszustand statt Charge") nach diesem
Befund -- er hatte in der Praxis keine Wirkung in die vermutete Richtung. Die
verbleibenden drei (perfekte Störgrößenkenntnis, keine Stellratenbegrenzung, unbelegte
`critical_moisture_kg_per_kg`) bleiben unberührt -- Befund 54 (DA1) beziffert den dritten:
Regret-Prozentbereich stabil über 0,10-0,30 kg/kg, Größenordnung nicht. Auch der hier neu
gefundene `Q_rec`-Chargenmittel-Vorbehalt hängt an einem unbelegten Platzhalter
(`bed_mass_solid_setpoint_kg`, Befund 54 DA2): er bleibt über den gesamten getesteten
`tau_4`-Bereich bestehen (nie kleiner als 8,6 % Median), während Befund 52s Regret selbst
gegen denselben Parameter robust ist. Neu und nicht in Befund 52 antizipiert:
`Q_rec` im Chargenmittel liegt deutlich über dem Fixpunktwert -- relevant für jede
Wärmeverbund-Bilanzierung, die mit Beharrungszustandswerten rechnet, aber ohne Bezug zu
Regret oder Zulässigkeit. CB4 ergänzt eine eigene, regimeabhängige Einschränkung für
künftige RL-Arbeit: `solve_steady_state` als billige Umgebung ist im feuchtegebundenen
Regime unbedenklich, im unbeschränkten Regime (dort, wo Befund 52c bereits den
`gas_velocity`-Box-Rand als bindend auswies) potenziell irreführend um bis zu 32 %
chargengemittelte Energie -- diese Zahl selbst gilt nur mit dem in (e) genannten
Konvergenzvorbehalt.

**g) Reproduzierbarkeit.** `git/commit` und `git/dirty` je Lauf in
`artifacts/runs/study_operating_point_transient.json` (CB1-CB3) und
`artifacts/runs/study_operating_point_transient_cb4.json` (CB4, separater Lauf mit
`--cb4 --output ...`, identische CB1-CB3-Zahlen als eingebaute Gegenprobe der
Determinismus, siehe Log) -- beide tragen das `environment`-Feld
(`twinlink_ml.utils.tracking.environment_fingerprint`). Beide Läufe waren dirty (dieser
Befund/dieses Skript selbst uncommittet) -- unvermeidbar für ein Skript, das im selben
Commit wie sein Artefakt landet; der `git/diff_digest` macht den Stand nachträglich
identifizierbar.

**Provenienzvorbehalt (nachträglich, mit Entwarnung).** Beide Artefakte
(`study_operating_point_transient.json`, `..._cb4.json`) tragen `git/dirty=True` und den
Digest `e3b0c44298fc1c14` -- den sha256 des leeren Strings. Ursache ist die inzwischen
behobene Lücke, dass `git diff HEAD` ungetrackte Dateien nicht sieht: die erzeugenden
Skripte waren zum Laufzeitpunkt selbst ungetrackt, und der geloggte `git/commit` liegt in
beiden Fällen VOR dem ersten Commit des jeweiligen Skripts. Entwarnung: der Mechanismus ist
damit vollständig erklärt (die noch nicht committete Skriptdatei selbst, kein unbekannter
Diff) statt eine offene Frage zu bleiben; derselbe Mechanismus wurde bei Befund 54 an einem
der fünf betroffenen Skripte (`probe_qrec_warmstart.py`) durch tatsächliche Reproduktion aus
dem sauberen Baum bestätigt (Abweichung 1e-13 bis 1e-14, Gleitkommarauschen, s. dort) -- die
Lücke selbst ist mit `git/diff_digest` (s. o., erfasst jetzt auch ungetrackte
Dateiinhalte) für künftige Läufe geschlossen, ändert aber nichts an den oben berichteten
Zahlen.

### 54. Robustheit von Befund 52/53 gegen die zwei unbelegten Platzhalter `critical_moisture_kg_per_kg` und `tau_4` (Bettmasse-Sollwert) -- Regret robust, Chargenmittel-`Q_rec` nicht

**a) Zweck und Abgrenzung.** Befund 52/53 hängen an zwei Zahlenwerten ohne Anlagendaten:
`material.critical_moisture_kg_per_kg` (0,200 kg/kg, Laborniveau-Platzhalter,
`greybox_gab_placeholder.yaml`-Dateikopf) und `tau_4 = bed_mass_solid_setpoint_kg /
solid_inflow` (16 000 s, aus zwei TODO-Feldern -- `geometry.bed_mass_solid_setpoint_kg`
und implizit dem Feststoffzulauf). Statt beide zu erfragen, wird hier gemessen, wie stark
Befund 52 (Regret) und Befund 53 (Chargenverhalten) von ihnen abhängen -- kein
Netztraining, kein Reinforcement Learning, nur `solve_steady_state` plus die
unveränderte Optimierermechanik aus `study_operating_point_regret.py`/
`study_operating_point_transient.py` (beide per `importlib` geladen, keine Zeile dort
verändert), implementiert in `scripts/probe_critical_moisture_sensitivity.py` (DA1) und
`scripts/probe_tau4_sensitivity.py` (DA2). Toleranzkonvention (Zulässigkeit = Bindung,
`_TOLERANCE_MOISTURE_KG_PER_KG` etc.) unverändert aus dem Regret-Modul übernommen.

Laufzeitgrund für die Gitterreduktion: Befund 52 kostete ~2,5 h für EIN 5×5-Gitter; neun
Werte (fünf für DA1, vier für DA2) wären damit grob 22 h gewesen. Beide Studien laufen
deshalb auf einem 3×3- statt 5×5-Gitter (`ambient_temperature_celsius` ∈ {5; 20; 35} °C,
`ambient_relative_humidity` ∈ {0,25; 0,55; 0,85}); am jeweils aktuellen Parameterwert wird
zusätzlich das committete 5×5-Ergebnis aus Befund 52 als Anker verglichen. Gemessene
Gesamtlaufzeit: DA1 110,6 min, DA2 103,4 min (davon Chargenrollouts <1 min je Wert) --
zusammen 3,6 h, unter der 4-h-Marke, deshalb ohne Zwischen-Rückfrage vollständig
durchgelaufen.

**b) DA1 -- `critical_moisture_kg_per_kg` ∈ {0,10; 0,15; 0,20; 0,25; 0,30} kg/kg.** An
allen fünf Werten existiert ein an allen 9 (3×3) Gitterpunkten zulässiger fester
Betriebspunkt (BB3-Existenzfrage bleibt über den gesamten Bereich mit Ja beantwortet).

| `critical_moisture` [kg/kg] | Regret Median [kW / %] | P95 [%] | Max [kW / %] | bindet Feuchte (von 9) | fester Punkt (`T_in` / `u_gas`) |
|---|---|---|---|---|---|
| 0,10 | 0,627 / **42,16 %** | 98,13 % | 1,000 / **102,51 %** | 3 | 62,64 °C / 2,129 m/s |
| 0,15 | 0,442 / **42,01 %** | 55,08 % | 0,664 / **56,01 %** | 5 | 72,91 °C / 1,331 m/s |
| 0,20 (aktuell, 3×3) | 0,187 / 15,22 % | 23,02 % | 0,242 / 25,80 % | 6 | 90,41 °C / 0,800 m/s |
| 0,20 (Anker, 5×5, Befund 52) | 0,174 / 16,03 % | 22,20 % | 0,241 / 25,80 % | 6 (5×5: 10/25) | 90,41 °C / 0,800 m/s |
| 0,25 | 0,196 / 16,59 % | 24,73 % | 0,254 / 27,74 % | 6 | 89,95 °C / 0,800 m/s |
| 0,30 | 0,210 / 18,03 % | 26,71 % | 0,267 / 29,81 % | 6 | 89,48 °C / 0,800 m/s |

Anker-Vergleich am aktuellen Wert: 3×3-Median (15,22 %) liegt 0,81 Prozentpunkte unter dem
5×5-Median aus Befund 52 (16,03 %) -- Gitterrauschen, keine Diskrepanz, die vor der
Interpretation der übrigen vier Werte zu melden wäre (Auftragskriterium erfüllt: beide
Gitter einig in Größenordnung und Vorzeichen).

Gitteraufgelöste Bindungslage (3×3, `ambient_temperature_celsius` × `ambient_relative_
humidity`): bei 0,20-0,30 kg/kg bindet die Feuchtegrenze an der gesamten kalten/mittleren
Zeile (`T_amb` = 5 °C und 20 °C, alle drei Luftfeuchten) und nirgends bei 35 °C --
dieselbe Struktur wie in Befund 52c. Bei 0,15 kg/kg fehlt bereits ein Teil der 5-°C-Zeile
(bindet nur noch bei `phi` ≤ 0,55). Bei 0,10 kg/kg **kehrt sich das Bild um**: die
5-°C-Zeile bindet nirgends mehr, nur noch die 20-°C-Zeile (3 von 9 statt der bei höheren
Werten üblichen 6). Die im Auftrag explizit zu prüfende (nicht anzunehmende) Erwartung
"kleiner Wert bindet überall, großer nirgends" **trifft damit nicht zu** -- die
Bindungshäufigkeit ist nicht monoton fallend in `critical_moisture`, sondern wächst von
3/9 (0,10) über 5/9 (0,15) auf ein Plateau von 6/9 (0,20-0,30).

Mechanismus (an den Rohdaten nachvollzogen, nicht nur vermutet): bei 0,10 kg/kg sitzt
`inlet_temperature_celsius` an der kalten Zeile exakt an seiner Box-UNTERGRENZE (60,00 °C)
-- der Optimierer kann nicht weiter heizen, um die Feuchte exakt auf 0,10 zu treffen, und
weicht stattdessen auf höhere `gas_velocity_m_per_s` aus (1,38-2,13 m/s statt der sonst
üblichen 0,8 m/s-Untergrenze), was mehr trocknet als nötig (Feuchte 0,065-0,039 kg/kg,
klar unter 0,10, nicht bindend) und wegen der quadratisch in die Ventilatorleistung
eingehenden Gasgeschwindigkeit teuer ist (Energie bis 2,53 kW an diesem einen
Gitterpunkt, gegen 0,94-1,00 kW an der warmen Zeile) -- daher der drastische
Regret-Sprung auf über 100 % Max bei 0,10 kg/kg: ein einzelner fester Betriebspunkt muss
jetzt eine noch größere Spannweite instanzweiser Optima überbrücken als bei 0,20 kg/kg.

Kernfragen: **Regret bleibt über den gesamten geprüften Bereich klar im Prozentbereich**
(Median 15-42 %, nie im Promillebereich) -- die Einordnung aus Befund 52
("Prozentbereich → Policy begründet") ist damit über 0,10-0,30 kg/kg **stabil**. Die
GRÖSSENORDNUNG ist es nicht: unterhalb des aktuellen Werts (0,15 und darunter) verdoppelt
bis verdreifacht sich der Median gegenüber dem committeten 16 %-Wert, mit Einzelpunkten
über 100 % Regret. Der Regimewechsel selbst **verschwindet an keinem Ende des Bereichs**,
verschiebt seine Lage in der Störgrößenebene aber nicht-monoton, entgegen der (geprüften
und widerlegten) einfachen Erwartung.

**c) DA2 -- `tau_4` ∈ {4000; 8000; 16000 (aktuell); 32000} s über `geometry.
bed_mass_solid_setpoint_kg`.** Einziger Knopf, der `tau_4 = bed_mass_solid_setpoint_kg /
solid_inflow` ändert, ohne den Feststoffzulauf selbst (`spray_rate`, laut Befund 52 bewusst
fest) zu verschieben -- wörtlich aus `greybox.py::_product_mass_flow`:

> ``\dot m_{prod} = \max(0,\; \dot m_{s,in} + (m_s - m_{s,soll})/\tau_m)`` -- der
> Vorsteueranteil `\dot m_{s,in}` verhindert eine bleibende Regelabweichung.

Am Beharrungszustand ist `solid_mass = bed_mass_solid_setpoint_kg`, also
`tau_4 = bed_mass_solid_setpoint_kg / solid_inflow` (geschlossene Form,
`diagnostics._solid_residence_time_s`). Geprüft (vor dem Lauf, mit Nutzerfreigabe): am
Nominalpunkt bleiben `d50`, Fluidisierungsfenster und Gutsfeuchte exakt gleich, wenn nur
`bed_mass_solid_setpoint_kg` variiert wird -- ABER die Bettmasse geht linear in den
Schüttbett-Druckverlust (`bed_pressure_drop_pa`, `energy.py`) und damit in die
Ventilatorleistung ein: `power_total_kw` schwankt am Nominalpunkt um bis zu 4,6 % über den
vollen Bereich -- reale Modellphysik (mehr Bettmasse → mehr Druckverlust), kein Artefakt,
hier als Niveauversatz explizit mitgeführt, nicht verschwiegen.

*Stationärer Regret (BB1-BB3, wie Befund 52):*

| `tau_4` [s] | Bettmasse-Sollwert [kg] | Regret Median [kW / %] | Max [kW / %] | fester Punkt (`T_in` / `u_gas`) |
|---|---|---|---|---|
| 4000 | 0,500 | 0,190 / 15,67 % | 0,247 / 27,26 % | 90,41 °C / 0,800 m/s |
| 8000 | 1,000 | 0,190 / 15,53 % | 0,245 / 26,76 % | 90,41 °C / 0,800 m/s |
| 16000 (aktuell, 3×3) | 2,000 | 0,187 / 15,22 % | 0,242 / 25,80 % | 90,41 °C / 0,800 m/s |
| 16000 (Anker, 5×5, Befund 52) | 2,000 | 0,174 / 16,03 % | 0,241 / 25,80 % | 90,41 °C / 0,800 m/s |
| 32000 | 4,000 | 0,185 / 14,65 % | 0,234 / 23,99 % | 90,42 °C / 0,800 m/s |

Der Regret-Median schwankt über den gesamten 8-fachen `tau_4`-Bereich nur zwischen 14,65 %
und 15,67 % (unter 1,1 Prozentpunkten Spannweite) -- der von Ihnen vorhergesagte Effekt
trifft zu: der Niveauversatz aus der Ventilatorkopplung kürzt sich in der Differenz
(Regret = fest − instanzoptimal, gleiches `tau_4` auf beiden Seiten) weitgehend heraus.
**Befund 52s Regret-Aussage ist gegen `tau_4` im gesamten getesteten Bereich robust.**

*Chargenrollout (CB1, wie Befund 53, 28 800 s, `Delta_t = 25` s, je 9 Punkte mit den
dort optimalen Sollwerten):*

| `tau_4` [s] | `tau_coupled_slow` [s] (Diagnose) | `d50`-Abw. Chargenende | `bed_temp`-Abw. Chargenende | `Q_rec` Chargenmittel-Abw. Median | Max |
|---|---|---|---|---|---|
| 4000 | 90 | −0,14 % | 0,00 % | **8,58 %** | 13,09 % |
| 8000 | 180 | −4,63 % | 0,00 % | **17,45 %** | 26,55 % |
| 16000 (aktuell) | 360 | −19,60 % | 0,00 % | **35,18 %** | 53,46 % (Befund 53, 5×5: Median 35,9 %, Max 76,3 %) |
| 32000 | 720 | −32,65 % | 0,18 % | **70,33 %** | 106,34 % |

`solid_mass_kg` und `particle_moisture_kg_per_kg` bleiben an allen vier Werten praktisch
am Fixpunkt (< 1 % Abweichung, meist < 0,01 %) -- nur `d50`/Partikelanzahl zeigen die
erwartete, an `tau_4` gekoppelte Lag-Struktur (Befund 53b), wachsend von −0,14 % (ein
Viertel des aktuellen `tau_4`) auf −32,65 % (doppeltes `tau_4`).

**Die `Q_rec`-Chargenmittel-Abweichung wächst dagegen fast linear mit `tau_4`, obwohl
`bed_temperature_celsius` an allen vier Werten bereits am Chargenende nahezu konvergiert
ist (Tabelle oben, ≤ 0,18 %).** `d50` spielt in `Q_rec`
(`compute_recoverable_power_kw`, Funktion von `outlet_temperature`/`outlet_humidity`
allein) keine Rolle -- der eigentliche Treiber ist `tau_coupled_fast_s`/
`tau_coupled_slow_s` (die Zeitkonstanten der gekoppelten Wassermasse-/Betttemperatur-Mode,
`diagnostics.time_constant_report`), und die skalieren, gegen den Auftrags-Rahmen NICHT
isolierbar, mit demselben Parameter: `heat_capacity = solid_mass * cp_solid + water_mass *
cp_water` (`greybox._rhs`) wächst linear mit `bed_mass_solid_setpoint_kg`, also
verdoppelt sich `tau_coupled_slow` exakt mit `tau_4` (90 → 180 → 360 → 720 s, numerisch
geprüft). Der Chargenmittel-`Q_rec`-Fehler ist damit in diesem Modell strukturell **kein**
`tau_4`-Effekt (Feststoff-Verweilzeit) im engeren Sinn, sondern ein
Bettwärmekapazitäts-Effekt (Dauer des Anfahrtransienten, Befund 53d-Mechanismus): je
größer die Bettmasse, desto länger dauert es, bis die Anfangsbedingung
(`bed_temperature_celsius = 55` °C) auf den kälteren Fixpunkt (30-45 °C, Befund 52b)
relaxiert, und desto mehr trägt die wärmere/trockenere Anfangsphase zum Chargenmittel bei.
Beide Effekte hängen im Modell an DEMSELBEN Konfigurationsfeld
(`bed_mass_solid_setpoint_kg`) und sind mit dem hier verfügbaren Optimierer-Werkzeug nicht
sauber trennbar -- genau der im Auftrag vorgesehene Fall "andere Ursache als die
unvollständige Annäherung", hier aber durch Diagnose statt durch Spekulation belegt.

Kernfrage: **eine tau_4-Schwelle, ab der die stationäre Betrachtung für 28 800 s
unbedenklich wäre, gibt es im getesteten Bereich nicht** -- selbst beim kleinsten
getesteten Wert (ein Viertel des aktuellen) bleibt eine Chargenmittel-`Q_rec`-Abweichung
von 8,6 % (Median) bestehen, während der Regret (Befund 52s eigentliche Zielgröße)
unabhängig von `tau_4` durchgehend robust bleibt. Für `energy_target`/Regret ist die
stationäre Betrachtung bei jedem getesteten `tau_4` vertretbar; für eine
Wärmeverbund-Planung auf `Q_rec`-Basis (Befund 53s Vorbehalt) ist sie es bei keinem.

**d) DB1 -- ist der nominale Anfangszustand die Ursache der `Q_rec`-Abweichung? Ja,
vollständig.** Nachtrag, kein neuer Sweep: derselbe Chargenrollout am aktuellen Wert
(`tau_4 = 16 000` s, 3×3-Gitter, dieselben in (c) gefundenen optimalen Sollwerte je
Punkt, unverändert wiederverwendet), aber mit dem `solve_steady_state`-Beharrungszustand
als Anfangszustand statt dem nominalen (`bed_temperature_celsius = 55` °C etc.),
implementiert in `scripts/probe_qrec_warmstart.py`.

| Start | Abweichung Chargenmittel (Median / Max) | Anfahrzeit bis 5 %-Band (Median / Max) |
|---|---|---|
| nominal (wie DA2c) | +35,18 % / 53,46 % | 28 800 s / 28 800 s (nie innerhalb der Charge) |
| eingeschwungen (`solve_steady_state`) | −0,00 % / 0,00 % | 0 s / 0 s |

Beim eingeschwungenen Start ist die Abweichung an allen 9 Punkten null bis auf
Löserrestfehler (< 0,001 %) -- erwartbar, da eine exakte Nullstelle von
`evaluate_derivative` auch ein exakter Fixpunkt des RK4-Zeitschritts ist (alle
Stufenableitungen bereits null). Beim nominalen Start bleibt die LAUFENDE
(kumulative) Mittel-`Q_rec` an JEDEM der 9 Punkte bis zum letzten Zeitschritt außerhalb
des 5-%-Bands -- sie konvergiert nicht etwa langsam, sondern gar nicht innerhalb der
28 800 s, weil ein einfacher Chargenmittelwert die frühe, warme Anfahrphase nie
"vergisst" (ihr Beitrag zur kumulativen Summe schrumpft nur wie `1/t`, nicht
exponentiell mit der zugrundeliegenden Zeitkonstante des Zustands selbst).

**Die Erklärung aus Befund 53 (der nominale Anfangszustand, nicht `tau_4`, treibt die
`Q_rec`-Abweichung) trifft zu, vollständig.** Konsequenz: das Chargenmittel ist bei
kontinuierlichem Betrieb die falsche Statistik für `Q_rec` -- es mittelt einen
Anfahrvorgang mit, den die Anlage im eingeschwungenen Normalbetrieb (Charge folgt auf
Charge, nicht auf einen Kaltstart) nicht durchläuft. Für eine Wärmenetz-/
Verbundauslegung ist der STATIONÄRE `Q_rec`-Wert die maßgebliche Zahl, nicht das
Chargenmittel aus Befund 53 -- der dortige Vorbehalt ("Wärmerückgewinnungsplanung
unterschätzt das chargengemittelte Potenzial") kehrt sich damit um: es ist das
Chargenmittel selbst, das für einen kontinuierlich laufenden Prozess die falsche
Referenzgröße ist, nicht der stationäre Wert.

**e) DB2 -- zwei Modellgrenzen, dokumentiert statt stillschweigend vorausgesetzt.**

*`bed_mass_solid_setpoint_kg` ist kein isolierter Verweilzeitparameter, sondern ein
Skalenparameter.* Er verschiebt gleichzeitig (1) `tau_4` (Feststoff-Verweilzeit, über
die Massenbilanz), (2) die Bettwärmekapazität (`heat_capacity = solid_mass * cp_solid +
water_mass * cp_water`, `greybox._rhs`) und damit `tau_coupled_fast_s`/
`tau_coupled_slow_s` (90 bis 720 s über den in (c) geprüften Bereich, exakt
proportional) und (3) den Schüttbett-Druckverlust und damit `power_total_kw` (um bis zu
4,6 %, (c)). Eine unabhängige Variation NUR der Verweilzeit ist mit diesem Modell nicht
möglich -- dafür bräuchte es Bettmasse und Produktmassenstrom als zwei getrennte,
unabhängig setzbare Größen (aktuell beide über denselben P-Regler und dieselbe
Bettmasse-Konstante gekoppelt, `greybox._product_mass_flow`). DA2/DB1 (c, d) sind damit
Aussagen über diesen SKALENPARAMETER, nicht über die Feststoff-Verweilzeit isoliert --
für Befund 52s Regret war das folgenlos (robust gegen den ganzen Skalenbereich), für
DB1s Erklärung des `Q_rec`-Chargenmittel-Fehlers ist es der eigentliche Mechanismus
(die Bettwärmekapazität, nicht `tau_4` selbst, bestimmt die Dauer des Anfahrtransienten).

*Der aktuelle `critical_moisture_kg_per_kg`-Platzhalter (0,20 kg/kg) liegt am GÜNSTIGEN
Ende des in (b) geprüften Bereichs.* Regret-Median 15,22 % bei 0,20 kg/kg gegen 42,16 %
bei 0,10 kg/kg und 18,03 % bei 0,30 kg/kg -- der aktuelle Wert ist im gesamten
geprüften Bereich (0,10-0,30 kg/kg) der Wert mit dem KLEINSTEN Regret. Die qualitative
Aussage aus Befund 52 ("Prozentbereich, Policy begründet") ist damit einseitig
abgesichert: jeder andere in (b) getestete Wert würde die Aussage eher verstärken
(höherer Regret) als sie zu entkräften. Das ist kein Beleg, dass 0,20 kg/kg richtig
liegt -- nur, dass ein Irrtum in Richtung des unbelegten Platzhalterwerts die Befund-52-
Schlussfolgerung eher stützt als untergräbt.

**f) Praktische Konsequenz für die offene Prozessparameter-Liste.**
`critical_moisture_kg_per_kg` bleibt SPE-seitig zu erfragen, aber nicht mehr, weil
Befund 52 sonst nicht interpretierbar wäre -- die qualitative Aussage (Regret im
Prozentbereich, Policy begründet) steht unabhängig vom genauen Wert und ist durch die
Lage des aktuellen Platzhalters am günstigen Rand des Bereichs (e) sogar einseitig
abgesichert; nötig ist die Nachfrage nur noch für die ABSOLUTHÖHE des Regrets
(Faktor 2-3 zwischen 0,10 und 0,20 kg/kg) und für BB4-artige Folgestudien, die an der
genauen Lage des Regimewechsels hängen. `geometry.bed_mass_solid_setpoint_kg`/
Feststoffzulauf (SPE: "Geometrie und Bettmasse der Laboranlage") bleibt für Befund 52s
Regret irrelevant (robust über den vollen getesteten Skalenbereich) -- die Nachfrage
danach ist für DIESE Aussage nicht mehr entscheidend. Für eine `Q_rec`-/
Wärmeverbund-Aussage ist sie dagegen NACH DB1 ebenfalls entschärft: nicht der genaue
Zahlenwert von `bed_mass_solid_setpoint_kg` ist dort kritisch, sondern welche Statistik
verwendet wird -- der stationäre `Q_rec`-Wert ist unter kontinuierlichem Betrieb die
richtige Referenz (d), unabhängig vom konkreten `tau_4`/Bettmasse-Wert. Erfragt werden
muss deshalb (SPE: Geometrie/Bettmasse) nur noch, falls tatsächlich chargenweiser statt
kontinuierlicher Betrieb vorliegt (Punkt 6 der offenen Fragen) -- dann ist das
Chargenmittel die richtige Statistik und die Bettmasse wieder relevant.

**g) Reproduzierbarkeit.** `scripts/probe_critical_moisture_sensitivity.py` →
`artifacts/runs/probe_critical_moisture_sensitivity.json`; `scripts/
probe_tau4_sensitivity.py` → `artifacts/runs/probe_tau4_sensitivity.json`; `scripts/
probe_qrec_warmstart.py` (DB1, baut auf dem `tau_4=16000`-Eintrag von
`probe_tau4_sensitivity.json` auf, kein neuer Sweep) → `artifacts/runs/
probe_qrec_warmstart.json`. Alle drei tragen `environment` (`git/commit`, `git/dirty`) je
Lauf; alle Läufe waren dirty (dieser Befund und diese Skripte selbst uncommittet, wie in
Befund 53g) -- der `git/diff_digest` macht den Stand nachträglich identifizierbar.
Budget/Toleranzen identisch zu Befund 52/53, einzige Änderungen sind die Gitterauflösung
(3×3 statt 5×5, mit 5×5-Anker am jeweils aktuellen Wert, DA1/DA2) und der Anfangszustand
(DB1).

**Provenienzvorbehalt (nachträglich, mit Entwarnung).** Alle drei Artefakte
(`probe_critical_moisture_sensitivity.json`, `probe_tau4_sensitivity.json`,
`probe_qrec_warmstart.json`) tragen `git/dirty=True` und den Digest `e3b0c44298fc1c14` --
den sha256 des leeren Strings. Ursache ist die inzwischen behobene Lücke, dass `git diff
HEAD` ungetrackte Dateien nicht sieht: die erzeugenden Skripte waren zum Laufzeitpunkt
selbst ungetrackt, und der geloggte `git/commit` liegt in allen drei Fällen VOR dem ersten
Commit des jeweiligen Skripts. Entwarnung: `probe_qrec_warmstart.py` wurde aus dem
sauberen Baum (nach dessen Commit) erneut ausgeführt und gegen den hier berichteten Lauf
verglichen -- Abweichung 1e-13 bis 1e-14, also reines Gleitkommarauschen, keine inhaltliche
Abweichung. Der dirty-Zustand war ausschließlich das noch nicht committete Skript selbst,
nicht eine unentdeckte Codeänderung im Rest des Baums. Die Lücke ist mit `git/diff_digest`
(s. o., erfasst jetzt auch ungetrackte Dateiinhalte) für künftige Läufe geschlossen, ändert
aber nichts an den oben berichteten Zahlen für alle drei Skripte.

**Addendum (2026-09-28, English -- see Befund 64 for the full analysis).**
`probe_critical_moisture_sensitivity.py`'s `build_variant_config` replaces
`material.critical_moisture_kg_per_kg` directly for DA1's five tested values (0.10-0.30
kg/kg). That field has a double role in this model: it is not only the quality-constraint
threshold checked in `study_operating_point_regret.py::check_constraints`, but also `X_cr`
in `thermo.normalised_moisture` -- the drying-kinetics falling-rate-period parameter, used
inside `solve_steady_state` itself (`config.py`'s own comment on the field: "Trocknungskinetik:
normierte Trocknungskurve, linear fallende Rate"). DA1's sweep therefore moved the drying
kinetics and the quality limit together at every tested value, not the quality limit alone.
The reported regret-vs-`critical_moisture` curve (b) -- in particular the claim that the
regret magnitude is not robust below 0.15 kg/kg, and the "b) kehrt sich das Bild um" binding-
pattern reversal at 0.10 kg/kg -- mixes both effects and cannot be attributed to the quality
constraint in isolation. Not recomputed here for DA1's full range; a decoupled repeat
(kinetics fixed at 0.20 kg/kg, only the constraint threshold varied) exists for the narrower
0.12-0.20 kg/kg range in Befund 64's addendum, using the same decoupling method
(`scripts/probe_margin_energy_cost_decoupled.py`).

### 55. Braucht man das Anregungsexperiment? Konstant trainierter Zwilling zu Befund 51 auf demselben Konstantlauf-Holdout -- überwiegend nein: Betriebsdaten genügen im getesteten Budgetbereich nicht

**Frage dahinter.** Alle bisherigen Trainingsläufe dieses Projekts (Befund 51 eingeschlossen)
trainieren auf ANGEREGTEN Pools (Stellgrößen springen/PRBS/rampen innerhalb jedes Laufs,
`constant_run_fraction=0.10`). Kein Arm trainierte bislang auf rein konstanten Läufen. Am
Anwendungspartner entscheidet genau das zwischen "Modell aus vorhandenen Betriebsaufzeichnungen
bauen" und "eigener Anlagenversuch mit Anregung nötig". Dieser Befund liefert den fehlenden Arm.

**Zwei Betriebsarten, bislang nirgends explizit unterschieden:**
- TRAINIERT auf angeregt, AUSGEWERTET auf konstant (Befund 51, bisheriger Stand,
  `test_set="constant_holdout"` in `arch_learning_curve_projected.jsonl`).
- TRAINIERT auf konstant, AUSGEWERTET auf konstant (dieser Befund,
  `test_set="constant_holdout"` in `arch_learning_curve_constant_trained.jsonl`, neu).

**Konfiguration.** Zwillingsskript zu Befund 51 (`run_arch_constant_trained_learning_curve_worker.py`
+ `..._sweep.py`, Commit `1b85ce4`): identische `DataSpec` (Budgetpunkte `n_train∈{8,16,32,64}`,
5 Seeds, Zielkanäle `energy_target`/`product_moisture_kg_per_kg`/`bed_temperature_celsius`,
`history_length=32`, `horizon=36`, Architekturen `persistence,static,arx,gru,lstm`,
Fingerprint `2d2679eeed716577`), einzige Änderung `constant_run_fraction=1.0` im Trainingspool.
Ausgewertet auf **demselben** Konstantlauf-Holdout wie Befund 51 (`HOLDOUT_SEED=999`, identisches
`split/digest`) -- gleiche Testmenge, verschiedene Trainingsart. `test_set` der eigenen
Pool-Testmenge heißt hier `"own_pool"`, nicht `"excited"` (der Pool ist konstant, `"excited"`
wäre irreführend). Pilot (1 Seed, `n_train=8`) vorab bestanden: `git/commit` ein 40-stelliger
Hash, `git/dirty=False`, `data/train_constant_run_fraction=1.0` in allen 10 Pilot-Records
(LA1-Feld, s. u.). Arbeitsbaum vor dem Vollauf sauber, während der 40,5 Minuten kein weiterer
Schreibzugriff.

**LA1 -- Trainingsart im Record eindeutig gemacht.** `_log_model`/`_log_result` bauten `params`
bislang nur aus `**excited_spec.as_parameters()` -- `data/constant_run_fraction` spiegelte zwar
bereits die TRAININGS-Spec, aber unter einem Namen, der für jeden `test_set`-Wert desselben
Records identisch bleibt (auch für `test_set="constant_holdout"`), die Trainingsart also nicht
mehr sicher erkennen ließ. Neues, additives Feld `data/train_constant_run_fraction` (explizit aus
`excited_spec.constant_run_fraction`/`train_spec.constant_run_fraction`) in allen drei Skripten mit
demselben Muster (`run_arch_learning_curve_worker.py`, `run_arch_constant_holdout_worker.py`,
`run_arch_multipool_eval.py`, Commit `27ab699`) -- rückwärtskompatibel, da `RunRecord.params` ein
ungeprüftes `Mapping` ohne Schema-Validierung ist; ältere Records ohne das Feld bleiben gültig,
dürfen sein Fehlen aber nicht als `0.0` interpretieren. Dasselbe Muster kommt in neun weiteren,
für diesen Auftrag nicht angefassten Skripten vor (`run_extra_trees_split_comparison.py`,
`run_mamba_constant_holdout_worker.py`, `run_mamba_reduced_comparison.py`,
`run_outlet_temperature_gru_protocol.py`, `run_tbed_exhaust_cost_probe.py`,
`run_tbed_exhaust_sweep_worker.py`, `run_transformer_reduced_comparison.py`,
`run_xlstm_constant_holdout_worker.py`, `run_xlstm_reduced_comparison.py`).

**Laufzeit -- Schätzung grob zu hoch, nicht zu niedrig.** Vorab geschätzt (Referenz Befund 51,
gleiche Struktur, gleiche Rechenkosten je Fit erwartet): ~4 h. Tatsächlich: **40,5 Minuten**
(2429,3 s, 5 Jobs in 2 Batches bei `--parallel=4`, 200 Records) -- Faktor ~6 schneller als
geschätzt. Ursache: Early Stopping (`patience=20`, `max_epochs=200`, unverändert gegenüber
Befund 51) greift im konstant trainierten Arm systematisch viel früher.

**Früh erschöpft, nicht schneller -- die eigentlich aussagekräftigste Zahl dieses Befunds.**
Mittlere Epochenzahl je Fit (Median über 5 Seeds, gemittelt über die drei Zielkanäle):

| `n_train` | GRU konstant | GRU angeregt | LSTM konstant | LSTM angeregt |
|---|---|---|---|---|
| 8  | 24,3 | 166,3 | 23,0 | 170,7 |
| 16 | 25,3 | 176,0 | 23,7 | 167,7 |
| 32 | 53,0 | 165,7 | 26,7 | 174,3 |
| 64 | 78,7 | 150,0 | 99,7 | 144,3 |

Der angeregte Arm nutzt bei `n_train∈{8,16,32}` durchgehend nahezu das volle `max_epochs=200`-Budget
(einzelne Seeds erreichen exakt 200, s. Rohdaten); der konstante Arm bricht bei denselben
Budgetpunkten nach 23-25 Epochen ab -- ein Achtel bis ein Siebtel. Das ist **keine** Recheneffizienz
des konstanten Arms, sondern eine direkte Messung des Informationsgehalts der Trainingsdaten:
Early Stopping greift, weil die Validierungs-Loss über `patience=20` Epochen nicht mehr sinkt --
bei konstanten Läufen erschöpft sich das lernbare Signal fast sofort. Erst bei `n_train=64` nähert
sich die Epochenzahl an (78,7-99,7 gegen 144,3-150,0), aber auch dort bleibt der konstante Arm
deutlich darunter.

**Vorbehalt: Plateau oder Überanpassung nicht unterscheidbar.** `train_loss_ratio_last10`
(TRAIN-Verlust der letzten 5 gegen die vorherigen 5 Epochen) liegt am Abbruchpunkt bei 0,71-0,93
(Pilot-Record) -- der TRAININGSverlust fällt zum Zeitpunkt des Abbruchs noch spürbar. Das ist mit
BEIDEN Lesarten vereinbar: bei echtem Plateau ebenso wie bei beginnender Überanpassung (Training
fällt weiter, Validierung stagniert -- Lehrbuchbild), die Kennzahl entscheidet also nichts und ist
hier nur als Näherung zu lesen, nicht als Plateau-Beleg. Die eigentlich nötige Größe --
Validierungs-Loss am Abbruchpunkt gegen `best_loss` -- wird nirgends persistiert:
`TorchSequenceModel.training_history` (Verlust je Epoche inkl. `val_loss`) ist eine reine
Laufzeit-Property, geht weder in `training_diagnostics()` (und damit den Record) noch in
`save_checkpoint()`/`CompositeSequenceModel.state_dict()` (geprüft: nur Netzgewichte plus
Schema-Metadaten) ein. Schema-Lücke derselben Klasse wie LA1, hier nur vorgemerkt, NICHT
umgesetzt (Schemaänderung mitten im/direkt nach einem provenienzsensitiven Sweep hätte die
Records dieses Befunds inkonsistent gemacht): kleinstes schließendes Feld wäre `val_loss` am
Abbruchpunkt plus `best_epoch` im Record -- kein voller Verlauf nötig.

**Vergleich -- Faktor besser als Persistenz (derselbe Arm, `nrmse_run_median`, h=36,
`test_set="constant_holdout"`, Median über 5 Seeds):**

| Kanal | Modell | Arm | n=8 | n=16 | n=32 | n=64 |
|---|---|---|---|---|---|---|
| `product_moisture_kg_per_kg` | GRU | konstant | 0,15x | 0,28x | 0,39x | 0,52x |
| `product_moisture_kg_per_kg` | GRU | angeregt | 0,33x | 0,69x | 0,89x | 1,31x |
| `product_moisture_kg_per_kg` | LSTM | konstant | 0,25x | 0,27x | 0,34x | 0,71x |
| `product_moisture_kg_per_kg` | LSTM | angeregt | 0,68x | 0,73x | 1,05x | 1,70x |
| `energy_target` | GRU | konstant | 0,83x | 1,00x | 1,16x | 1,76x |
| `energy_target` | GRU | angeregt | 1,12x | 1,48x | 1,69x | 2,03x |
| `bed_temperature_celsius` | GRU | konstant | 0,90x | 0,86x | 0,95x | 1,03x |
| `bed_temperature_celsius` | GRU | angeregt | 1,08x | 1,22x | 1,43x | 1,48x |

(> 1x heißt besser als Persistenz. LSTM bei `energy_target`/`bed_temperature_celsius` im selben
Muster wie GRU, hier aus Platzgründen nicht tabelliert -- Rohdaten in beiden JSONL-Dateien.)

**Befund je Kanal, nicht ein einziges Urteil für den ganzen Arm:**
- **`product_moisture_kg_per_kg` (das im Projekt am meisten diskutierte Ziel): schärfere
  Aussage bestätigt, mit einer Einschränkung bei LSTM/`n=64`.** Der konstante Arm schlägt
  Persistenz bei KEINEM getesteten Budgetpunkt (0,15x-0,71x, also durchgehend schlechter als die
  triviale Baseline), der angeregte Arm ab `n_train≈32` (0,89x-1,05x) und deutlich ab
  `n_train=64` (1,31x-1,70x). Die Lücke schließt sich nicht mit wachsendem Budget -- das
  NRMSE-Verhältnis konstant/angeregt bleibt über alle vier Budgetpunkte bei 2,2-3,1 (GRU) bzw.
  2,4-3,1 (LSTM), ohne erkennbaren Trend. Geprüft gegen die Seed-Spannweite (Ersatzmaßstab, s. u.):
  die Differenz der Mediane übersteigt die größere der beiden Arm-internen Seed-Spannweiten bei
  GRU ab `n_train=32` (Differenz 0,64 gegen Spannweite 0,55 bei `n=32`; 0,52 gegen 0,43 bei
  `n=64`) und bei LSTM bei `n=32` (0,87 gegen 0,54) -- bei LSTM/`n=64` dagegen NICHT (Differenz
  0,37 kleiner als die dortige Seed-Spannweite des konstanten Arms von 0,66): dort ist die Lücke
  vorhanden, aber mit nur 5 Seeds nicht sauber von Seed-Rauschen zu trennen. Bei `n_train∈{8,16}`
  ist die Seed-Spannweite in beiden Armen so groß, dass keines der beiden Modelle dort eine von
  Rauschen trennbare Aussage zulässt.
- **`energy_target`: uneindeutig, eher Richtung Aufholen, aber nicht von der Seed-Streuung
  trennbar.** Verhältnis konstant/angeregt fällt von 1,35 (`n=8`) auf 1,15 (`n=64`, GRU); die
  Differenz der Absolutwerte bei `n=64` (0,2218 gegen 0,1925) liegt in derselben Größenordnung wie
  die Seed-Spannweite innerhalb jedes Arms an dieser Stelle (0,0601 bzw. 0,0344) -- ob hier ein
  echtes Aufholen oder Seed-Rauschen vorliegt, ist mit den vorhandenen fünf Seeds nicht sicher zu
  trennen.
- **`bed_temperature_celsius`: kein Aufholen, tendenziell größer werdender Abstand.** Verhältnis
  steigt von 1,20 (`n=8`) auf 1,50 (`n=32`) und bleibt bei 1,43 (`n=64`) -- keine Konvergenz
  Richtung Parität im getesteten Bereich.

**Poolstreuung fehlt -- Vorbehalt, nicht behoben.** Der geforderte Vergleichsmaßstab
(`scripts/run_arch_multipool_eval.py`, drei unabhängige Konstantlauf-Pools, Seeds 997/998/999)
ist nicht lauffähig: der Aufruf schlägt vorab an der Konsistenzprüfung der 20 geladenen
Checkpoints (`gru_seed{1..10}.pt`/`lstm_seed{1..10}.pt`, aus einem früheren, nicht mit diesem
Auftrag verbundenen Experiment) mit `CheckpointSpecMismatchError` fehl -- die Checkpoints tragen
uneinheitliche `target_names` (`gru_seed4.pt` u. a. mit `d50_um` als drittem Kanal statt
`bed_temperature_celsius`). Kein Schreibzugriff vor dem Absturz (geprüft: `git status --porcelain`
vor und nach dem Aufruf identisch), also ohne Risiko für den Arbeitsbaum ausprobiert, aber NICHT
repariert (nicht Teil dieses Auftrags). Ersatzweise oben die Seed-Spannweite (Median über 5 Seeds,
Spannweite in eckigen Klammern) als schwächerer, verfügbarer Maßstab herangezogen -- sie misst nur
Seed-zu-Seed-Streuung innerhalb EINES Pools, nicht Pool-zu-Pool-Streuung, und ist damit
systematisch kleiner als die eigentlich verlangte Poolstreuung. Ein Unterschied, der schon die
Seed-Spannweite unterschreitet (wie bei `energy_target`), ist erst recht nicht von der größeren
Poolstreuung zu trennen; ein Unterschied, der die Seed-Spannweite klar übersteigt (wie bei
`product_moisture_kg_per_kg`/GRU ab `n_train=32`, s. u.), ist zumindest ein notwendiges, wenn
auch nicht hinreichendes Indiz für einen über die Poolstreuung hinausgehenden Effekt -- bei
`product_moisture_kg_per_kg`/LSTM gilt das bei `n_train=64` gerade NICHT (Detailrechnung s. u.),
der Ersatzmaßstab liefert also selbst innerhalb eines Kanals kein einheitliches Bild.

**LA4 -- vorab festgelegte Erwartung, Ergebnis.** Erwartung: deutliche Verschlechterung bei
kleinem Budget, Aufholen bei `n_train=64` würde für "Betriebsdaten genügen" sprechen, Ausbleiben
für "dynamischer Gehalt fehlt". Eingetreten: **überwiegend kein Aufholen** --
`product_moisture_kg_per_kg` (das im Projekt zentrale Ziel) zeigt eine über den ganzen
Budgetbereich flach bei Faktor 2,2-3,1 verlaufende Lücke, `bed_temperature_celsius` eine mit dem
Budget eher wachsende Lücke; nur `energy_target` nähert sich an, aber nicht klar von der
Seed-Streuung getrennt. Die Lernkurven laufen für `product_moisture_kg_per_kg` und
`bed_temperature_celsius` im getesteten Bereich **parallel** (kein Schnittpunkt, kein
konvergierender Trend), für `energy_target` **möglicherweise konvergierend**, aber nicht
gesichert. Gesamtaussage für den Anwendungspartner: die schärfere der beiden vorab formulierten
Lesarten trifft überwiegend zu -- dem konstant trainierten Arm fehlt im getesteten Budgetbereich
nicht (nur) Abdeckung, sondern dynamischer Gehalt; ein Anregungsexperiment ist für das
projektzentrale Feuchteziel durch mehr rein konstante Betriebsdaten (bis `n_train=64`) nicht
ersetzbar. Ob sich das bei deutlich größerem Budget (`n_train≫64`) ändert, ist mit diesem Befund
nicht beantwortet -- die getestete Obergrenze ist dieselbe wie in Befund 51 (Zweierpotenzen bis
64), keine Erweiterung war Teil dieses Auftrags.

**Vorbehalt zur Konstruktion des konstanten Trainingspools.** Die Stellgrößen sind INNERHALB
jedes Laufs konstant, ZWISCHEN den 128 Pool-Läufen variieren sie weiterhin (Sobol-Design wie im
angeregten Arm, nur ohne Sprünge/PRBS/Rampen innerhalb der Laufdauer). Ein realer
Betriebsdatensatz hätte möglicherweise auch zwischen den Läufen wenig Variation (wenige,
selten gewechselte Sollwerte über lange Zeiträume) -- dieser Arm ist also noch der GÜNSTIGE Fall
für Betriebsdaten, nicht der ungünstigste. Die oben berichtete Lücke zum angeregten Arm dürfte bei
einem realistischeren, zwischen den Läufen ebenfalls wenig variierenden Betriebsdatensatz eher
größer als kleiner ausfallen.

**Reproduzierbarkeit.** `scripts/run_arch_constant_trained_learning_curve_worker.py` +
`..._sweep.py` (Commit `1b85ce4`) → `artifacts/runs/arch_learning_curve_constant_trained.jsonl`
(200 Records, alle `git/commit` identisch, `git/dirty=False`, `data/train_constant_run_fraction
=1.0`). Verglichen gegen `artifacts/runs/arch_learning_curve_projected.jsonl` (Befund 51,
unverändert). LA1-Schemaänderung: Commit `27ab699`.

### 56. Erste Validierung gegen ECHTE Anlagendaten (Anwendungspartner, Anlage F-5, Granulator D 3204) -- Messstellenzuordnung teils unbestätigt, `energy_target` an diesen Daten nicht validierbar -- **Betttemperatur-Treffer durch NB1 zurückgezogen (Luftstrom-Fehler)**

**Erstmals seit AP1.** Alle bisherigen Befunde (1-55) validieren ausschließlich gegen den
eigenen Simulator. Dieser Befund ist die erste Auswertung gegen zwei 24h-Exporte des
Anwendungspartners (Minutentakt, je 1441 Zeilen, zwei Produkte -- Dateibezeichnung "ACE"/"RID",
Produktnamen selbst nicht in diesem Dokument, siehe unten zur Vertraulichkeit) plus drei
Leitsystem-Screenshots zur Messstellenzuordnung. Code: `twinlink_ml.plant_data`
(Messstellen-Mapping, Commit `e3efc19`) und `scripts/study_plant_validation.py` (Auswertung,
Commit `88c0746`).

**Vertraulichkeit (NA1).** Die Rohdaten (Exporte, Screenshots) enthalten Kundendaten des
Anwendungspartners, unter anderem einen Produktnamen im Leitsystem -- sie bleiben außerhalb
des Repos. `twinlink_ml.plant_data.load_plant_export` nimmt den Dateipfad als Parameter;
`study_plant_validation.py` liest ihn ausschließlich aus den Umgebungsvariablen
`TWINLINK_PLANT_DATA_ACE`/`TWINLINK_PLANT_DATA_RID`. Committet sind nur abgeleitete Kennzahlen
(dieser Befund, das Mapping-Modul, das Auswertungsskript), keine Rohdaten.

**NA2 -- Messstellenzuordnung, gegen Screenshots geprüft statt übernommen.** Die vorgeschlagene
Zuordnung (31 exportierte Kanäle plus im Screenshot sichtbare, nicht exportierte Kanäle) steht
jetzt als `PlantChannel`-Datenstruktur in `twinlink_ml/plant_data/channels.py`, nicht als
Zeichenketten im Auswertungsskript. Drei Abweichungen vom Ausgangsstand:
- `TIRSAH-320414/415/416` als Betttemperatur **bestätigt**: die Leitsystem-Anzeige
  "Betttemperatur: 43-44 °C" trifft diese drei Kanäle exakt.
- `FIRAL-323805`/`FIRAL-323604` tragen in der Exportdatei die **fehlerhafte** Einheit `[°C]`
  (Zeile 2) -- Tag-Präfix (`FIR`, Durchflussanzeige) und Screenshot-Werte (17,98 bzw. 37,2 m³/h,
  beide nahe am Spaltenmittel) zeigen: beides sind Durchflüsse in m³/h. Im Loader korrigiert
  (`_UNIT_OVERRIDES`), nicht aus der Datei übernommen.
- `A-3215` (Bettmasse) und `DIR-325160` (Suspensionsdichte) sind **nicht** Teil der
  Exportdatei, obwohl live im Leitsystem sichtbar -- der ursprünglich angenommene Datenumfang
  ("soweit vorhanden") war zu optimistisch, siehe NA5.

**Nachtrag NA2 -- die Zuordnung `A-3215` = Bettmasse ist eine Vermutung, nicht bestätigt.**
Grundlage war, dass `A-3215` in allen drei Leitsystem-Screenshots 600,0 kg plus einen
kleinen wechselnden Wert zeigt; ein Regler ist nicht abgebildet. Das Präfix `A-` bezeichnet
in diesem Leitsystem an anderer Stelle Aggregate (`A-3234` Kompressorstation, `A-3203`
pneumatische Förderung) -- `A-3215` kann also auch ein Behälter oder eine Waage sein, nicht
zwingend eine Bettmasse-Messung. Der Anwendungspartner nannte mündlich 800 kg Bettmasse
(NA5); ob sich das auf denselben Messpunkt bezieht, ist offen. Zu klären beim
Anwendungspartner, nicht durch weitere Interpretation der Screenshots.

**NA3 -- Temperaturprofil über die Apparathöhe nicht monoton, Ursache nicht abschließend
geklärt.** `TRZSAH-320417/418/419` ("höhere Position") liegt in BEIDEN Betriebspunkten über der
Betttemperatur, nicht nur bei RID: ACE +0,89 K, RID +3,10 K (Mittel über 24 h). Kein Einzelfall
also, sondern ein reproduzierbares Muster. Auffällig: anders als die eng zusammenliegenden
Bett- und Zuluft-Triaden (Streuung zwischen den drei Sensoren typisch < 0,4 K) streut die obere
Trias selbst deutlich -- bei RID 1,78 K Median-Spannweite zwischen den drei Sensoren
(417/418/419: 45,6/47,4/46,1 °C im Mittel) gegen 0,43 K am Bett. Das spricht eher für eine
ortsabhängige Messung (Wandeinfluss, unterschiedliche Einbautiefe je Sensor) als für drei
Redundanzen derselben, gut durchmischten Kernströmung. Aus den Screenshots ist die Position von
417/418/419 nur als "im Zylinderteil von D 3204, oberhalb der Bett-Trias" zu bestätigen, nicht
Wand- vs. Gasmessung -- das bleibt eine offene Rückfrage an den Anwendungspartner. Für diesen
Befund und alle folgenden bleibt `TIRSAH-320414/415/416` die Betttemperatur des Modells, mit
diesem Vorbehalt.

**NA4 -- `outlet_duct_heat_loss_k`: Näherung innerhalb eines Betriebspunkts, kein
Leitungsverlust im engen Sinn.** Bett-Abluft-Versatz: ACE 6,59±0,19 K (Min/Max 6,18/6,93 K),
RID 8,87±0,59 K (Min/Max 7,78/9,77 K). Innerhalb einer Datei ist der Versatz praktisch konstant
(Standardabweichung 3-7 % des Mittels) -- passend dazu ist auch die Anregung praktisch null,
siehe unten. Zwischen den beiden Betriebspunkten verschiebt er sich aber um +2,3 K bei mehr als
doppelter Sprührate (934 → 2118 kg/h); bei RID korreliert der Versatz innerhalb des Tages mit
der Sprührate (+0,53) und (gegenläufig) mit der Umgebungstemperatur (-0,53), bei ACE ist beides
praktisch unkorreliert (< 0,2) -- konsistent mit der ohnehin geringen Anregung bei ACE. Antwort
auf die Kernfrage: **eher betriebspunktabhängig als universell konstant**, ein einzelner fester
Parameter ist also eine Näherung, kein Naturgesetz. Das passt zum bereits im Code stehenden
Vorbehalt (`ExhaustConfig.outlet_duct_heat_loss_k`-Docstring): das Modell referenziert alle
Abströme auf die (ideal durchmischte) Betttemperatur, real kann das Austrittsgas schon unter der
Partikeltemperatur liegen und die Verdampfung im Freeboard weiterlaufen -- der Parameter
absorbiert mehrere Effekte, sein Name ist eine Vereinfachung. Kein Wert wird an dieser Stelle in
die Config übernommen; das ist ein eigener, hier nicht getroffener Kalibrierungsschritt.

**Befund 56a (stützt Befund 55 empirisch) -- Betriebsdaten enthalten praktisch keine
Anregung.** Variationskoeffizient der Stellgrößen INNERHALB je eines Betriebstages: Luftmenge
0,22 %/0,27 % (ACE/RID), Zulufttemperatur 0,21 %/0,21 %, Sprührate 2,01 %/0,12 %. Alle Werte
liegen zwischen 0,12 und 2,0 % -- gemessen, nicht angenommen. Zwischen den zwei Produkten sind
die Unterschiede dagegen groß (Sprührate 934 vs. 2118 kg/h, Betttemperatur 67,3 vs. 43,3 °C).
Das ist genau das Muster, das Befund 55 aus dem synthetischen Konstantlauf-Experiment ableitet
(dem konstant trainierten Arm fehlt dynamischer Gehalt, nicht nur Abdeckung) -- hier an echten
Anlagendaten wiedergefunden: Betriebsaufzeichnungen decken verschiedene Arbeitspunkte ab, aber
innerhalb eines Arbeitspunkts nahezu keine Dynamik.

**NA5 -- Beharrungszustand mit Anlagenparametern, ungetunte Annahmen, ein Parametersatz für
beide Betriebspunkte.** `solve_steady_state` mit gemessenen Eingangsgrößen (Zulufttemperatur,
Luftmenge, Sprührate, Umgebungstemperatur) und Modellplatzhaltern/dokumentierten Annahmen sonst
(vollständige Liste unten) -- **keine** Anpassung an die Messung:

| | gemessen T_bett | Modell T_bett (m=600/800 kg) | Abweichung |
|---|---|---|---|
| ACE | 67,3 °C | 66,8 °C (beide Bettmassen identisch) | -0,4 K |
| RID | 43,3 °C | 41,1 °C (beide Bettmassen identisch) | -2,1 K |

Beide Betriebspunkte werden mit DEMSELBEN Parametersatz auf unter 2,5 K getroffen, beide in
dieselbe Richtung (Modell unterschätzt leicht) -- bei n=2 kein belastbarer Trend, aber kein
Widerspruch zur Hypothese "ein Parametersatz reicht". Die A-3215/Partner-Diskrepanz (600 vs.
800 kg Bettmasse) ist für die Betttemperatur **irrelevant**: beide Werte liefern identisches
`T_bed` (die stationäre Energiebilanz hängt nicht spürbar an der Bettmasse) -- **das gilt
unabhängig davon, ob `A-3215` überhaupt die Bettmasse ist (NA2-Nachtrag: unbestätigt)**, da
beide Kandidatenwerte ohnehin durchgerechnet werden. Ordnungsabschätzung
bestätigt: Luftwärmestrom 24,5-26,4 kW/K (nicht 11 kW/K, wie zu Auftragsbeginn grob geschätzt);
Sprühratendifferenz (1184 kg/h, davon 70 % Wasser bei angenommenem Feststoffanteil) ×
Verdampfungsenthalpie ≈ 552 kW gegen Luftwärmestrom × gemessene Betttemperaturdifferenz
(25 kW/K × 24,0 K ≈ 600 kW) -- Größenordnung trifft auf ~10 %.

**Fluidisierungs-Zulässigkeit schlägt in beiden Fällen fehl -- kein Befund zur Energiebilanz.**
`solve_steady_state` markiert beide Lösungen als `feasible=False`
(`u_gas > u_terminal(d50)`, z. B. ACE 8,22 m/s gegen 3,59 m/s). Ursache: Partikeldurchmesser/
-dichte sind unveränderte Laborplatzhalter (`d50≈750 µm` aus `nucleus_diameter_m`/
`nucleation_solid_fraction`, unabhängig von Bettmasse und Sprührate, da `seed_mass_flow_kg_per_h
=0`) -- realistische Industriegranulate dieser Anlage sind vermutlich größer/dichter, wurden
aber NICHT angepasst (Auftrag: nicht an die Messung tunen). Die gute Temperaturübereinstimmung
und die fehlgeschlagene Fluidisierungsprüfung sind zwei unabhängige Aussagen des Lösers; letztere
braucht eine echte Partikelmessung (Siebanalyse/Laserbeugung), nicht Kalibrierung an T_bed.

**Numerische Randnotiz (kein physikalischer Eingriff).** Für RID reicht der
Solver-Standardbracket (`T_amb+10=21,5 °C`) nicht: unterhalb von ~28 °C befeuchtet sich das
Bett bei JEDER angenommenen Partikelfeuchte netto (keine Nullstelle der inneren Bisektion) --
bei der hohen Sprührate von RID (>2× ACE) existiert dort schlicht kein Trocknungsregime. Die
Lösung (41,13 °C) ist über die gesamte getestete Bracket-Untergrenze von 28 bis 40 °C identisch
bis auf 1e-9 -- eine echte, robuste Nullstelle, kein Bracket-Rand-Artefakt. Behoben durch
schrittweise Anhebung der UNTEREN Bracket-Grenze (`_solve_steady_state_widening_bracket`,
Schrittweite 5 K) bis zum ersten Erfolg -- eine Erweiterung des Suchbereichs, keine Änderung der
Bilanzgleichungen oder eines Parameters.

**Was in den Daten vollständig fehlt (Auftragsliste, geprüft).**
- **Kein Gasverbrauch, keine Heizleistung.** Die geloggten `Leistung V-3226`/`V-3246` sind
  Ventilator-STROM, keine Gasmenge. `energy.target` (`power_kw`/`specific_kwh_per_kg`) ist an
  diesen Daten **nicht validierbar**.
- **Keine Feuchte der Ansaugluft** -- nur `TIR-322412` (Temperatur). NA5 nimmt 75 % relative
  Feuchte an (unbestätigter Platzhalter, keine Quelle).
- **Keine Produktfeuchte, keine Partikelgröße** -- die Fluidisierungs-Zulässigkeitsprüfung
  (siehe oben) bleibt deshalb unvalidiert, unabhängig vom Temperaturtreffer.
- **Kein Produktmassenstrom** -- im Beharrungszustand ohnehin ein Lösungsergebnis, keine
  unabhängige Eingangsgröße; ohne Messung bleibt die Feststoff-Verweilzeit/das
  Partikelwachstum aber unvalidiert.
- Zusätzlich, erst beim Abgleich mit den Exportspalten sichtbar geworden: **Bettmasse
  (A-3215) und Suspensionsdichte (DIR-325160) sind nicht exportiert**, obwohl live im
  Leitsystem angezeigt (siehe NA2) -- beide nur aus dem Screenshot (dritter, nicht validierter
  Zeitpunkt) geschätzt.

**Vollständige Annahmenliste (NA5, `ASSUMPTIONS` in `study_plant_validation.py`).** Relative
Feuchte Ansaugluft 75 % (keine Quelle); Suspensions-Feststoffanteil 30 % (= Config-Nominalwert,
nicht anlagenspezifisch); Suspensionstemperatur 20 °C (Config-Default, Ansatz- statt
Außenlufttemperatur); Keimzufuhr 0 (kein Hinweis in den Daten); Partikeldichte/Keimdurchmesser/
Nukleationsanteil/Sorptionsisotherme/kritische Gutsfeuchte unverändert Laborplatzhalter;
`geometry.heat_loss_ua_w_per_k` unverändert (2 W/K, bei den hier beobachteten
Bett-Umgebungs-Differenzen von 30-80 K gegen 24,5-26,4 kW/K Luftwärmestrom < 200 W,
vernachlässigbar); Querschnittsfläche aus Bettdruckverlust geschätzt (`dp_bed=m·g/A`,
PDIRAH-320404=19,2 mbar, Screenshot, dritter Zeitpunkt) -- geht NICHT in die Energiebilanz ein,
siehe `study_plant_validation.py`-Moduldocstring, nur in die (fehlschlagende)
Fluidisierungsprüfung.

**Reproduzierbarkeit.** `scripts/study_plant_validation.py`, Rohdatenpfade über
`TWINLINK_PLANT_DATA_ACE`/`_RID`. `twinlink_ml.plant_data` Commit `e3efc19`, Auswertungsskript
Commit `88c0746`.

**NB1 -- Korrektur (extern angestoßen): der Betttemperatur-Treffer aus NA5 war ein
Rechenfehler-Artefakt, keine Validierung. Zurückgezogen.** Anlass: eine unabhängige
Handrechnung der Luftwärmebilanz kam für beide Betriebspunkte auf einen erheblich
kleineren maximal möglichen Wasserdampfanteil im Sprühstrom, als die dokumentierte
30-%-Feststoffannahme impliziert (318 bzw. 957 kW Verdampfungsbedarf gegen deutlich
weniger verfügbare Luftwärme). Nachgeprüft am Code: `_gas_velocity_m_per_s`
(`study_plant_validation.py`) wurde mit der angenommenen **relativen** Luftfeuchte
(0,75) aufgerufen, behandelte den Wert aber als **absolute** Feuchte in kg/kg -- sowohl
im Trocken/Feucht-Umrechnungsterm als auch in der Dichteberechnung. Das trieb den vom
Modell abgeleiteten Trockenluftmassenstrom auf das **2,19-fache (ACE)/2,20-fache (RID)**
des Werts, der aus Normvolumenstrom mal Normdichte folgt (ACE: 25,996 statt 11,847 kg/s;
RID: 24,071 statt 10,949 kg/s) -- entdeckt im Rahmen der Vorbereitung der M2-Synthese,
behoben in `study_plant_validation.py` (Commit `e06b6c7`, Regressionstest
`tests/test_study_plant_validation.py`).

**Was das für NA5 bedeutet.** Mit korrigiertem Luftstrom und unverändert 30 % Feststoff
(`scripts/study_plant_solid_fraction_prediction.py`, R2,
`artifacts/runs/study_plant_solid_fraction_prediction.json`):

| Punkt | m_bed | T_bett_Modell | Abweichung zur Messung | Fluidisierung |
|---|---|---|---|---|
| ACE | 600 kg | 48,55 °C | **−18,71 K** | Marge oben **−0,224 m/s** (weiterhin unzulässig) |
| ACE | 800 kg | 48,55 °C | **−18,71 K** | Marge unten +2,588 m/s, Marge oben +0,714 m/s (zulässig) |
| RID | 600/800 kg | -- | -- | **kein Beharrungszustand gefunden** (Bisektion bis 86,5 °C erfolglos) |

Die ursprünglich berichtete Übereinstimmung (0,4/2,1 K) war ein Artefakt des um Faktor
~2,2 überhöhten Luftstroms -- **"validiert die Enthalpiebilanz" ist falsch und wird
hiermit zurückgezogen.** Mit korrektem Luftstrom trifft das Modell bei unverändert 30 %
Feststoff die gemessene Betttemperatur an keinem der beiden Punkte.

**Fluidisierungsverletzung: teilweise, nicht vollständig, ein Artefakt desselben
Fehlers.** Bei ACE/m_bed=800 kg verschwindet die Verletzung mit korrigiertem Luftstrom
(zulässig); bei ACE/m_bed=600 kg bleibt sie knapp bestehen (−0,224 m/s); bei RID lässt
sich die Frage bei 30 % Feststoff gar nicht mehr stellen, weil in diesem Regime kein
Beharrungszustand existiert. **Keine pauschale Aussage "beide Punkte jetzt im Fenster"**
-- das Bild ist gemischt und hängt von der (unbekannten) Bettmasse und dem (jetzt als
falsch erkannten) Feststoffanteil ab.

**NB2 -- Feststoffanteil als Vorhersage, nicht als Anpassung (R3).** Bei gemessenem
Luftstrom und gemessener Zulufttemperatur legt die Betttemperatur im Beharrungszustand
im Wesentlichen den Wassergehalt der Suspension fest (die Bettmasse wirkt nur auf
Fläche/Fluidisierung, nicht auf die Energiebilanz, siehe Moduldocstring). Gitter-Scan
über den Feststoffanteil (0,05 bis 0,95, m_bed=800 kg), linear interpoliert auf die
gemessene Betttemperatur:

| Punkt | vorhergesagter Feststoffanteil | vorhergesagte Partikelfeuchte |
|---|---|---|
| ACE | **0,726** (Gitterklammer [0,725; 0,75]) | 0,0037 kg/kg |
| RID | **0,712** (Gitterklammer [0,70; 0,725]) | 0,0112 kg/kg |

Beide Produkte -- unabhängig voneinander, aus unterschiedlichen Betriebspunkten --
sagen einen Feststoffanteil um **71-73 %** vorher, deutlich über dem bisherigen
30-%-Platzhalter. **Das ist eine Vorhersage, keine Validierung:** der Feststoffanteil
war vorher unbekannt, die Betttemperatur ist der unabhängige Messwert -- ein Treffer an
dieser Stelle ist keine Bestätigung des Modells, solange der reale Feststoffanteil nicht
unabhängig bekannt ist.

**Unabhängiger Konsistenzhinweis (RID, kein Beweis).** Aus dem vorhergesagten
Feststoffanteil (0,712) und der im Leitsystem angezeigten Suspensionsdichte für RID
(1339 kg/m³, Screenshot) folgt über ideale Volumenadditivität
(1/ρ_Susp = w/ρ_Feststoff + (1−w)/ρ_Wasser, ρ_Wasser=998 kg/m³) eine implizierte
Feststoffdichte von **1554 kg/m³** -- innerhalb des für eine organische Formulierung
plausiblen Bereichs (grob 1300-1800 kg/m³). Das bestätigt nicht, dass 71-73 % der
tatsächliche Wert ist, spricht aber nicht dagegen.

**Ausdrücklich unberührt von dieser Korrektur.** NA3 (Temperaturprofil über die
Apparathöhe) und NA4/Befund 56a (Bett-Abluft-Versatz, Variationskoeffizient der
Stellgrößen 0,12-2,01 %) beruhen ausschließlich auf den Zeitreihen selbst, nicht auf
dem Grey-Box-Modell oder der fehlerhaften Luftstromumrechnung -- beide bleiben gültig
wie berichtet. Ebenso unberührt: die Fluidisierungs-Zulässigkeitsprüfung war bereits vor
NB1 als eigenständig fehlschlagend dokumentiert (unkalibrierter d50-Platzhalter) --
NB1 ändert, WELCHE Punkte an der Fluidisierungsgrenze scheitern, nicht dass die Prüfung
grundsätzlich von einem unbelegten Platzhalter abhängt.

**Nachtrag zu NB2: Feststoffanteil über den ganzen Apparat statt nur das Bett (F1).**
Die obige 71-73-%-Vorhersage löst die Energiebilanz bis zur Betttemperatur (Bilanzraum
= Bett). NA4 legt aber nahe, dass der Bett-Abluft-Versatz eher fortlaufende Verdampfung
oberhalb des Betts ist als ein reiner Leitungsverlust im engen Sinn -- dann ist die
energetisch passende Referenz für die Bilanz nicht T_bed, sondern die gemessene
T_abluft, und der Bilanzraum muss den ganzen Apparat umfassen (Luft/Suspension bis zur
Abluft, kein ODE-Steady-State, direkte algebraische Lösung). Mit dem Modell-Platzhalter
für den Wandverlust (`GeometryConfig.heat_loss_ua_w_per_k`, UA·(T_abluft−T_amb)) ergibt
sich:

| Punkt | Wandverlust (Modell-Platzhalter) | Wandverlust = 30 kW (Sensitivität) |
|---|---|---|
| ACE | **59,1 %** | **64,0 %** |
| RID | **64,0 %** (implizierte Feststoffdichte 1658 kg/m³) | **66,1 %** (implizierte Feststoffdichte 1624 kg/m³) |

Der Modell-Platzhalter für den Wandverlust ist selbst unbelegt (< 0,15 kW, siehe
Abschnitt 6 der Synthese) -- die 30-kW-Spalte ist eine reine Sensitivitätsrechnung, kein
Anspruch auf einen realistischeren Wert. Zusammen mit der bettmassebasierten R3-Zahl
(oben) ergibt sich pro Produkt eine Spanne statt eines Einzelwerts: **ACE rund 59-73 %,
RID rund 64-71 %** -- je nachdem, welcher Bilanzraum (Bett vs. ganzer Apparat) und
welcher Wandverlust angenommen wird. Beide Enden bleiben Vorhersagen, keine
Validierungen (siehe oben); die implizierten RID-Feststoffdichten (1554-1658 kg/m³)
liegen für die gesamte Spanne im plausiblen Bereich (1300-1800 kg/m³) und erlauben daher
keine Eingrenzung innerhalb der Spanne. Werte am oberen Ende (>70 %) sind für eine
sprühbare Suspension hoch zu bewerten.

**Reproduzierbarkeit (NB1/NB2).** `scripts/study_plant_validation.py`, Fix Commit
`e06b6c7`. `scripts/study_plant_solid_fraction_prediction.py`, R2/R3-Fassung Commit
`6f7262c`, F1-Erweiterung Commit `914233a`, Ergebnis
`artifacts/runs/study_plant_solid_fraction_prediction.json` (Commit `ae426f3`).

### 57. Kondensierender Wärmeübertrager (Model v3): Vorzeichenwechsel der Beladungsempfindlichkeit bestätigt, Netto-Zielfunktion zeigt die MA3-Gewichtung als dominanten Effekt, Sechs-Fälle-Modell hat einen Einheitenfehler -- der Feuchtekanal steigt von zweitrangig auf mitbestimmend

**Anlass.** Der Verbundpartner hat sein Wärmemodell erweitert -- sechs Verschaltungsfälle
statt vier, und ein KONDENSIERENDER Wärmeübertrager (Model v3) statt des bisher
angenommenen rein sensiblen Betriebs. Damit ändert sich die Bedeutung der Abluftbeladung
`Y_out` grundlegend: ohne Kondensation ist Feuchte ein Störfaktor (hebt den Taupunkt,
engt die sensible Kühlgrenze ein), mit Kondensation eine Ressource (liefert zusätzliche
latente Wärme). Fünf Teilaufträge (MA1-MA5), hier gemeinsam berichtet, weil sie
aufeinander aufbauen: der Kondensationsmodus (a) ist Voraussetzung für seine
Unsicherheitsanalyse (b) und für die Netto-Zielfunktion (c), die Sechs-Fälle-Rechnung (d)
ist eigenständig, nutzt aber dieselbe Formel wie (a).

**a) Kondensationsmodus im `heatnetwork`-Modul (MA1).** Zweiter Rückgewinnungsmodus
`HeatNetworkConfig.recovery_mode ∈ {"sensible", "condensing"}` (Default `"sensible"`,
bitgleich zum Stand vor diesem Befund -- Regressionstest, gesamte bestehende Testsuite
unverändert grün). Im kondensierenden Modus kühlt die Quelle bis
`cold_temperature_celsius` (`T_kalt`, neuer Platzhalter, Default 15 °C -- Begründung und
offene Frage siehe `docs/06_waermenetz.md`) statt bis `Taupunkt + dT_approach`, und
liefert zusätzlich die latente Wärme des dabei anfallenden Kondensats. In der Kaskade
(`evaluate_cascade`) wird die latente Leistung als fester Vorrat je Quelle geführt und
erst angezapft, sobald die sensible Kühlreserve einer Stufe erschöpft ist --
Kondensation setzt modellhaft erst bei Erreichen von `T_kalt` ein, nicht graduell
vorher. Vier Tests decken die geforderten Invarianten: sensibler Modus bitgleich,
`Q_rec`(kondensierend) ≥ `Q_rec`(sensibel) bei gleichem `T_kalt` ≤ Taupunkt, stetiger
Rückfall auf reine sensible Kühlung oberhalb des Taupunkts (kein Sprung), Massenbilanz
des Kondensats (`source_condensate_kg_per_s`, neues Ergebnisfeld). Commit `b830b60`.

**b) Unsicherheit des Feuchtekanals in kW (MA2) -- Vorzeichenwechsel am realen
Nominalpunkt bestätigt.** Fehlerquelle je Kanal:
`rmse_run_median_physical/<Kanal>/h36`, LSTM, `n_train=64`, `test_set=constant_holdout`
(Konstantlauf-Holdout -- der betriebsrelevante, siehe Befund 55), Median über 5 Seeds,
aus `arch_learning_curve_constant_trained.jsonl` (Befund 55, Commit `1b85ce4`):
Betttemperatur 0,539 K, Produktfeuchte 0,001488 kg/kg. Störung einzeln und in allen vier
Vorzeichenkombinationen am Nominalpunkt des Simulators (Beharrungszustand,
`greybox_gab_placeholder.yaml`, `T_bed=62,55 °C`, `Y_out=0,01703 kg/kg`,
`m_L=0,0294 kg/s` Labor), `Y_out`/`T_out` über `reconstruct_exhaust_state` neu gerechnet,
`Q_rec` in beiden Modi, bei Labor- UND Verbund-Massenstrom (Skalenvorbehalt, siehe unten):

| Störung | ΔQ_rec sensibel | ΔQ_rec kondensierend |
|---|---|---|
| Betttemperatur +0,539 K | +0,63 % (+1,96 kW) | +2,24 % (+11,65 kW) |
| Produktfeuchte +0,001488 kg/kg | **-1,60 % (-4,95 kW)** | **+2,73 % (+14,20 kW)** |
| beide, je ungünstigste Kombination | +2,30 % (+7,11 kW) | +5,05 % (+26,30 kW) |

Die ungünstigste Vorzeichenkombination ist NICHT für beide Modi dieselbe: sensibel steigt
`Q_rec` sowohl bei `T+` (+0,63 %) als auch bei `Feuchte-` (+1,65 %, gehobener Taupunkt
umgekehrt) -- `(T+, Feuchte-)` verstärkt sich deshalb zum größten sensiblen Ausschlag
(+2,30 %). Kondensierend steigt `Q_rec` bei `T+` UND bei `Feuchte+` (mehr Kondensat) --
`(T+, Feuchte+)` ist dort der größte Ausschlag (+5,05 %), mit `(T-, Feuchte-)` fast
gleich groß in die Gegenrichtung (-4,82 %). Ein Hinweis, dass "ungünstig" vom
Rückgewinnungsmodus abhängt, nicht vom Kanal allein. (kW-Spalten bei
`DISTRICT_MASS_FLOW_KG_PER_S=8 kg/s`, Verbund-Skalierungsrechnung, keine
Anlagenaussage -- siehe Skalenvorbehalt.) **Der Vorzeichenwechsel bestätigt sich exakt wie
in der Anlassbeschreibung vermutet**, jetzt mit verifizierten Zahlen statt Überschlag:
mehr Feuchte SENKT `Q_rec` sensibel (der gehobene Taupunkt engt die Kühlgrenze ein) und
HEBT es kondensierend (mehr latente Wärme). Kernfrage beantwortet: der Feuchtekanal
(2,73 %, 14,2 kW) trägt am Verbund-Massenstrom **mehr** Unsicherheit in die
kondensierende Rückgewinnung als der Temperaturkanal (2,24 %, 11,65 kW) -- vergleichbare
Größenordnung, Feuchte leicht dominant. Baseline `Q_rec` am Nominalpunkt (Verbundskala):
sensibel 308,9 kW, kondensierend 520,6 kW (+68,5 %). Skript
`scripts/study_humidity_channel_uncertainty.py`, Ergebnis
`artifacts/runs/study_humidity_channel_uncertainty.json`, Commits `cdfa010`/`30d59b0`.

**c) Netto-Zielfunktion (MA3/MA4) -- Befund 52 mit Rückgewinnungsgutschrift, zwei
Nutzerentscheidungen vorab eingeholt.**

Zwei Festlegungen, explizit als solche gekennzeichnet, keine Messungen: (1) die Gutschrift
ist das Brennstoffäquivalent `Q_rec / eta_H` (`eta_H = HeatNetworkConfig.
heater_efficiency`), nicht `Q_rec` thermisch -- sie ersetzt Erdgas im Lufterhitzer, der
selbst nur mit `eta_H < 1` arbeitet. (2) zwei Zielfunktionsvarianten NEBENEINANDER, keine
ersetzt die andere: `"kw_sum"` (ungewichtete kW-Differenz, Status quo fortgeführt) und
`"co2_weighted"` (Bruttoleistung × 0,366 kg CO2e/kWh Strom, Gutschrift × 0,201 kg CO2e/kWh
Gas -- Gesamtvorhabenbeschreibung/Umweltbundesamt, Strom 1,82-mal schlechter gewichtet als
Gas). Beide Varianten waren nötig, weil sonst nicht trennbar ist, was aus der
Netto-Formulierung selbst folgt und was aus der Gewichtung (Nutzerbegründung wörtlich).
Implementiert in `evaluation/net_energy.py`, Commit `9bd0e37`.

`scripts/study_operating_point_regret_net.py` wiederholt BB1-BB3 aus Befund 52
unverändert (Gitter, `differential_evolution`+Nelder-Mead, Nebenbedingungen,
Regret-Berechnung -- über einen neuen, verhaltenserhaltenden Override-Hook in
`study_operating_point_regret.py`, Commit `0307f91`, keine Zeile der Mechanik selbst
geändert) für fünf Varianten auf dem 3×3-Gitter aus Befund 54 (DA1), Verschaltung: EINE
Linie, die Rückgewinnung wärmt ihre eigene Zuluft vor (im Einzelanlagenfall die einzige
sinnvolle Konfiguration), `Q_rec` am Labor-Massenstrom (dieser Prozess recycelt nur seine
EIGENE, kleine Abwärme, kein Verbund):

| Variante | Regret Median | Fester Punkt `u_gas` [m/s] |
|---|---|---|
| V1 Brutto (Referenz) | **15,222 %** | 0,800 (untere Schranke) |
| V2 Netto sensibel, kW-Summe | 14,901 % | **2,500 (obere Schranke)** |
| V3 Netto sensibel, CO2-gewichtet | 8,576 % | 0,800 (untere Schranke) |
| V4 Netto kondensierend, CO2-gewichtet | 18,802 % (P95 94,3 %, Max 102,4 %) | 0,800 (untere Schranke) |
| V5 Netto kondensierend, kW-Summe | 128,3 % (P95 387,0 %, Max 520,6 %) | **2,500 (obere Schranke)** |

V1 trifft den Befund-54-Anker (15,22 %) auf 0,002 Prozentpunkte -- die wiederverwendete
Mechanik ist unverändert, V2-V5 sind interpretierbar. **Kernfrage beantwortet, Erwartung
NICHT bestätigt:** die Gasgeschwindigkeit verlässt ihre untere Schranke -- aber NUR unter
kW-summierter Gewichtung (V2, V5, beide an der OBEREN Schranke 2,5 m/s), nicht unter
CO2-Gewichtung (V3, V4 bleiben bei 0,8 m/s wie der Bruttofall). Die vorab formulierte
Erwartung ("mit Kondensation deutliche Verschiebung, ohne Kondensation moderate") trifft
NICHT zu -- der Rückgewinnungsmodus ist für die Verschiebung fast irrelevant (V2 und V5
verhalten sich nahezu gleich, V3 und V4 ebenso), die **MA3-Gewichtung dominiert den
Effekt vollständig**, genau die im Auftrag als wahrscheinlichste Erklärung vorgesehene
Alternative. V5s relative Regret-Werte sind zusätzlich numerisch instabil (Median 128 %,
Max 521 %) -- Artefakt eines nahe null liegenden Netto-Nenners an einzelnen
Gitterpunkten (`compute_regret`s Epsilon-Fallback greift), kein Fehler in der Mechanik,
aber ein Hinweis, dass relative Regret-Prozentsätze bei stark rückgewinnungsdominierten
Zielfunktionen mit Vorsicht zu lesen sind (absolute Werte bleiben aussagekräftig). Laufzeit
1,18 h (Schätzung 1,8 h nach Befund-54-Referenz). Commits `26db05c` (Skript, geprüft),
`83851e0` (Ergebnis).

**d) Sechs Verschaltungsfälle (MA5, `Model_v3.xlsx`) -- Einheitenfehler gefunden, Angebot
deckt jetzt 93 statt 49 % des Bedarfs.** Datengrundlage vom Nutzer aus
`Model_v3.xlsx` rekonstruiert (Rohdatei nicht im Repo): zwei Abluftquellen (Q1
Sprühturm, 20 kg/s, 125,233 °C, Y=0,06482, kondensiert auf 10 °C; Q2 Wirbelschicht,
8 kg/s, 81,416 °C, Y=0,01998, kondensiert auf 15 °C -- ZWEI unterschiedliche `T_kalt`,
deshalb hier nicht `heatnetwork.balance.evaluate_cascade` direkt, sondern derselbe
EA1/EA2-Algorithmus mit thermo.py-Primitiven auf ein `T_kalt` je Quelle verallgemeinert,
`scripts/probe_six_case_reconciliation.py`), zwei Zuluftsenken (S1 20 kg/s 15→300 °C, S2
8 kg/s 15→120 °C), `H_u=45 MJ/kg`, `eta_H=0,80`.

Bei der Reproduktion der Blattwerte (Q1: 2671,8 kW, Q2: 561,6 kW) fiel eine Abweichung von
über 100 % zur MA1-Formel auf -- zu groß für Buck-gegen-Antoine (< 0,1 K, erwartet) oder
den bereits bekannten Kondensat-Vorzeichenfehler (1-2 %, erwartet). Nutzerrückfrage klärte
die tatsächliche Ursache: `Balances!E20`/`E27` verrechnet die Verdampfungsenthalpie
(`Inputs!I12 = 2501`, im Blatt als kJ/kg beschriftet) OHNE Umrechnung mit den J-basierten
spezifischen Wärmekapazitäten -- effektiv wie 2501 J/kg statt 2,501·10⁶ J/kg, Faktor 1000
zu klein, die latente Wärme fehlt im Blatt praktisch vollständig. Isoliert auf
`E20`/`E27`: die Trocknerbilanz selbst (`Balances!F5/F12`, `Inputs!I9`) verwendet den
korrespondierenden Heizwert korrekt und ist nicht betroffen. Korrigierte Quellenleistung
(MA1-Formel, hier verifiziert -- Q1: 5429,8 kW, davon 1809,1 kW oberhalb, 3620,7 kW
unterhalb des Taupunkts von 44,94 °C; Q2: 738,3 kW, Restabweichung zur vom Nutzer
unabhängig korrigierten Blattformel 0,2-0,4 % (Buck gegen Antoine)):

**Das ändert die Aussage des Fallvergleichs qualitativ.** Angebot gesamt korrigiert
6168,1 kW gegen Bedarf 6640,1 kW = **92,9 % Deckung** (statt 48,7 % mit dem fehlerhaften
Blattwert 3233,4 kW). Bei knapper Deckung ist das Angebot der Engpass und jede
Verschaltung, die es vollständig nutzt, landet beim selben Brennstoffverbrauch -- bei
fast gedecktem Bedarf wird die ZUORDNUNG entscheidend. Ohne Grädigkeitsprüfung (reine
Energiebilanz wie im Blatt, jetzt mit korrigiertem Angebot): Fall 1 = 386,22 kg/h, Fall 2
= 504,89 kg/h, Fall 3 = 440,17 kg/h, Fall 4 = 578,72 kg/h, Fall 5 = 440,17 kg/h, Fall 6 =
578,72 kg/h. **Fall 3 und Fall 5 unterscheiden sich nur in der Kaskadenreihenfolge
(kalte/heiße Quelle zuerst) und liefern denselben Brennstoffverbrauch -- der Beleg, dass
das Blatt Temperaturniveaus nicht prüft.** Bei genauerem Hinsehen ist das kein Zufall,
sondern selbst aufschlussreich: in Fall 5 liefert Q2 in der zweiten Stufe exakt 0 kW (die
Senke ist durch Q1 in Stufe 1 bereits auf 125,23 °C erhitzt, über Q2s eigener
Eintrittstemperatur von 81,42 °C -- eine Verletzung von 43,8 K, physikalisch
unrealisierbar, ein Wärmeübertrager kann die Senke nicht über die Eintrittstemperatur der
Quelle hinaus erwärmen), während Q1 allein zufällig exakt denselben Gesamtbetrag liefert
wie Q2+Q1 zusammen in der richtigen Reihenfolge (Fall 3) -- weil beide Kaskaden am selben
Deckel (Q1s eigene Eintrittstemperatur) enden.

Mit Grädigkeit `dT_min=20 K` (Sensitivität 10 K): Fall 1 = 443,08 kg/h (**bester machbarer
Fall**, 414,65 kg/h bei 10 K), Fall 2 = 516,88 kg/h, Fall 3 = 480,78 kg/h, Fall 4 =
590,72 kg/h, Fall 5 = 480,78 kg/h, Fall 6 = 590,72 kg/h. **Erwartung NICHT bestätigt:**
die (auf dem fehlerhaften Blatt basierende) Ausgangserwartung nannte Fall 2 (522,62 kg/h,
53 % über dem damaligen 340,73-kg/h-"Optimum") als besten machbaren Fall -- mit
korrigierten Quellenleistungen ist stattdessen **Fall 1** (getrennte Verschaltung, Q1→S1,
Q2→S2, keine Kaskadierung zwischen den Linien) der beste, bei einem 340,73-kg/h-Wert, der
selbst hinfällig ist (er beruhte auf den fehlerhaften Quellenleistungen). Commits
`82926d6` (Skript, geprüft), `ad2313b` (Ergebnis).

**e) Kernaussage: der Feuchtekanal steigt von zweitrangig auf mitbestimmend -- und ist
genau der Kanal, der an der Anlage nicht messbar, am Optimum bindend und aus
Betriebsdaten nicht lernbar ist.** Vor diesem Befund war `Y_out` für die Bruttozielfunktion
(Befund 52) irrelevant (sie kennt keine Rückgewinnung) und für die sensible
Rückgewinnung (Befund 50) ein Störfaktor zweiter Ordnung (hebt den Taupunkt, reduziert
die Kühlgrenze). Mit Kondensation (a) ist Feuchte eine Ressource, die (b) messbar mehr
Unsicherheit als die Betttemperatur in `Q_rec` trägt, und (d) darüber entscheidet, ob ein
Angebot 49 % oder 93 % des Bedarfs deckt -- eine qualitative, nicht nur graduelle
Verschiebung. Drei bereits bestehende Befunde verschärfen das: `product_moisture_kg_per_kg`
ist an der Anlage nicht gemessen (Befund 56, NA-Annahmenliste -- Softsensor, keine direkte
Messung), bindet als harte Nebenbedingung am Optimum von Befund 52 (kalte/feuchte
Störgrößenregime, Befund 52c), und ist aus rein historischen Betriebsdaten (konstant
trainierter Arm) im getesteten Budgetbereich nicht auf das Niveau des angeregten Arms zu
bringen (Befund 55, Faktor 2,2-3,1 NRMSE-Lücke, kein Aufholen bei `n_train=64`). Der
Kanal, der jetzt am meisten zählt, ist derselbe, der am wenigsten robust vorhergesagt
wird.

**f) Vorbehalte.**

- **`T_kalt` ist ein neuer, unbelegter Platzhalter** (15 °C, Default) -- wie `dT_min`
  keine Verbundpartner-Angabe, sondern ein Ankerpunkt aus dem eigenen Überschlag, und
  **optimistisch**: die reale Kaltseite liegt eher bei ~29 °C (Nachtrag NB1, (h)), bei dem
  der MA2-Vorzeichenwechsel für die Wirbelschichtlinie nicht mehr gilt. MA5 (d) zeigt
  zudem, dass das reale Partnermodell UNTERSCHIEDLICHE `T_kalt` je Quelle fährt (10 °C/
  15 °C) -- `HeatNetworkConfig.cold_temperature_celsius` ist netzweit, nicht je Quelle;
  eine Verallgemeinerung ist erst nötig, wenn mehr als eine Studie sie braucht (siehe
  `docs/06_waermenetz.md`).
- **Die MA3-Gewichtung ist eine Festlegung, keine Messung**, und bestimmt in (c) allein
  das Vorzeichen der Verschiebung -- eine andere (z. B. kostenbasierte) Gewichtung könnte
  das Ergebnis umkehren. Explizit dafür wurden beide Varianten (kW-Summe UND
  CO2-gewichtet) gerechnet, nicht nur eine. Nachtrag NB2 (i): unter Kosten- UND
  CO2-Bewertung lohnt sich der Tausch nicht -- nur die (sachlich unbegründete) kW-Summe
  zeigt die Verschiebung; die Wahl selbst ist eine offene Verbundentscheidung.
- **Ein Prozess, kein Verbund (c), zwei Größenordnungen Skalenunterschied (b).** MA4
  rechnet `Q_rec` am Laborsimulator-Massenstrom (~0,03 kg/s) -- die Rückgewinnungsgutschrift
  ist absolut klein gegen den Bruttoverbrauch (Median `Q_rec` am Optimum 0,08-1,23 kW je
  nach Variante), ob sich der Betriebspunkt spürbar verschiebt, ist damit auch eine Frage
  dieses Maßstabs, nicht nur der Gewichtung. MA2/MA5 rechnen dagegen am Verbund-Massenstrom
  (8 kg/s) -- jede kW-Zahl dort ist eine Skalierungsrechnung, keine Anlagenaussage (wie
  durchgehend in `evaluation/recoverable_power.py` und `docs/06_waermenetz.md`
  gekennzeichnet).
- **MA5s Datengrundlage liegt außerhalb des Repos** (`Model_v3.xlsx`, vom Nutzer
  übermittelt, nicht committet) -- die sechs Fälle, Quellen-/Senkenwerte und der
  identifizierte Einheitenfehler sind damit nur über diesen Befund und
  `scripts/probe_six_case_reconciliation.py` nachvollziehbar, nicht über eine eigene,
  parsbare Eingabedatei im Repo (ausdrücklich nicht Teil des Auftrags, siehe
  `docs/06_waermenetz.md`).
- **Vorbehalt in Befund 52 aktualisiert:** die dortige Bruttoformulierung war eine
  bewusste Einschränkung ("Ein Prozess, kein Verbund", Befund 52d) -- mit (c) liegt jetzt
  die Nettofassung vor, siehe `docs/06_waermenetz.md`, Abschnitt "Bezug zu Befund 52 und
  57".

**g) Reproduzierbarkeit.** Commits in Reihenfolge: `b830b60` (MA1), `cdfa010`/`30d59b0`
(MA2 Skript/Ergebnis), `9bd0e37` (MA3), `0307f91` (MA4-Vorbereitung, Override-Hook),
`26db05c`/`83851e0` (MA4 Skript/Ergebnis), `82926d6`/`ad2313b` (MA5 Skript/Ergebnis).
Alle Skripte vor Ausführung committet (Provenienzregel), Ruff und die vollständige
Testsuite vor jedem Commit grün.

**h) Nachtrag NB1: `T_kalt=15 °C` ist ein OPTIMISTISCHER Platzhalter -- die reale
Kaltseite liegt bei rund 29 °C, und der MA2-Vorzeichenwechsel gilt nur, solange die
Linie tatsächlich kondensiert.** 15 °C entspricht der Umgebungstemperatur des Modells --
die Kaltseite hätte damit eine Treibtemperaturdifferenz von null, physikalisch nur mit
einer Kältemaschine erreichbar. Das Leitsystem des Anwendungspartners nennt für die
Kompressorstation 29,1 °C Kühlwassertemperatur -- ein realer, gemessener Anhaltspunkt
für die tatsächliche Kaltseite, kein Modellwert.

`Q_rec` je Quelle über `T_kalt ∈ {15; 20; 25; 29,1; 35} °C` (MA1-Formel, mit den MA5-Werten
für Q1/Sprühturm und Q2/Wirbelschicht, Taupunkte 44,94 °C bzw. 24,92 °C nach der
Buck-Korrelation des Projekts -- **geklärt (Befund 60, `lp-benchmark`-Zweig):** die vom
Nutzer zuvor genannten Werte 47,9 °C (Q1) und 27,93 °C (Q2, an anderer Stelle verwendet)
waren keine Taupunkte, sondern Taupunkt PLUS die 3-K-Abkühlgrenze aus Model_v2,
`Balances!F19` (Magnus-Formel dort endet auf "+ 3") -- eine Zellen-Verwechslung, keine
Formel- oder Modellabweichung. Die Buck-Werte 44,94 °C/24,92 °C sind die korrekten
Taupunkte; die frühere Gegenüberstellung mit der Buck-gegen-Antoine-Abweichung aus (d)
(0,2-0,4 % bei `Q_rec`) war damit kein Vergleich zweier Korrelationen, sondern zweier
unterschiedlicher physikalischer Größen (Taupunkt vs. Taupunkt+Approach):

| `T_kalt` [°C] | Q1 (Sprühturm) | kondensiert? | Q2 (Wirbelschicht) | kondensiert? | Summe |
|---|---|---|---|---|---|
| 15,0 | 5154,9 kW | ja | 738,3 kW | ja | 5893,2 kW |
| 20,0 | 4830,9 kW | ja | 616,3 kW | ja | 5447,2 kW |
| 25,0 | 4443,4 kW | ja | 470,8 kW | **nein** | 4914,2 kW |
| 29,1 | 4065,6 kW | ja | 436,6 kW | nein | 4502,2 kW |
| 35,0 | 3399,3 kW | ja | 387,4 kW | nein | 3786,7 kW |

Deckt sich mit der zu verifizierenden Nutzerrechnung (5897 kW bei 15 °C, 4506 kW bei
29,1 °C) auf unter 0,1 %. **Verifiziert: oberhalb 25 °C kondensiert die
Wirbelschichtlinie (Q2) nicht mehr** (Taupunkt 24,92 °C, `T_kalt=25` liegt knapp darüber)
-- die Sprühturmlinie (Q1, Taupunkt 44,94 °C) kondensiert im gesamten getesteten Bereich
weiter.

**Konsequenz für (b):** der dort verifizierte Vorzeichenwechsel der
Feuchteempfindlichkeit gilt nur, SOLANGE die Linie kondensiert. Bei der realistischen
Kaltseite (29,1 °C statt 15 °C) gilt für die Wirbelschichtlinie -- genau die Linie mit dem
im gesamten Befund als Rechenbeispiel verwendeten Betriebspunkt (8 kg/s, `Y=0,01998`,
`T_out=81,4 °C`, siehe (b)) -- wieder die SENSIBLE Betrachtung: Feuchte SENKT `Q_rec`,
hebt es nicht. Der Vorzeichenwechsel ist damit an eine Modellannahme gebunden, die selbst
nicht gesichert ist, nicht an eine universelle Eigenschaft des kondensierenden Modus.

**Zusätzliche, bereits in `docs/06_waermenetz.md` vermerkte Abweichung:** das Blatt
(`Model_v3.xlsx`) verwendet ZWEI verschiedene `T_kalt`-Werte (10 °C für WÜT1/Sprühturm,
15 °C für WÜT2/Wirbelschicht), `HeatNetworkConfig.cold_temperature_celsius` dagegen einen
einzigen, netzweiten Wert -- eine Modellvereinfachung gegenüber dem realen Partnermodell,
nicht nur ein Platzhalterproblem.

**i) Nachtrag NB2: die Zielfunktionswahl aus (c) ist eine Verbundentscheidung, keine
Implementierungsdetailfrage -- unter den inhaltlich begründeten Bewertungen lohnt der
Tausch nicht.** Von den in (c) gerechneten Bewertungsmaßstäben ist die kW-Summe (V2, V5)
die einzige, die elektrische und thermische Leistung als GLEICHWERTIG behandelt -- ohne
inhaltliche Rechtfertigung, ein reines Artefakt derselben Einheit (kW). Industriestrom
kostet je kWh grob das Drei- bis Vierfache von Erdgas (Hausnummer, **nicht geprüft,
Vorbehalt**) und emittiert nach den bereits in (c) verwendeten Emissionsfaktoren
(`0,366` gegen `0,201` kg CO2e/kWh) das 1,8-fache CO2. **Unter BEIDEN inhaltlich
begründeten Bewertungen (Kosten, CO2) lohnt sich der Tausch "mehr Strom für mehr
Rückgewinnung" also nicht** -- nur die (interpretationsfreie, aber sachlich unbegründete)
kW-Summe treibt die Gasgeschwindigkeit in (c) an ihre obere Schranke.

Die Wahl der Zielgröße ist damit keine Implementierungsdetailfrage, sondern eine
**Verbundentscheidung mit unmittelbarer Konsequenz für den späteren RL-Aktionsraum**: eine
Policy, die auf der kW-Summe trainiert würde, strebte einen Betriebspunkt an (hohe
Gasgeschwindigkeit), der unter jeder inhaltlich plausiblen Bewertung (Kosten, CO2)
schlechter ist als der Bruttofall. **Als offene, an den Verbundpartner zu klärende
Entscheidung festgehalten, hier nicht getroffen.**

### 58. Lernkurve um Transformer/xLSTM/Mamba erweitert -- protokollgleich zu Befund 51, GRU/LSTM bleiben bei kleinem Budget vorn, Transformer-Epochenobergrenze bindet bei `n_train=64` an zwei von drei Köpfen durchgehend

**Anlass.** Die Gesamtvorhabenbeschreibung nennt Transformer-Modelle ausdrücklich neben
rekurrenten Netzen. Der Lernkurven-Datensatz aus Befund 51 deckt nur `persistence,
static, arx, gru, lstm` ab -- xLSTM und Mamba sind im Repo implementiert
(`models/composite.py`), ihre Sweep-Worker waren aber am alten (mittlerweile durch die
Gasgeschwindigkeits-Projektion überholten) Fingerprint blockiert. Drei der im Antrag
genannten Architekturen fehlten im Hauptergebnis von AP1 -- diese Lücke wird hier
geschlossen.

**OA1 -- bestehende Skripte ungeeignet, Worker erweitert statt drei Skripte repariert.**
Geprüft: `run_transformer_reduced_comparison.py`, `run_xlstm_reduced_comparison.py`,
`run_mamba_reduced_comparison.py` sowie die `_constant_holdout_worker`/`_sweep`-Paare für
xLSTM/Mamba. Keins der sechs Skripte erzeugt eine Lernkurve mit `n_budget`-Staffelung --
jeweils nur ein einzelner Budgetpunkt. Keins überschreibt
`NumericsConfig.enforce_gas_velocity_projection` (Default aus), der resultierende
Fingerprint ist deshalb nirgends `2d2679eeed716577`. `run_xlstm_reduced_comparison.py`
und `run_mamba_reduced_comparison.py` sind zusätzlich aktuell **nicht lauffähig**: ihr
hartkodiertes `EXPECTED["data/target_names"]` verlangt einen Drei-Kanal-Eintrag mit
`d50_um`, der `DataSpec`-Default hat aber seit dessen Entfernung nur noch zwei Kanäle --
der Vergleich schlägt bei jedem Aufruf fehl. `data/train_constant_run_fraction` fehlt in
allen sechs Skripten. Entscheidung (mit Rückfrage getroffen, nicht vorausgesetzt):
`run_arch_learning_curve_worker.py` um die drei Kerne erweitert
(`ARCHITECTURE_LABELS`/`_architecture_factory`), statt die sechs Skripte einzeln
nachzuziehen -- ein Fingerprint-Check, ein Protokoll, kein Risiko stiller Abweichungen
zwischen parallelen Kopien.

**Konfiguration -- bitgenau Befund 51.** `process/fingerprint=2d2679eeed716577`, drei
Zielkanäle (`energy_target`/`product_moisture_kg_per_kg`/`bed_temperature_celsius`),
`n_train ∈ {8,16,32,64}`, 5 Seeds, `history_length=32`, `horizon=36`,
`MKL_NUM_THREADS`/`OMP_NUM_THREADS=1`. Bitgleichheit mit Befund 51 nicht nur behauptet,
sondern geprüft: beide Datensätze tragen für jeden `master_seed` **identische**
`split/digest`-Werte (`{3a82e4a1, 751a5bc0, b4d224da, 5b9628a1, a054672b}`, je Seed
derselbe Wert in beiden Dateien) und identische `seed/data`/`seed/noise`/`seed/model`
(z. B. Seed 1: `57731891`/`1868603354`/`2509151119` in beiden Dateien) --
**protokollgleich, damit gemeinsam auswertbar**, wie unten in OA5 genutzt.
Kapazitäts-Defaults der drei neuen Kerne unverändert aus `composite.py`: Transformer/Mamba
`hidden_size=16` (bewusst klein gegen den GRU/LSTM/xLSTM-Standard, siehe dortiger
Docstring zum historischen >12h-Fall bei kapazitäts-angepasster Größe), xLSTM
`hidden_size=64` wie GRU/LSTM.

OA3-Pilot (1 Seed, `n_train=8`, alle drei Architekturen) vorab bestanden: `git/commit` ein
Hash, `git/dirty=False`, Fingerprint/Zielkanäle/`train_constant_run_fraction=0.10`/
`history_length=32`/`horizon=36` wie erwartet, `rmse_run_median_physical` belegt. Ein
erster Pilotversuch war durch einen zwischenzeitlich dirty Arbeitsbaum (eigene
unfertige Skriptänderungen) kontaminiert -- verworfen, auf sauberem Baum wiederholt,
nicht durch Nachbearbeitung "repariert" (dieselbe Lehre wie Befund 51s Zwischenfall).

**GRU/LSTM/Baselines bewusst NICHT erneut gerechnet.** Erste Ausbaustufe des Vollaufs
rechnete versehentlich alle acht Architekturen je Seed (`ARCHITECTURE_LABELS` enthielt
GRU/LSTM bereits) -- unnötig, weil deren Records unter identischem Protokoll bereits in
`arch_learning_curve_projected.jsonl` liegen (siehe Bitgleichheits-Nachweis oben). Der
Worker wurde um `--architectures`/`--skip-baselines` ergänzt (wählt eine Teilmenge der
Architekturen/lässt Baselines weg), der Sweep-Orchestrator reicht das durch und bezieht
die Auswahl in den Fortsetzungs-Auftragsschlüssel ein (sonst hielte `--resume` einen
absichtlich schlanken Lauf fälschlich für unvollständig). **Zwischenfall:** der Commit
mit dieser Nachrüstung fiel in einen bereits laufenden Sweep (vier Seed-Jobs liefen
bereits) -- ab dem Commit hätten neue Records einen anderen `git/commit` getragen als die
bereits geschriebenen desselben Seeds. Alle vier Jobs abgebrochen (keine verwaisten
Prozesse, geprüft), der kontaminierte Zwischenstand gelöscht, der Sweep auf dem jetzt
stabilen Commit neu gestartet -- nur `transformer xlstm mamba`, `--skip-baselines`.
Dieselbe Lehre wie in Befund 51 und im Pilot-Zwischenfall oben: kein Skript ändert
getrackte Dateien, während ein provenienzsensitiver Lauf aktiv ist.

**Lücke in der Provenienz-Wache, hier nur vermerkt, nicht behoben (SA4).** Der Commit
landete mitten in einem LAUFENDEN Sweep, nicht bei einer Fortsetzung -- die in OA2b
gebaute Wache (`run_arch_learning_curve_sweep.py`, Abschnitt "PROVENIENZ-WACHE") prüft den
`git/commit` ausschließlich beim `--resume`-Start gegen die bereits vorliegenden
Records. Sie hätte diesen konkreten Zwischenfall nicht automatisch gefangen: der
laufende Sweep selbst führt während seiner Ausführung keine Commit-Prüfung durch, nur
beim (Wieder-)Start. Gefunden wurde der Zwischenfall hier durch aufmerksames Beobachten,
nicht durch die Wache. Siehe auch die Ergänzung im Docstring von
`run_arch_learning_curve_sweep.py`.

**Vollauf.** `artifacts/runs/arch_learning_curve_projected_extended.jsonl`, 120 Records
(5 Seeds × 4 Budgetpunkte × 3 Architekturen × 2 Testmengen), **ein einziger
`git/commit`** (`256d644`), **`git/dirty=False`** in jedem Record -- geprüft, nicht nur
erwartet. Der Sweep-Prozess überlebte einen Sitzungsneustart und eine
Rechner-Standby-Phase unbeschadet (kein Job-Objekt, das Kindprozesse an die Sitzung
bindet); Fortsetzbarkeit (`--resume`, Auftragsschlüssel
`master_seed/n_train/model_label/test_set`, Provenienz-Wache gegen Commit-Drift und
dirty Arbeitsbaum) war für diesen Lauf am Ende nicht nötig, stand aber bereit.

**Kosten.** Einzelfit-Kostenprobe vorab (`scripts/probe_new_arch_cost.py`, 1 Seed,
`n_train=8`, 3 Zielkanäle, MKL/OMP=1): Transformer 2657 s, xLSTM 413 s, Mamba 517 s.
Hochrechnung auf den vollen Sweep (Untergrenze, `n_train=8` flach auf alle vier
Budgetpunkte angewendet) ergab ~4h zusätzliche Wanduhrzeit -- an der selbst gesetzten
4h-Schwelle, an die bei Überschreitung eine Rückfrage gebunden war. Melden und Fragen vor
dem Rechnen (Nutzerentscheidung: trotzdem vollständig rechnen, mit Fortsetzbarkeit
zuerst). Tatsächliche Kosten **wuchsen mit `n_train` deutlich stärker als die
Untergrenze nahelegte** -- Transformer-Trainingszeit (Median über 5 Seeds, Testmenge
"excited"):

| `n_train` | Transformer | xLSTM | Mamba |
|---|---|---|---|
| 8 | 3267 s | 418 s | 292 s |
| 16 | 5610 s | 1095 s | 761 s |
| 32 | 9304 s | 1579 s | 1078 s |
| 64 | 14698 s (~4,1 h) | 3477 s | 1992 s |

Transformer wächst um Faktor ~4,5 von `n_train=8` zu `n_train=64`, xLSTM um Faktor ~8,3,
Mamba um Faktor ~6,8 -- die n_train=8-Kostenprobe war für alle drei eine deutliche
Unterschätzung der großen Budgetpunkte, für Transformer am stärksten. Gesamtlaufzeit des
finalen (schlanken) Sweeps, erster bis letzter Record-Zeitstempel: **20,74 h**.

**OA5 -- Achtergleich (Befund 51 + hier), gemeinsam ausgewertet
(`scripts/analyze_arch_learning_curve_extended.py`).**

*a) Welche Testmenge.* Der Konstantlauf-Holdout ist die betriebsrelevante Größe, die
angeregte Menge misst Identifikationsgüte -- und verzerrt Baseline-Vergleiche
systematisch: Persistenz ist auf der angeregten Menge strukturell schwach (keine
Dynamik-Information), auf dem Konstantlauf dagegen trivial stark (die Holdout-Läufe SIND
näherungsweise konstant). Konkret in diesem Datensatz: `bed_temperature_celsius`, h=36,
`rmse_run_median_physical` von Persistenz: **6,408** (angeregt) gegen **0,630**
(Konstantlauf) -- Faktor 10,2 an derselben Größe, derselben Architektur, demselben
Datensatz. `energy_target`, h=1: 0,1014 gegen 0,0162 -- Faktor 6,3. Genau das Muster, das
in Befund 51 (K5) bereits einmal zu einer nicht vergleichbaren Zahl geführt hat (dort
Faktor 4,4-23 ohne Erklärung) -- hier vorab benannt, nicht erst nach einer verdächtigen
Kennzahl.

*b) Monotonie.* Median-`rmse_run_median_physical` über `n_train`, je Architektur/Kanal/
Testmenge/Horizont (24 Zellen je Architektur, Persistenz konstruktionsbedingt
ausgenommen -- kein Training auf `n_train`, Gleichstände sind kein Befund): **GRU 2/12,
LSTM 1/12, Transformer 1/12** nicht-monotone Zellen -- weitgehend glatt fallende
Lernkurven. **xLSTM 4/12, Mamba 4/12, static 5/12, ARX 7/12** deutlich häufiger
nicht-monoton. Auffälligster Einzelfall: xLSTM auf `energy_target` (angeregt, h=1)
steigt von `n=8` (0,098) auf `n=16` (0,112) und bleibt bei `n=32` exakt gleich (0,112) --
kein Aufholen mit mehr Daten auf diesem Kanal, deckt sich mit der bereits in
`run_xlstm_constant_holdout_worker.py` dokumentierten Beobachtung einer für xLSTM
divergierenden Optimierung auf `energy_target`. Nicht-Monotonie ist damit kein
architekturweiter Fehler, sondern konzentriert sich auf xLSTM/Mamba und auf die
statistischen Baselines (deren Kapazität bei mehr Daten nicht wächst, Rauschen in der
Parameterschätzung bleibt).

*Vorab festgelegte Erwartungen -- geprüft, nicht vorausgesetzt:*

- **`product_moisture_kg_per_kg` (höchster dynamischer Anteil): Architekturen liegen am
  weitesten auseinander -- bestätigt.** Angeregt, h=36, `n_train=64`: LSTM 0,00051, GRU
  0,00061, Transformer 0,00101, xLSTM 0,00123, Mamba 0,00505 -- Faktor 9,9 zwischen bestem
  und schlechtestem trainierten Modell, die größte Streuung aller drei Zielkanäle.
- **`energy_target` (98,7 % statisch erklärbar, Befund 2/39): alle nahe der statischen
  Regression -- NICHT bestätigt.** Angeregt, h=36, `n_train=64`: `static` liegt bei
  0,1115, GRU/LSTM/Transformer bei 0,0145/0,0143/0,0173 -- **Faktor 6,4-7,8 besser als
  die statische Regression**, nicht nahe daran. xLSTM (0,428 bei `n=64`) und Mamba (0,053)
  liegen dagegen deutlich schlechter als GRU/LSTM. xLSTM ist bei kleinem Budget sogar
  schlechter als die naive Persistenz (`n=8`: 0,5315 gegen Persistenz 0,4857 -- konstant
  über `n_train`, kein Training; `n=16`: 0,5662 gegen 0,4857) und wird erst ab `n=32`
  knapp besser (0,4601/0,4277 gegen 0,4857) -- auf diesem Kanal ist xLSTM über weite Teile
  des Budgetbereichs nicht besser als gar kein Training. Die Erwartung war zu grob:
  "statisch erklärbar" sagt etwas über
  den durch ein perfektes statisches Modell erklärbaren Varianzanteil, nicht darüber, wie
  nah eine LINEARE Regression (`StaticRegressionModel`) an dieses Optimum herankommt --
  hier klafft eine Lücke, die rekurrente/Transformer-Architekturen schließen, xLSTM/Mamba
  aber nicht.
- **Kleines Budget: größere Modelle schlechter als GRU/LSTM -- bestätigt.** `n_train=8`,
  `product_moisture_kg_per_kg`, h=36, angeregt: LSTM 0,00203, GRU 0,00264, Transformer
  0,00412, xLSTM 0,01624, Mamba 0,00898 -- GRU/LSTM klar vorn, xLSTM am weitesten dahinter.
- **Ob die neuen Architekturen mit wachsendem Budget aufholen oder überholen -- deutliches
  Aufholen, kein Überholen INNERHALB DES KONFIGURIERTEN EPOCHENBUDGETS (Korrektur SA3/
  Befund 59c: die Zelle unten ist für den Transformer bei `n_train=32` UND `64` an allen
  5 Seeds epochenbegrenzt -- ob er bei ausreichendem Budget überholt, ist mit diesen Daten
  nicht entscheidbar, siehe dort).** Konstantlauf-Holdout,
  `bed_temperature_celsius`, h=36 (die betriebsrelevante Kombination aus a): Transformer
  ist bei `n=8` Faktor 1,40 schlechter als LSTM (0,937 gegen 0,670), bei `n=64` nur noch
  Faktor 1,03 (0,461 gegen 0,449) -- nahezu eingeholt, aber nicht überholt. xLSTM holt auf
  demselben Kanal am stärksten auf (Faktor 6,19 bei `n=8` auf Faktor 1,80 bei `n=64`),
  bleibt aber weiterhin deutlich zurück; Mamba ähnlich (Faktor 3,90 auf 2,46). Auf
  `product_moisture_kg_per_kg` (angeregt, h=36) dagegen bleibt der Transformer-Abstand zu
  LSTM über den gesamten Budgetbereich nahezu konstant (Faktor 2,02 bei `n=8`, Faktor 1,99
  bei `n=64`) -- Aufholen ist kanalabhängig, kein durchgängiges Muster.

**OA5b -- Epochenobergrenze als Konfundierung.** `training/epochs_run/<Kanal>` gegen
`model/net_max_epochs=200`, Anteil der Fits an der Obergrenze, je Architektur über alle
Budgetpunkte/Köpfe/Seeds (60 Zellen = 4 Budgetpunkte × 3 Köpfe × 5 Seeds):
**Transformer 44/60 (73,3 %)**, LSTM 31/60 (51,7 %), GRU 30/60 (50,0 %), Mamba 20/60
(33,3 %), **xLSTM 6/60 (10,0 %)**. Erwartung "bindet mit wachsendem `n_train` häufiger,
nicht seltener" bestätigt sich **nur für Transformer** (60 %→73 %→87 %→73 % von `n=8` zu
`n=64`) und schwächer für xLSTM (0 %→13 %→7 %→20 %); bei GRU (67 %→60 %→47 %→27 %) und
LSTM (53 %→67 %→60 %→27 %) sinkt der Anteil dagegen mit wachsendem Budget -- mehr
Trainingsfenster je Epoche scheinen dort schnellere Konvergenz pro Epoche zu bedeuten,
nicht langsamere.

**Der Transformer-Befund bei `n_train=64` ist der wichtigste Vorbehalt dieses gesamten
Befunds:** von den drei Köpfen sind `energy_target` und `bed_temperature_celsius` bei
`n_train=64` an **allen 5 Seeds** an der Epochenobergrenze (`5/5` je Kopf), nur
`product_moisture_kg_per_kg` stoppt früh (`1/5`). Der Transformer-Ast der Lernkurve bei
`n_train=64` ist für zwei von drei Kanälen deshalb **keine Datenbedarfsaussage, sondern
eine Budgetaussage** -- ob mehr Epochen die Lücke zu GRU/LSTM weiter schließen würden,
ist mit diesem Datensatz nicht zu beantworten. `max_epochs` wurde bewusst NICHT erhöht
(bitgenaues Protokoll war der Zweck dieser Erweiterung); eine gezielte Nachmessung mit
höherem `max_epochs` nur für Transformer bei `n_train=64` wäre ein eigener, hier nicht
ausgeführter Folgeauftrag.

**Nachtrag (Folgeauftrag, `net_max_epochs=600`, nur `n_train=64`, Transformer UND LSTM
neu gerechnet): die Frage ist beantwortet -- aufgeholt, nicht überholt.** SA3 (Befund 59c)
zeigte zusätzlich, dass auch LSTM auf `bed_temperature_celsius` bei `n_train=64` an 2/5
Seeds gecappt war (Transformer: 5/5) -- ein reiner Transformer-Nachlauf hätte die
Konfundierung nur mit umgekehrtem Vorzeichen reproduziert. Deshalb wurden BEIDE
Architekturen bei `net_max_epochs=600` neu gerechnet
(`scripts/run_arch_learning_curve_worker.py --net-max-epochs`, Commit `7e3fb1d`; Datensatz
`artifacts/runs/arch_epoch_budget_probe600.jsonl`, Commit `0ed48e5`, 20 Records, EINZIGE
Protokollabweichung: das Epochenbudget, alles andere bitgleich --
`process/fingerprint=2d2679eeed716577`, dieselben Seeds/Split-Mechanik,
`history_length=32`/`horizon=36`, drei Zielkanäle; Baselines nicht mitgerechnet, kein
Epochenbegriff).

*Erreichte Epochenzahl.* Kein Kopf läuft in irgendeinem der 5 Seeds × 2 Architekturen in
die neue Grenze -- beide Architekturen konvergieren jetzt auf allen drei Köpfen vor 600
Epochen (Transformer maximal 439, Seed 4, `bed_temperature_celsius`; LSTM maximal 345,
Seed 2, `energy_target`). Die Epochenbegrenzung aus Befund 58/59c ist für `n_train=64`
damit vollständig aufgelöst.

*Kernergebnis, `bed_temperature_celsius`, Konstantlauf-Holdout, h=36 (dieselbe Zelle wie
oben/Befund 59c):* Faktor Transformer/LSTM sinkt von 1,026 (200 Epochen) auf **1,000** (600
Epochen, Median über 5 Seeds: Transformer 0,44881 K, LSTM 0,44901 K) -- exakte Parität.
**Transformer überholt LSTM nicht, holt aber vollständig auf, sobald das Epochenbudget
nicht mehr bindet.** Die vierte vorab festgelegte Erwartung aus Befund 58 ("holt auf oder
überholt knapp") trifft im "holt auf"-Teil zu, nicht im "überholt knapp"-Teil.

Auf `energy_target` (Konstantlauf-Holdout, h=36) wächst der Abstand dagegen leicht, statt
gleich zu bleiben: Faktor 1,082 (200 Epochen) auf 1,158 (600 Epochen) -- LSTM verbessert
sich mit mehr Budget stärker als Transformer (Median über 5 Seeds: -5,9 % gegen +0,8 %
relativ). Die vorab festgelegte Erwartung ("Abstand bleibt etwa gleich") trifft damit
NICHT zu, wenn auch die Verschiebung moderat bleibt.

Auf der angeregten Testmenge (h=36) bleibt Transformer bei 600 Epochen auf allen drei
Köpfen hinter LSTM zurück (Faktor 1,15-1,99, Median über 5 Seeds), am deutlichsten bei
`product_moisture_kg_per_kg` -- für diesen Kanal galt weder eine vorab festgelegte
Erwartung noch (mit Ausnahme von Seed 2) eine Epochenbegrenzung.

*Offen bleibt `n_train=32`* (SA3/Befund 59c: Transformer dort auf allen 5 Seeds ebenfalls
epochenbegrenzt). Die oben berichtete Lernkurvensteigung zwischen `n=32` und `n=64`
(-11,6 % gegen -5,8 % bei LSTM) liegt damit zwischen zwei abgeschnittenen Punkten und
vermischt Daten- und Budgeteffekt -- dieser Folgeauftrag rechnet ausschließlich
`n_train=64` nach, eine saubere Steigungsaussage bräuchte zusätzlich `n_train=32` bei
erhöhtem Budget, was hier bewusst nicht gemacht wurde. Analyse:
`scripts/analyze_epoch_budget_probe600.py`.

**Fazit.** Transformer/xLSTM/Mamba sind jetzt protokollgleich zu Befund 51 vermessen, alle
drei liefen ohne Abbruch, keine Architektur musste ausgeschlossen werden. GRU/LSTM bleiben
über den gesamten getesteten Budgetbereich (`n_train≤64`) die besten trainierten Modelle
auf allen drei Zielkanälen und beiden Testmengen; Transformer ist der nächstplatzierte
Kandidat mit klarer Aufholtendenz, aber durch die Epochenobergrenze bei großem `n_train`
selbst mit-konfundiert; xLSTM und Mamba bleiben durchgehend zurück, mit vereinzelten
Nicht-Monotonien, die eher auf Optimierungsverhalten als auf Datenmangel hindeuten.

### 59. Befund 58 korrigiert und eingeordnet: der `energy_target`-Faktor gegen die statische Regression ist keine Testmengen-Falle wie Befund 51 (K5), aber die R²-basierte "fast statisch"-Einordnung trug nicht

**a) SA1 -- auf welcher Testmenge gilt der Faktor 6.4-7.8, und wie sieht er auf beiden
Testmengen aus.** Befund 58 (OA5, vorab festgelegte Erwartungen) zitiert den Faktor für
genau eine Zelle: Testmenge **angeregt**, h=36, `n_train=64`, `rmse_run_median_physical`,
`energy_target` -- statisch 0,1115 kW gegen GRU 0,0145/LSTM 0,0143/Transformer 0,0173 kW
(hier nachgerechnet: Faktor 7,68/7,82/6,43, `scripts/analyze_energy_target_static_factor.py`,
Grundlage `scripts/analyze_arch_learning_curve_extended.py`, Commit `bc6b097`). Diese Zelle
war in Befund 58 bereits korrekt mit Testmengenname zitiert -- offen war, ob derselbe
Faktor auf dem Konstantlauf-Holdout, der betriebsrelevanten Menge (Befund 58 OA5a), Bestand
hat oder wie in Befund 51 (K5) auf einen Bruchteil einbricht.

Vollständiger Vergleich (Median über 5 Seeds, `rmse_run_median_physical`, statisch/Modell,
`energy_target`):

| Testmenge | h | n_train | statisch [kW] | GRU (Faktor) | LSTM (Faktor) | Transformer (Faktor) |
|---|---|---|---|---|---|---|
| angeregt | 1 | 8 | 0,13102 | 0,02384 (5,50) | 0,02928 (4,47) | 0,04815 (2,72) |
| angeregt | 1 | 16 | 0,12832 | 0,01799 (7,13) | 0,01957 (6,56) | 0,03174 (4,04) |
| angeregt | 1 | 32 | 0,11794 | 0,01637 (7,20) | 0,01628 (7,24) | 0,02424 (4,87) |
| angeregt | 1 | 64 | 0,11128 | 0,01490 (7,47) | 0,01491 (7,47) | 0,01852 (6,01) |
| angeregt | 36 | 8 | 0,13243 | 0,03132 (4,23) | 0,04166 (3,18) | 0,04719 (2,81) |
| angeregt | 36 | 16 | 0,12842 | 0,01963 (6,54) | 0,02429 (5,29) | 0,02995 (4,29) |
| angeregt | 36 | 32 | 0,11975 | 0,01640 (7,30) | 0,01704 (7,03) | 0,02250 (5,32) |
| angeregt | 36 | 64 | 0,11152 | 0,01452 (**7,68**) | 0,01426 (**7,82**) | 0,01734 (**6,43**) |
| Konstantlauf | 1 | 8 | 0,12744 | 0,01751 (7,28) | 0,01903 (6,70) | 0,03720 (3,43) |
| Konstantlauf | 1 | 16 | 0,10246 | 0,01513 (6,77) | 0,01680 (6,10) | 0,02345 (4,37) |
| Konstantlauf | 1 | 32 | 0,08709 | 0,01403 (6,21) | 0,01524 (5,71) | 0,01831 (4,76) |
| Konstantlauf | 1 | 64 | 0,07938 | 0,01340 (5,92) | 0,01406 (5,65) | 0,01584 (5,01) |
| Konstantlauf | 36 | 8 | 0,12791 | 0,02196 (5,82) | 0,03229 (3,96) | 0,03385 (3,78) |
| Konstantlauf | 36 | 16 | 0,10336 | 0,01733 (5,96) | 0,01937 (5,34) | 0,02548 (4,06) |
| Konstantlauf | 36 | 32 | 0,08727 | 0,01489 (5,86) | 0,01646 (5,30) | 0,01853 (4,71) |
| Konstantlauf | 36 | 64 | 0,08023 | 0,01399 (**5,74**) | 0,01427 (**5,62**) | 0,01544 (**5,20**) |

**Kernfrage beantwortet: der Faktor bricht auf dem Konstantlauf-Holdout NICHT ein.** An
derselben Budget-/Horizontzelle (`n_train=64`, h=36): Faktor 5,74 (GRU)/5,62 (LSTM)/5,20
(Transformer) -- niedriger als auf der angeregten Menge (7,68/7,82/6,43), aber derselbe
Größenordnungsbereich und durchgehend über der in Befund 51 (K3) verwendeten Schwelle
(Faktor 5). Über den gesamten Budget-/Horizontbereich liegt der Faktor auf dem
Konstantlauf-Holdout zwischen 3,43 (h=1, n=8, Transformer) und 7,28 (h=1, n=8, GRU) --
keine Zelle fällt unter die Schwelle.

**Warum das anders ist als Befund 51 (K5).** Dort brach der Faktor ein, weil die dort
verwendete Baseline (Persistenz) zwischen den Testmengen strukturell unterschiedlich
stark ist: auf `product_moisture_kg_per_kg` ist Persistenz auf der angeregten Menge
strukturell schwach (kann Sollwertsprünge nicht abbilden) und auf dem Konstantlauf
trivial stark (der Holdout IST konstant) -- der große Faktor kam dort ausschließlich aus
dem Nenner, nicht aus einem besseren Modell. Hier ist die Baseline die **statische
Regression**, nicht Persistenz, und die kennt die aktuellen Stellgrößen -- ihr absoluter
Fehler ändert sich zwischen den Testmengen nur um Faktor ~1,4 (0,1115 kW angeregt gegen
0,0802 kW Konstantlauf, h=36/n=64), während Persistenz an derselben Zielgröße um Faktor
6,3 schwankt (Befund 58 OA5a, h=1: 0,1014 gegen 0,0162). Die statische Regression ist
damit auf beiden Testmengen ungefähr gleich schwach -- der Faktor gegen sie ist deshalb
kein Testmengenartefakt, sondern eine robuste Aussage über die Modellgüte.

**b) SA2 -- `energy_target` als "fast statisch"/schwacher Vergleichskanal: Grep-Suche vor
der Änderung, dann Korrektur (analog Befund 50 j).** Gesucht in `docs/*.md` nach "fast
statisch", "wenig dynamisch", "schwacher Vergleichskanal" und sinngemäßen Einordnungen von
`energy_target`. Treffer:

- `docs/04_stand_ap1.md:2223-2225` (Befund 39e): "...Zielgröße `energy_target` (Faktor
  0,71, die laut ... Befund 2, zu 98,7 % statisch erklärbar ist). Nicht bewiesen, aber
  naheliegend: `energy_target` ist nahezu gedächtnislos..." -- **Korrekturziel.**
- `docs/04_stand_ap1.md:2438` (Befund 41c): "...die in Befund 2/39 dokumentierte
  'quasi-statische' Charakteristik (`energy_target` zu 98,7 % statisch erklärbar)."
  -- **Korrekturziel.**
- `docs/04_stand_ap1.md:2455` (Befund 41d): "Ein viertes, mutmaßlich quasi-statisches
  Teilnetz (200 volle Epochen wie `energy_target`)..." -- dieselbe Stelle, im selben
  Nachtrag mitkorrigiert.
- `docs/04_stand_ap1.md:3469` (Befund 50e): "...anders als bei `energy_target`
  (R²_stat = 98,7 %, Befund 2), keine belastbare Referenz für diesen Kanal" -- **kein
  Korrekturziel**: der Satz behauptet nicht, dass die statische Regression nahe am
  Optimum liegt, sondern kontrastiert nur, dass sie (anders als bei der Ablufttemperatur)
  nicht komplett flach bei ~6 K verharrt. Bleibt unverändert.
- `docs/04_stand_ap1.md:5211-5224` (Befund 58, OA5): bereits korrekt mit Testmenge
  ("Angeregt") zitiert und widerlegt dort bereits selbst die pauschale "alle nahe der
  statischen Regression"-Erwartung -- **kein Korrekturziel**, durch (a) oben nur um die
  zweite Testmenge ergänzt.
- `docs/05_arbeitsplan_ap1.md:105-106`: "`energy_target` läuft als Kontrollgröße mit"
  -- **kein Korrekturziel**: das ist eine Priorisierungsentscheidung für das
  Trainingsziel des Rollout-Trainings (welche Zielgröße im Fokus steht), keine Aussage
  über die Validität von `energy_target` als Architekturvergleichskanal. Bleibt
  unverändert.
- `docs/02_prozess_setup.md`: keine der drei gesuchten Formulierungen gefunden. Enthält
  aber die Zahl, an der die Korrektur unten hängt: R²_stat = 0,972 (GAB-reproduziert, fp
  `82f65f576a2e5253`, Zeile 191) -- **nicht** die in Befund 2/39/41/58 zitierten 98,7 %,
  die dieselbe Tabelle (Zeile 200) explizit als "historisch, nicht reproduzierbar"
  kennzeichnet. Die überall zitierte Zahl ist bereits die ungünstigere von beiden
  (höheres R² → kleinere Restvarianz → die "fast statisch"-Einordnung sähe mit ihr noch
  berechtigter aus, als sie ist).

**Korrektur, wie Befund 50 j: die Einordnung war aus R² abgeleitet und trägt in
physikalischen Einheiten nicht -- hier zusätzlich durch SA1 (a) abgesichert: die
Korrektur gilt auf BEIDEN Testmengen, nicht nur der angeregten.** Restvarianz der
statischen Regression bei R²_stat = 0,972: 2,8 %. Ein Faktor k im RMSE entspricht Faktor
k² in der Residuenvarianz (gleicher Nenner vorausgesetzt) -- bei den in (a) gemessenen
Faktoren (`n_train=64`, h=36) ergibt sich:

| Testmenge | Modell | Faktor RMSE | Resultierendes R² |
|---|---|---|---|
| angeregt | GRU | 7,68 | 0,999525 |
| angeregt | LSTM | 7,82 | 0,999542 |
| angeregt | Transformer | 6,43 | 0,999323 |
| Konstantlauf | GRU | 5,74 | 0,999150 |
| Konstantlauf | LSTM | 5,62 | 0,999113 |
| Konstantlauf | Transformer | 5,20 | 0,998964 |

Auf beiden Testmengen liegen die trainierten Modelle bei R² ≥ 0,9990 (angeregt) bzw.
≥ 0,9989 (Konstantlauf) -- gegenüber R²_stat = 0,972 (bzw. sogar 0,987, würde die
historische, nicht reproduzierbare Zahl verwendet) eine Verschiebung, die in R²-Einheiten
winzig aussieht (dritte/vierte Nachkommastelle), aber in physikalischen Einheiten Faktor
5,2-7,8 im RMSE bedeutet -- exakt dasselbe Muster wie Befund 50 j bei der Ablufttemperatur
(dort Faktor 20 hinter R²_stat = 0,768/R²_holdout = 0,965), nur bei kleinerer absoluter
Fehlerskala. **Einordnung korrigiert:** `energy_target` ist trotz hohem R²_stat kein
schwacher Architekturvergleichskanal -- die trainierten Modelle schließen eine
physikalisch bedeutsame Lücke, die R² allein verdeckt, und das gilt (anders als bei
Befund 51 K5) auf beiden Testmengen, nicht nur der angeregten.

**c) SA3 -- Transformer: "überholt nicht" ist auf das getestete Epochenbudget
einzuschränken, nicht als generelles Ergebnis zu lesen.** Befund 58 (OA5, vierte vorab
festgelegte Erwartung) formuliert "deutliches Aufholen, kein Überholen im getesteten
Bereich" und zitiert dafür `bed_temperature_celsius`, Konstantlauf-Holdout, h=36:
Transformer Faktor 1,40 hinter LSTM bei `n=8`, Faktor 1,03 bei `n=64`. Nach Befund 58
(OA5b) selbst ist der Transformer bei `n_train=64` an zwei von drei Köpfen (`energy_target`,
`bed_temperature_celsius`) an allen 5 Seeds an der Epochenobergrenze
(`model/net_max_epochs=200`) -- genau der Kanal, an dem die Aufhol-Aussage gemessen wird,
ist also an der Stelle abgeschnitten, an der der Transformer die Lücke schließt.

Epoch-Bindung, geprüft statt behauptet (`training/epochs_run/bed_temperature_celsius ==
model/net_max_epochs`, je 5 Seeds; dasselbe trainierte Modell wie bei "angeregt", da nur
einmal trainiert und zweimal ausgewertet wird, siehe Skript-Docstring
`run_arch_learning_curve_worker.py`):

| n_train | Transformer epochenbegrenzt | LSTM epochenbegrenzt |
|---|---|---|
| 8 | 4/5 | 4/5 |
| 16 | 4/5 | 4/5 |
| 32 | **5/5** | 3/5 |
| 64 | **5/5** | 2/5 |

Bei `n_train=32` UND `64` ist der Transformer an ALLEN 5 Seeds epochenbegrenzt -- die
Zelle, in der die "Faktor 1,40 → 1,03"-Aufholung gemessen wird, ist für den Transformer
durchgehend eine Budget-, keine Datenbedarfsaussage. LSTM ist an denselben Punkten nur zu
3/5 bzw. 2/5 gebunden -- bei der Mehrheit der Seeds konvergiert vor der Obergrenze.

**Korrigierte Formulierung:** Transformer überholt LSTM **innerhalb des konfigurierten
Epochenbudgets** nicht. Ob er bei ausreichendem Budget überholt, ist mit diesen Daten
**nicht entscheidbar** -- offene Frage, kein Ergebnis.

Lernkurven-Steigung zwischen `n_train=32` und `64`, für die epochenbegrenzten Zellen,
Transformer gegen LSTM (`rmse_run_median_physical/bed_temperature_celsius/h36`,
Konstantlauf-Holdout, Median über 5 Seeds):

| Modell | n=32 | n=64 | Δ absolut | Δ relativ |
|---|---|---|---|---|
| Transformer | 0,5212 K | 0,4606 K | -0,0606 K | **-11,6 %** |
| LSTM | 0,4769 K | 0,4490 K | -0,0278 K | -5,8 % |

Der Transformer fällt zwischen `n=32` und `64` relativ knapp doppelt so stark wie LSTM --
**das ist ein Indiz, kein Beleg**, dass er bei mehr Budget weiter aufholen oder überholen
könnte: dieselbe Zelle ist für den Transformer an BEIDEN Stützstellen epochenbegrenzt, für
LSTM überwiegend nicht, sodass die steilere Transformer-Kurve auch bedeuten könnte, dass
die Epochenobergrenze bei `n=64` weniger bindend wirkt als bei `n=32` (mehr
Trainingsfenster je Epoche, siehe Befund 58 OA5b, wo GRU/LSTM genau diesen Effekt zeigen)
-- nicht zwingend, dass der Transformer mit mehr Epochen tatsächlich überholen würde.
Nicht weiter geprüft (kein Lauf mit erhöhtem `max_epochs`; bewusst nicht umgesetzt, um das
bitgenaue Protokoll zu Befund 51 nicht zu verlassen, siehe Befund 58).

### 60. Pinch-Zielwert (Problem-Table-Algorithmus): die Luecke der sechs Verschaltungsfaelle zum thermodynamischen Optimum liegt bei 2-4 %, nicht bei den mehreren Faktoren, die die 2.-Hauptsatz-Verletzungen aus Befund 57 nahelegen

**Nummerierung.** Nummer bei der Zusammenfuehrung vergeben. Ein Fast-Forward war nicht
moeglich, weil master sich zwischenzeitlich bewegt hatte; zusammengefuehrt wurde per echtem
Merge. Code, Tests und Doku auf dem Zweig `lp-benchmark` (separates Arbeitsverzeichnis,
siehe unten), Hauptverzeichnis wurde fuer diesen Auftrag nicht angefasst.

**Frage:** Befund 57/57h stellen fest, dass drei der sechs Verschaltungsfaelle des
Partnermodells (`Model_v3.xlsx`) rechnerisch Waerme "bergauf" verschieben (2.
Hauptsatz-Verletzung) -- aber wie GUT ist der BESTE der sechs Faelle im Vergleich zum
thermodynamisch bestmoeglichen Betrieb? Ohne diesen Massstab ist unklar, ob die
Verschaltungsfrage praktisch erledigt ist oder ob eine bessere Verschaltung als alle sechs
existiert.

**Methode: Problem-Table-Algorithmus statt LP** (neues Modul
`heatnetwork/pinch.py`). Fuer das reine Zielwertproblem (minimaler Zusatzbrennstoff, keine
Frage nach der Anzahl/Anordnung der Waermeuebertrager) ist der klassische Problem-Table-
Algorithmus (Pinch-Analyse) exakt und braucht keinen Solver -- ein LP waere erst fuer die
Netzwerksynthese noetig (welcher Uebertrager wohin), nicht Teil dieses Auftrags. Verschobene
Temperaturen (heiss um `dT_min/2` nach unten, kalt nach oben), Kaskade der
Intervall-Waermebilanzen von der hoechsten Temperatur nach unten, groesste negative
Kaskadenlast = minimaler Heizbedarf, Nulldurchgang = Pinch-Temperatur.

**Kondensation: keine zweite Formel.** Die Abluftquellen haben unterhalb ihres Taupunkts
keine konstante Waermekapazitaet mehr (latente Waerme). Statt einer neuen
Kondensationsformel tastet `hot_segments_for_source` die Enthalpie-Temperatur-Kurve am
BEREITS VORHANDENEN `heatnetwork.balance.evaluate_cascade` (`recovery_mode="condensing"`)
ab -- derselbe, bereits getestete Code, nur wiederholt bei verschiedenen `T_kalt`-Werten
aufgerufen, um Stuetzstellen einer Kurve zu erzeugen. Oberhalb des Taupunkts ist die Kurve
exakt linear (ein Segment ohne Diskretisierungsfehler); unterhalb wird sie an
`n_segments_below_dew_point` Stuetzstellen abgetastet, mit dem Taupunkt selbst als exakter
Intervallgrenze (kein Raster, das ihn zufaellig treffen muesste).

**Datenfall (PA2, korrigierte Werte nach Befund 57):** Q1 (Sprühturm, 20 kg/s, 125,233 °C,
`Y=0,06482`), Q2 (Wirbelschicht, 8 kg/s, 81,416 °C, `Y=0,01998`), S1 (20 kg/s, 15→300 °C),
S2 (8 kg/s, 15→120 °C), `H_u=45 MJ/kg`, `eta_H=0,80`. `T_kalt` als Parameter ueber
`{15; 25; 29,1; 35} °C`, `dT_min` ueber `{10; 15; 20} K` -- 12 Kombinationen
(`scripts/run_pinch_benchmark.py`).

**Korrektur einer frueher genannten Zahl (WICHTIG, betrifft Befund 57h).** Die dort (und im
urspruenglichen Auftragstext dieses Zweigs) genannten Taupunkte 47,9 °C (Q1) und 27,93 °C
(Q2, an anderer Stelle verwendet) waren KEINE Taupunkte, sondern Taupunkt PLUS die 3-K-
Abkuehlgrenze aus `Model_v2`, `Balances!F19` (die dortige Magnus-Formel endet auf "+ 3") --
eine Zellen-Verwechslung, keine Formel- oder Modellabweichung. Verifiziert (per
Handrechnung durch den Nutzer, mit `heatnetwork.pinch` gegengeprueft: Q1 bei `T_kalt=10 °C`
gesamt 5429,8 kW, davon 1809,1 kW oberhalb und 3620,7 kW unterhalb des Taupunkts, beide
Werte auf < 0,1 % Uebereinstimmung mit der unabhaengigen Handrechnung):

- Q1 Taupunkt korrekt: **44,94 °C** (Buck-Korrelation, `thermo.dew_point_celsius`)
- Q2 Taupunkt korrekt: **24,92 °C**

Die bereits in Befund 57h dokumentierte Tabelle (`Q_rec` je Quelle ueber `T_kalt`) verwendet
bereits die korrekten Buck-Werte und war nicht betroffen -- nur die textliche
Gegenueberstellung mit den beiden falschen Zahlen ist hiermit aufgeloest (dort korrigiert,
siehe Nachtrag an Ort und Stelle).

**PA4 -- wo die Kondensationswaerme anfaellt, geschaerft statt abgeschwaecht.** Bei
`T_kalt=15 °C` liegen von Q1s 5154,9 kW rueckgewinnbarer Leistung nur 1809,1 kW (35 %)
oberhalb ihres Taupunkts (44,94 °C); die restlichen 65 % sind niedrigwertige Waerme
(sensibler Rest plus Kondensationsenthalpie unterhalb 45 °C). Senke S1 muss Luft aber auf
300 °C bringen -- ZWEI DRITTEL der Q1-Waerme liegen damit auf einem Temperaturniveau, das
fuer S1 physikalisch nicht direkt nutzbar ist (2. Hauptsatz). Das ist die mechanistische
Erklaerung dafuer, dass drei der sechs Faelle (Befund 57) Waerme "bergauf" verschieben:
sie verrechnen diese niedrigwertige Waerme, als waere sie hochwertig.

**PA3 -- die Kernaussage: die Luecke ist klein (2,3-4,2 %), nicht die mehreren Faktoren,
die die Verletzungen vermuten liessen.**

| `T_kalt` [°C] | `dT_min` [K] | Pinch [kg/h] | bester Fall | Fall-Wert [kg/h] | Luecke | Pinch-T (heiss/kalt) |
|---|---|---|---|---|---|---|
| 15,0 | 10 | 404,90 | Fall1 | 414,65 | 2,4 % | 81,42/71,42 °C |
| 15,0 | 20 | 433,33 | Fall1 | 443,08 | 2,3 % | 81,42/61,42 °C |
| 29,1 | 10 | 404,90 | Fall1 | 416,81 | 2,9 % | 81,42/71,42 °C |
| 35,0 | 10 | 404,90 | Fall1 | 421,74 | 4,2 % | 81,42/71,42 °C |
| 35,0 | 20 | 433,33 | Fall1 | 443,08 | 2,3 % | 81,42/61,42 °C |

(volle 4×3-Tabelle in `artifacts/runs/run_pinch_benchmark.json`). Fall1 (Q1→S1 getrennt,
Q2→S2 getrennt -- die naheliegendste, unkaskadierte Verschaltung) ist in JEDER getesteten
Kombination der beste der sechs Faelle und bleibt durchgaengig innerhalb von 2,3-4,2 % des
Pinch-Ziels. **Die Verschaltungsfrage ist damit fuer diesen Datenfall praktisch geloest** --
der Vorbehalt unten grenzt ein, was das genau heisst.

**Unerwarteter Nebenbefund: der Pinch sitzt am HEISSEN Ende, nicht in der
Kondensationszone -- `T_kalt` aendert den minimalen Heizbedarf gar nicht.** Die
Pinch-Temperatur liegt in JEDER getesteten Kombination exakt bei 81,42 °C (heiss) --
bit-identisch mit `T_out` von Q2. Das bedeutet: der bindende Engpass liegt im Intervall
oberhalb von Q2s Eintritt, wo nur Q1 verfuegbar ist, um S1s Anspruch auf 300 °C zu decken --
weit oberhalb beider Taupunkte. Konsequenz: `minimum_hot_utility_kw` ist ueber die gesamte
`T_kalt`-Sweep (15/25/29,1/35 °C) BIT-IDENTISCH (4048,99 kW bei `dT_min=10`, geprueft bis
auf die letzte Nachkommastelle) -- nur der Kuehlbedarf faellt mit steigendem `T_kalt`
(3302→1196 kW bei `dT_min=10`). Aus demselben Grund aendert sich der Pinch-Zielwert bei
Verfeinerung von `n_segments_below_dew_point` (2 bis 1000 Stuetzstellen getestet) um
0,0000 % -- die Kondensationskurve liegt komplett im Ueberschussbereich unterhalb des
Pinch. **Das ist eine Eigenschaft DIESES Datensatzes** (S1 verlangt 300 °C, weit oberhalb
jeder Quellentemperatur), keine allgemeine Garantie: bei einem anderen Sollwertprofil
koennte der Pinch in die Kondensationszone fallen, dann wuerde sowohl `T_kalt` als auch die
Abtastaufloesung sehr wohl zaehlen. Praktische Konsequenz fuer den Verbund: die im
Sechs-Faelle-Modell beobachtete Empfindlichkeit gegenueber `T_kalt` (Befund 57h) ist ein
Artefakt der GEWAEHLTEN Verschaltungen, nicht der Physik -- das thermodynamische Optimum
braucht die tiefe Kondensation von Q2 gar nicht, um seinen Bestwert zu erreichen.

**Vorbehalt (MUSS bei jeder Verwendung dieser Zahlen genannt werden): untere Schranke,
keine Bauvorschrift.** Der Pinch-Zielwert setzt beliebig viele, ideal gegeneinander
geschaltete Waermeuebertrager voraus. Eine Luecke von wenigen Prozent (wie hier) kann
vollstaendig durch die Beschraenkung der sechs Faelle auf zwei Uebertrager erklaert sein --
sie ist kein Beweis, dass eine bessere REALISIERBARE Verschaltung nicht existiert, nur dass
sie, wenn sie existiert, hoechstens 2-4 % einsparen kann.

**Tests (`tests/test_heatnetwork_pinch.py`, 18 neue Tests):** Uebereinstimmung mit einer
selbst konstruierten Vierstrom-Aufgabe (Handrechnung im Testdocstring, kein zitierter
Lehrbuchwert -- eine falsch erinnerte Literaturzahl waere ein schlechterer Massstab),
Grenzfall `dT_min=0`, zwei Monotonien (`dT_min` und `T_kalt` duerfen den Heizbedarf nie
senken), Taupunkt als exakte Intervallgrenze, Aufloesungskonvergenz, und der wichtigste
Test: der Pinch-Zielwert darf ueber die gesamte 4×3-Parameterkombination NIE groesser sein
als der beste der sechs Faelle -- eine mathematische Eigenschaft, keine
Implementierungsdetailfrage (waere sie verletzt, waere die Implementierung falsch, nicht
die Faelle).

**Reproduzierbarkeit.** Modul `src/twinlink_ml/heatnetwork/pinch.py`, Skript
`scripts/run_pinch_benchmark.py` (JSON-Ausgabe `artifacts/runs/run_pinch_benchmark.json`,
inkl. heisser/kalter Verbundkurve und Grand Composite Curve als Stuetzpunkte, PA4), Tests
`tests/test_heatnetwork_pinch.py`. Ruff und die vollstaendige Testsuite vor jedem Commit
gruen.

### 61. Der Standort-Pinch wird von der Ablufttemperatur der Wirbelschichtlinie gesetzt -- und springt zwischen zwei verschiedenen Engpassstellen, je nachdem wie die Linie gefahren wird -- **AUSGESETZT (R5): baut auf der fehlerhaften NA5-Parametrierung auf, siehe Befund 56 NB1**

**R5 -- Status: ausgesetzt, nicht neu gerechnet.** Dieser gesamte Befund importiert
`build_plant_config`/`OperatingPoint`/`_solve_steady_state_widening_bracket` unveraendert
aus `study_plant_validation.py` (siehe Reproduzierbarkeit unten) -- also inklusive des in
Befund 56 (NB1) korrigierten Luftstrom-Fehlers (Faktor 2,19/2,20, relative statt absolute
Feuchte) UND des dort unveraendert gebliebenen 30-%-Feststoffplatzhalters, der laut
Befund 56 (NB2) mit korrigiertem Luftstrom nicht mehr haltbar ist (Vorhersage: 71-73 %).
Mit korrigiertem Luftstrom und weiterhin 30 % Feststoff findet der Loeser bei RID **keinen
Beharrungszustand** (Befund 56 NB1) -- das gesamte QA1-QA4-Gitter dieses Befunds haengt
damit direkt am (jetzt als falsch erkannten) Feststoffanteil und ist nicht ohne Weiteres
reproduzierbar. Betroffen sind insbesondere QA2 (Stellgroessengitter/Fluidisierungsausfaelle,
bisher dem d50-Platzhalter zugeschrieben -- der Luftstrom-Fehler kommt als zweite,
ununterscheidbare Ursache hinzu) sowie RA1-RA3 vollstaendig. **Nicht neu gerechnet, wie
angewiesen** -- erst sobald der reale Feststoffanteil vom Anwendungspartner vorliegt oder
Befund 56s Vorhersage (NB2) anderweitig bestaetigt ist. Alle folgenden Zahlen dieses
Befunds sind bis dahin als vorlaeufig/ausgesetzt zu lesen, nicht als gueltiges Ergebnis.

**Kernaussage, ohne den Rest lesbar (historischer Stand, vor R5).** Der minimale Brennstoffbedarf des Standorts (Befund
60, `lp-benchmark`-Zweig, Pinch-Zielwert) wird durchgaengig entweder von der
Ablufttemperatur der Wirbelschichtlinie (Quelle 2) ODER vom Taupunkt der Sprühturmlinie
(Quelle 1, fest) gesetzt -- NIE von etwas anderem. Da die Wirbelschichtlinie das einzige
Prozessmodell dieses Projekts ist und ihre Ablufttemperatur ueber die Prozesssollwerte
(Zulufttemperatur, Gasgeschwindigkeit, Spruehrate) verstellbar ist, ist die Prozessfahrweise
damit ein Hebel auf die Waermeintegration des GESAMTEN Standorts -- kein rein lokaler
Effekt. Das Bindeglied zwischen Befund 52 (Regret eines festen Betriebspunkts, 16-26 %,
Labormassstab) und der Waermeintegration (Anlagenmassstab) ist damit hergestellt: eine
Regelung der Wirbelschichtfahrweise wirkt nicht nur auf den eigenen Energieverbrauch,
sondern auch auf den Brennstoffbedarf des Verbunds.

**QA1 -- Massstab: Anlagenparameter statt Skalierung (Befund 56/NA5 wiederverwendet).**
Befund 52 liegt im Labormassstab (~0.03 kg/s), der Pinch-Datenfall im Anlagenmassstab
(8 kg/s) -- die dortigen optimalen Sollwerte wurden NICHT uebernommen. Stattdessen: der
bereits validierte NA5-Parametersatz (`scripts/study_plant_validation.py`,
`build_plant_config`), Anker ACE-Betriebspunkt (bessere Validierung, -0.4 K gegen -2.1 K
bei RID):

    T_amb=4.9 °C, T_zuluft=84.6 °C, Luftmenge=33003 Nm³/h, Spruehrate=934 kg/h,
    T_bett(gemessen)=67.3 °C, Bettmasse=800 kg (muendliche Partnerangabe, fuer T_bed
    irrelevant, Befund 56).

`openpyxl` fehlte anfangs in dieser Umgebung (nachinstalliert, um NA5 direkt gegen die
Rohdaten laufen zu lassen, statt die Anlagenparameter neu zu erfinden -- Nutzerentscheidung
nach Rueckfrage). Quelle 1 (Sprühturm) bleibt FEST auf den Model_v3-Werten (PA2):
20 kg/s, 125.233 °C, `Y=0.06482` -- dieses Projekt bildet nur die Wirbelschichtlinie ab.

**QA2 -- Gitter, dominiert von einer bereits bekannten Modellluecke.** Drei Stellgroessen
(Zulufttemperatur, Gasgeschwindigkeit, Spruehrate) bei 25/50/75 % der (aus NA5
uebernommenen) ActionConfig-Spanne, 27 Punkte. Ergebnis: **nur 3 von 27 Punkten sind
zulaessig.** Der Grossteil scheitert an der Fluidisierungsobergrenze
(`u_gas < u_terminal(d50)`) -- bei ALLEN Punkten mit der mittleren oder hohen
Geschwindigkeitsstufe (4.6/6.9 m/s), unabhaengig von Zulufttemperatur/Spruehrate. Ursache:
`u_terminal` haengt am Partikeldurchmesser (`d50≈750 µm`), einem unveraendert
uebernommenen LABORPLATZHALTER (Befund 56: "realistische Industriegranulate dieser Anlage
sind vermutlich groesser/dichter, wurden aber NICHT angepasst"). Das ist dieselbe
Modellluecke wie in Befund 56, hier mit einer NEUEN Konsequenz: sie schneidet zwei Drittel
des hier gebauten Stellgroessengitters weg, nicht nur die Fluidisierungspruefung selbst.
Zusaetzliche Ausfaelle: kein Beharrungszustand bei zu niedriger Zulufttemperatur
kombiniert mit hoher Spruehrate (physikalisch plausibel -- zu wenig Waermeeintrag zum
Trocknen), sowie Verletzung der (ebenfalls unbelegten) kritischen Gutsfeuchte
(0.02 kg/kg) bei hoher Spruehrate/niedriger Geschwindigkeit.

**QA3 -- der Pinch springt zwischen zwei Engpassstellen.**

| T_zuluft | u_gas | Spruehrate | T_out (Q2) | Pinch | Lage | Standort-Brennstoff | Waerme oberhalb Pinch |
|---|---|---|---|---|---|---|---|
| 75.0 °C | 2.31 m/s | 750 kg/h | 40.94 °C | 44.94 °C | Q1-Taupunkt | 454.84 kg/h | 1809.1 kW |
| 112.5 °C | 2.31 m/s | 1500 kg/h | 43.74 °C | 44.94 °C | Q1-Taupunkt | 454.84 kg/h | 1809.1 kW |
| 112.5 °C | 2.31 m/s | 750 kg/h | 73.50 °C | 73.50 °C | Q2-Eintritt | 438.00 kg/h | 1165.6 kW |

Antworten auf die drei Kernfragen:
- **Wandert der Pinch, und ueber welchen Bereich?** Ja -- 44.94 bis 73.50 °C ueber die drei
  zulaessigen Punkte (28.56 K Spanne), exakt der Bereich, in dem sich `T_out` von Quelle 2
  bewegt.
- **Bleibt er am Eintritt von Quelle 2, oder springt er?** Er SPRINGT -- und zwar genau
  dann, wenn `T_out`(Q2) den Taupunkt von Quelle 1 (44.94 °C) unterschreitet. Oberhalb
  dieser Schwelle sitzt der Pinch exakt am Eintritt von Q2 (verstellbar, folgt `T_out`
  direkt). Unterhalb dieser Schwelle wird Q2 fuer die obere Temperaturregion irrelevant --
  der Pinch springt auf einen FESTEN Punkt, den Taupunkt von Quelle 1 (44.94 °C,
  unveraendert, weil Quelle 1 fest ist) -- **und bleibt dort, unabhaengig davon, wie weit
  `T_out`(Q2) noch weiter faellt** (zwei Punkte mit `T_out`=40.94 °C und 43.74 °C liefern
  identisch 454.84 kg/h). Das ist der "interessantere Fall" aus dem Auftrag: eine
  Strukturaenderung des Problems, kein reiner Zahlenwert -- **die Sechs-Fälle-typische
  2.-Hauptsatz-Falle (Befund 57, PA4) trifft in diesem Regime NICHT Quelle 2, sondern
  weiterhin Quelle 1**, weil Quelle 2 fuer die Hochtemperatur-Seite entkoppelt ist.
- **Wie stark aendert sich der minimale Brennstoffbedarf?** Ueber die (sehr schmale)
  zulaessige Stichprobe: 438.00-454.84 kg/h, absolut 16.84 kg/h, relativ **3.8 %** --
  trotz der strukturellen Springerei bleibt die absolute Fuellstandsaenderung klein, weil
  Quelle 1 (fest, 20 kg/s) in jedem Fall den groessten Teil der Hochtemperaturlast traegt.

**QA4 -- ja, es gibt einen Zielkonflikt, klein aber real.** Der Punkt mit dem HOECHSTEN
`energy_target` (1215.7 kW, T=112.5 °C/S=1500 kg/h) braucht MEHR Standort-Brennstoff
(454.84 kg/h) als der Punkt mit MITTLEREM `energy_target` (1113.7 kW, T=112.5 °C/S=750 kg/h,
438.00 kg/h) -- eine Inversion. Groesster gefundener Effekt: **+297 kW Prozessleistung
"kaufen" 16.84 kg/h Standort-Erdgas-Ersparnis** (Vergleich des guenstigsten mit dem
mittleren Punkt: 816.6 kW/454.84 kg/h gegen 1113.7 kW/438.00 kg/h). Die physikalische
Ursache ist eindeutig: der teurere Punkt (S=1500 kg/h) druckt Q2s Ablufttemperatur unter
Q1s Taupunkt und schaltet damit auf das Q1-Taupunkt-Regime um (454.84 kg/h), waehrend der
mittlere Punkt (S=750 kg/h) Q2 heiss genug haelt, um selbst den Pinch zu setzen (438.00
kg/h, dem GUENSTIGEREN Regime). **Das ist eine thermodynamische Begruendung fuer eine
Regelung auf Standortebene, keine Gewichtungsannahme** -- mit dem Vorbehalt, dass der
gemessene Effekt (297 kW / 16.84 kg/h) auf einer Stichprobe von nur drei Punkten beruht
(siehe QA2-Vorbehalt zur Fluidisierungsgrenze) und `energy_target` (Prozessseite, gemischt
Strom/Gas ueber vier Aggregate, `heater_efficiency=0.98`) NICHT direkt mit dem
Standort-Brennstoff (reines Erdgas-Aequivalent, `eta_H=0.80`) aufaddierbar ist -- dieselbe
Vorsicht wie in Befund 57i gegen eine unbegruendete kW-Summe ueber Energietraeger hinweg.

**Nachtrag RA1 -- Doppelzaehlung geprueft: NEIN, die Sinkseite des Standort-Pinch ist von
der Grid-Fahrweise vollstaendig entkoppelt.**

*Komponenten von `energy_target`* (`simulation/energy.py:136-166`, `target="power_kw"`):

    P_Heizung  = m_L * (cp_L + Y_in*cp_D) * max(T_in - T_amb, 0) / eta_H   (heater_kw)
    P_Ventilator = u*A * (dp_bed + zeta*rho_g*u^2/2) / eta_V                (fan_kw)
    P_Pumpe    = m_spray * dp_P / (rho_susp * eta_P)                       (pump_kw)
    P_Zerstaeuberluft = m_Z * R_L * T_amb * ln(p_nozzle/p_amb) / eta_K     (atomiser_kw)
    energy_target = P_Heizung + P_Ventilator + P_Pumpe + P_Zerstaeuberluft

`P_Heizung` ist der Lufterhitzer der Wirbelschichtlinie, mit den GRID-Werten dieses
Punkts (`T_in` aus dem 27-Punkte-Gitter, `T_amb=4.9 °C` fest) und OHNE jede
Ruckgewinnungsgutschrift (reine Ambient->T_in-Erwaermung).

*Senken im Pinch-Datenfall* (`scripts/run_process_pinch_sensitivity.py:147-153`, woertlich
Modulkonstanten, Docstring nennt sie explizit "Senken unveraendert"):

    SOURCE_1_SPRUEHTURM = HeatSourceConfig("Q1", 20.0, 125.233, 0.06482)   # fest, PA2
    SINK_1 = HeatSinkConfig("S1", 20.0, 0.005, 15.0, 300.0)                # fest, PA2
    SINK_2 = HeatSinkConfig("S2", 8.0, 0.005, 15.0, 120.0)                 # fest, PA2

`SINK_2` (der Wirbelschicht-Lufterhitzer aus Sicht des Standort-Pinch) hat einen FESTEN
Sollwert 120 °C bei FESTEM Massenstrom 8 kg/s, unveraendert aus dem PA2/Model_v3-
Sechs-Faelle-Datenfall uebernommen -- fuer ALLE 27 Gitterpunkte IDENTISCH, unabhaengig
davon, welches `T_in` (37.5/75/112.5 °C) oder welche Spruehrate der jeweilige Gitterpunkt
tatsaechlich fahren. Einzig `source_2` (Q2) variiert mit dem Gitterpunkt (aus der
Beharrungszustandsloesung: `dry_air_mass_flow_kg_per_s`, `bed_temperature_celsius`,
`outlet_humidity_kg_per_kg`, siehe `evaluate_grid_point`, Zeile 261-266) -- `SOURCE_1`,
`SINK_1`, `SINK_2` sind Modulkonstanten, werden bei JEDEM der 27 Aufrufe unveraendert
uebergeben.

**Kernfrage.** Steckt der Heizbedarf der Wirbelschichtlinie in beiden Zahlen? Begrifflich
JA -- beide sollen denselben Aggregattyp abbilden (den Lufterhitzer der Wirbelschichtlinie).
Rechnerisch aber NEIN in dem Sinn, der fuer QA4s Vergleich zaehlt: weil `SINK_2` konstant
ist, traegt die Sinkseite NICHTS zu der Differenz zwischen zwei Gitterpunkten bei -- jede
Aenderung von `minimum_hot_utility_kw`/`pinch_fuel_kg_per_h` zwischen zwei Punkten kommt
ausschliesslich aus `source_2` (Q2), also aus der veraenderten ABLUFTVERFUEGBARKEIT, nicht
aus einer veraenderten Sinkennachfrage. Das im Skript-Docstring vermerkte "Senken
unveraendert" ist damit keine Nebensaechlichkeit, sondern methodisch entscheidend: es
isoliert genau den Source-seitigen Effekt (heissere Wirbelschicht-Abluft speist mehr
Rueckgewinnung fuer den GESAMTEN, unveraendert gehaltenen Standort), den QA4 zeigen soll.
Der Standort-Brennstoffrueckgang (16.84 kg/h zwischen den QA4-Vergleichspunkten) ist damit
ein ZUSAETZLICHER, von `energy_target` UNABHAENGIGER Effekt (mehr rueckgewinnbare Abluftwaerme
fuer die fixe Sinkennachfrage) -- keine teilweise Rueckerstattung der in `energy_target`
bereits ohne Rueckgewinnungsgutschrift verbuchten Mehrkosten. Die urspruengliche
Vermutung im Auftrag (Faktor 210.5 kW Brennstoffenergie gegen 297 kW Einsatz, "Verlust"
86.5 kW, FALLS beide Zahlen unabhaengig sind) beruht auf genau dieser Unabhaengigkeit --
sie ist bestaetigt, der "Verlust" ist keiner: `energy_target` und `pinch_fuel_kg_per_h`
sind zwei separate Fragen (Kosten dieses Gitterpunkts ohne Rueckgewinnung vs.
Standort-Rueckgewinnungspotenzial bei fixer Sinkennachfrage), kein Vergleich derselben
Groesse aus zwei Blickwinkeln. **Kein korrigierter Vergleich noetig -- QA4s
Zielkonflikt-Aussage bleibt wie berichtet gueltig.**

**Vorbehalt (neu, RA1).** Weil `SINK_2` NICHT an das Gitter gekoppelt ist, beantwortet
diese Studie nicht die Frage "wie viel Standort-Brennstoff spart eine bestimmte
Wirbelschicht-Fahrweise INKLUSIVE Selbstrueckgewinnung fuer die eigene Zuluft" -- dafuer
muesste `SINK_2` mit `T_in`/Massenstrom des jeweiligen Gitterpunkts mitgefuehrt werden
(eigene, hier nicht beauftragte Erweiterung). Die vorliegende QA4-Aussage bleibt auf die
tatsaechlich gerechnete Frage beschraenkt: Wirkung der Wirbelschicht-ABLUFT auf einen
FESTEN Standort-Waermebedarf.

**Vorbehalte (PA3/QA4, MUESSEN bei jeder Verwendung genannt werden).**
- Der Pinch-Zielwert ist eine UNTERE SCHRANKE (beliebig viele, ideal geschaltete
  Waermeuebertrager vorausgesetzt), keine Bauvorschrift.
- Quelle 1 (Sprühturm) ist fest -- in der Realitaet wuerde eine Aenderung der
  Wirbelschicht-Fahrweise den Spruehturm nicht beruehren, das ist also eine zulaessige,
  aber explizit zu nennende Vereinfachung.
- Reine Betriebspunktvariation im Beharrungszustand, keine Transiente.
- Die kritische Gutsfeuchte (0.02 kg/kg) ist ein Platzhalter ohne Quelle -- mehrere der 24
  unzulaessigen Punkte scheitern (auch) daran, nicht nur an der Fluidisierung.
- Der Partikeldurchmesser (`d50≈750 µm`, Laborplatzhalter, Befund 56) bestimmt
  `u_terminal` und damit direkt, WELCHE Gitterpunkte ueberhaupt zulaessig sind -- die 3-von-
  27-Ausbeute ist zu einem erheblichen Teil ein Artefakt dieser unkalibrierten Groesse,
  keine reine Prozessaussage.

**Reproduzierbarkeit.** `scripts/run_process_pinch_sensitivity.py` (importiert
`build_plant_config`/`OperatingPoint`/`_solve_steady_state_widening_bracket` direkt aus
`scripts/study_plant_validation.py`, keine zweite Anlagenparametrierung), JSON-Ausgabe
`artifacts/runs/run_process_pinch_sensitivity.json`. Ruff und die vollstaendige Testsuite
vor jedem Commit gruen.

**Nachtrag RA3 -- Geometrie-Hypothese geprueft und AUSGESCHLOSSEN, die Fluidisierungsgrenze
ist der Engpass.** Naheliegender erster Verdacht bei einer 3-von-27-Ausbeute: eine nicht
mitskalierte Laborgeometrie. Geprueft und verworfen -- die Anlagengeometrie IST skaliert,
das Problem liegt an einer anderen, bereits in QA2 benannten Groesse.

a) **Werte mit Quelle.** Querschnittsflaeche `4.0875 m²`: nicht gemessen, sondern aus dem
Bettdruckverlust abgeleitet (`bed_mass_solid_setpoint_kg * 9.81 / 1920 Pa`,
`scripts/study_plant_validation.py:361`, `_BED_PRESSURE_DROP_PA=1920.0 Pa` = 19.2 mbar,
Leitsystem-Screenshot PDIRAH-320404, `bed_mass=800 kg`, QA1-Anker). Partikeldichte
`1500 kg/m³` (`MaterialConfig.particle_density_kg_per_m3`,
`src/twinlink_ml/simulation/config.py:85`) und Keimdurchmesser `150 µm`
(`nucleus_diameter_m`, `config.py:119`) sind unveraendert uebernommene Laborplatzhalter
(NA5-`ASSUMPTIONS`, Befund 56: "keine Anlagenmessung vorhanden, NICHT angepasst").
Anfangs-d50 `400 µm` (`initial_diameter_m`, `config.py:121`) ist der SAAT-Wert -- nicht
identisch mit dem `d50≈750 µm` aus QA2/RA3: dieser ist ein POPULATIONSBILANZ-ERGEBNIS im
Beharrungszustand (Kornwachstum durch Schichtaufbau), keine Eingangsgroesse, und variiert
je Betriebspunkt (siehe `terminal_velocity_m_per_s` je Punkt unten, 3.43-3.68 m/s, statt
eines fixen Werts).

b) **Geschwindigkeit vs. Fluidisierungsfenster.** Leerrohrgeschwindigkeit am ACE-Ankerpunkt
(33003 Nm³/h, `4.0875 m²`, `T_bett=67.3 °C`): **≈2.83 m/s** (Diagnoseskript gegen
`_gas_velocity_m_per_s`/`fluidization.py`, nicht Teil des Repos) -- Groessenordnung
bestaetigt, keine fehlende Skalierung. Am unteren Gittergeschwindigkeitsniveau
(`2.3123 m/s`, `dry_air_mass_flow≈9.6-10.6 kg/s` je nach Punkt) ist `u_gas` **unterhalb**
`u_t(d50)` (3.43-3.68 m/s je Punkt) -- fluidisierungsseitig zulaessig, das ist auch das
einzige Geschwindigkeitsniveau, das ueberhaupt zulaessige Punkte liefert (siehe Tabelle
unten). Beim mittleren Niveau (`4.6246 m/s`) liegt `u_gas` bereits **0.96-1.19 m/s ueber**
`u_t`, beim oberen (`6.9368 m/s`) **3.25-3.51 m/s ueber** `u_t` -- weil das
Stellgroessengitter `u_gas` von Stufe zu Stufe VERDOPPELT (2.31→4.62→6.94 m/s), waehrend
`u_t` selbst nur schwach mit dem Betriebspunkt schwankt (3.43-3.68 m/s, siehe a): Zwei der
drei Gitterstufen reissen die Fluidisierungsobergrenze unabhaengig vom genauen
Betriebspunkt -- das Gitter selbst ist grob genug, um das zu erzwingen.

c) **Pro-Punkt-Aufschluesselung (alle 27 Gitterpunkte).** Ursachen der 24 gescheiterten
Punkte: 8 kein Beharrungszustand gefunden (Solver erreicht die erweiterte
Bracket-Untergrenze), 13 ausschliesslich Fluidisierungsobergrenze verletzt, 2
Fluidisierungsobergrenze UND Gutsfeuchte gleichzeitig, 1 ausschliesslich Gutsfeuchte
(Fluidisierung dort erfuellt). Gutsfeuchte-Verletzungen liegen bei 0.36-0.73 kg/kg gegen
die plausible Obergrenze 0.02 kg/kg (Faktor 18-37, derselbe unbelegte Platzhalter wie in
QA2 benannt).

| T_zuluft °C | u_gas m/s | Spruehrate kg/h | Status | Ursache / Marge |
|---|---|---|---|---|
| 37.5 | 2.31 | 750  | infeasible | kein Beharrungszustand (Bracket-Untergrenze 79.9 °C) |
| 37.5 | 2.31 | 1500 | infeasible | kein Beharrungszustand |
| 37.5 | 2.31 | 2250 | infeasible | kein Beharrungszustand |
| 37.5 | 4.62 | 750  | infeasible | u_gas-u_t = -1.19 m/s (u_t=3.43) |
| 37.5 | 4.62 | 1500 | infeasible | kein Beharrungszustand |
| 37.5 | 4.62 | 2250 | infeasible | kein Beharrungszustand |
| 37.5 | 6.94 | 750  | infeasible | u_gas-u_t = -3.49 m/s (u_t=3.45) |
| 37.5 | 6.94 | 1500 | infeasible | u_gas-u_t = -3.51 m/s (u_t=3.43) + Gutsfeuchte 0.472 kg/kg |
| 37.5 | 6.94 | 2250 | infeasible | kein Beharrungszustand |
| 75.0 | 2.31 | 750  | **zulaessig** | Pinch Q1-Taupunkt, 454.84 kg/h |
| 75.0 | 2.31 | 1500 | infeasible | kein Beharrungszustand |
| 75.0 | 2.31 | 2250 | infeasible | kein Beharrungszustand |
| 75.0 | 4.62 | 750  | infeasible | u_gas-u_t = -1.07 m/s (u_t=3.55) |
| 75.0 | 4.62 | 1500 | infeasible | u_gas-u_t = -1.12 m/s (u_t=3.50) |
| 75.0 | 4.62 | 2250 | infeasible | u_gas-u_t = -1.15 m/s (u_t=3.47) + Gutsfeuchte 0.364 kg/kg |
| 75.0 | 6.94 | 750  | infeasible | u_gas-u_t = -3.36 m/s (u_t=3.57) |
| 75.0 | 6.94 | 1500 | infeasible | u_gas-u_t = -3.40 m/s (u_t=3.54) |
| 75.0 | 6.94 | 2250 | infeasible | u_gas-u_t = -3.44 m/s (u_t=3.50) |
| 112.5 | 2.31 | 750  | **zulaessig** | Pinch Q2-Eintritt, 438.00 kg/h |
| 112.5 | 2.31 | 1500 | **zulaessig** | Pinch Q1-Taupunkt, 454.84 kg/h |
| 112.5 | 2.31 | 2250 | infeasible | nur Gutsfeuchte 0.735 kg/kg (Fluidisierungsmarge +1.19 m/s) |
| 112.5 | 4.62 | 750  | infeasible | u_gas-u_t = -0.96 m/s (u_t=3.66) |
| 112.5 | 4.62 | 1500 | infeasible | u_gas-u_t = -1.02 m/s (u_t=3.61) |
| 112.5 | 4.62 | 2250 | infeasible | u_gas-u_t = -1.07 m/s (u_t=3.56) |
| 112.5 | 6.94 | 750  | infeasible | u_gas-u_t = -3.25 m/s (u_t=3.68) |
| 112.5 | 6.94 | 1500 | infeasible | u_gas-u_t = -3.29 m/s (u_t=3.64) |
| 112.5 | 6.94 | 2250 | infeasible | u_gas-u_t = -3.33 m/s (u_t=3.61) |

Quelle: `artifacts/runs/run_process_pinch_sensitivity.json` (`results[]`,
`infeasibility_reasons`/`solver_error`/`fluidization_margin_upper_m_per_s`).

**Antwort auf die Kernfrage.** Die 24 Ausfaelle sind keine Parametrierungsluecke (keine
nicht mitskalierte Geometrie) -- die Anlagenflaeche ist aus einer echten
Anlagenmessgroesse (Bettdruckverlust) abgeleitet und plausibel. Sie scheitern an einer
ECHTEN Betriebsgrenze, die aber mit einem unkalibrierten Laborplatzhalter berechnet wird:
`d50≈750 µm` (Populationsbilanz-Ergebnis, nicht die Geometrie) bestimmt `u_t`, und zwei
von drei Gitterstufen der Gasgeschwindigkeit ueberschreiten diese Grenze strukturell. Ob
die reale Anlage mit groesseren/dichteren Granulaten (Befund 56) ein anderes, breiteres
Fluidisierungsfenster haette, ist damit weiterhin offen -- das ist eine Frage an
`critical_moisture_kg_per_kg`/`particle_density_kg_per_m3`/Kornwachstumsparameter, nicht
an die Geometrie, und **nicht Teil dieser Pruefung** (keine Aenderung am
Anlagenparametersatz, wie beauftragt).

**Nachtrag RA2 -- T_kalt-Sweep im unteren Pinch-Regime.** Bisher (`run_pinch_benchmark.py`,
PA2-Datenfall) nur fuer das OBERE Regime belegt: der Pinch sitzt dort in jeder getesteten
`T_kalt`-Kombination am Eintritt von Quelle 2, `T_kalt` aendert den minimalen Heizbedarf
nicht. Offene Frage: gilt das auch im UNTEREN Regime (Pinch am Taupunkt von Quelle 1,
Quelle 2 fuer die obere Temperaturregion entkoppelt)? Beide zulaessigen unteren-Regime-
Punkte aus QA3 verwendet (nicht nur einer):

| Punkt | T_kalt=15°C | T_kalt=25°C | T_kalt=29,1°C | T_kalt=35°C |
|---|---|---|---|---|
| T=75,0°C/v=2,31/S=750 (T_out=40,94°C) -- min. Heizbedarf | 4548,378 kW | 4548,378 kW | 4548,378 kW | 4548,378 kW |
| " -- min. Kuehlbedarf | 3532,039 kW | 2526,518 kW | 2103,776 kW | 1372,758 kW |
| T=112,5°C/v=2,31/S=1500 (T_out=43,74°C) -- min. Heizbedarf | 4548,378 kW | 4548,378 kW | 4548,378 kW | 4548,378 kW |
| " -- min. Kuehlbedarf | 3924,420 kW | 2855,468 kW | 2286,100 kW | 1404,946 kW |

Pinch-Temperatur (heiss) bleibt an beiden Punkten ueber den gesamten Sweep unveraendert
bei 44,94 °C (Q1-Kondensationskurve). **Antwort auf die Kernfrage: NEIN, `T_kalt` aendert
den minimalen Heizbedarf auch im unteren Regime nicht** -- nur der minimale Kuehlbedarf
faellt mit steigendem `T_kalt` (erwartbar: weniger Enthalpie wird den Quellen unterhalb
des Pinch entzogen). Grund: der minimale Heizbedarf ist beim Problem-Table-Algorithmus
durch die Kaskadenbilanz OBERHALB des Pinch bestimmt; `T_kalt` setzt nur die untere
Abkuehlgrenze der Quellensegmente UNTERHALB davon. Solange die Pinch-Lage selbst
unveraendert bleibt (hier durchgehend bestaetigt), bleibt die Kaskade oberhalb -- und
damit der Heizbedarf -- von `T_kalt` unberuehrt. Die Einschraenkung aus dem Auftrag
("Aussage nur fuer oberes Regime belegt") entfaellt damit; **keine Korrektur an Befund 57h
oder dem Pinch-Befundentwurf noetig**, die Aussage gilt regimeunabhaengig, soweit an
diesen zwei Punkten geprueft.

**Vorbehalt.** Nur zwei Punkte verfuegbar (RA3: das grobe QA2-Gitter liefert nicht mehr
zulaessige untere-Regime-Punkte) -- eine Stichprobe von zwei, keine Flaechendeckung. Beide
Punkte teilen dieselbe Pinch-Lage (Q1-Kondensationskurve, 44,94 °C); ein dritter Punkt mit
einer strukturell anderen unteren Pinch-Lage waere eine schaerfere Pruefung, aber mit dem
vorliegenden Gitter nicht verfuegbar.

**Reproduzierbarkeit.** Direkter Aufruf von `heatnetwork.pinch.evaluate_pinch_target` mit
Quelle 2 aus den bereits vorliegenden Beharrungszustandsergebnissen der beiden Punkte
(`artifacts/runs/run_process_pinch_sensitivity.json`, `results[]`) -- kein erneuter
Solverlauf noetig, da `T_kalt` nicht in den Prozess-Solver eingeht, nur in die
Waermeintegrationsrechnung. Diagnoseskript nicht Teil des Repos (reine Auswertung
bestehender, committeter Rohdaten).

### 62. Erste RL-Umgebung (Ein-Schritt-Bandit) reproduziert Befund 52 bitgleich -- Kosten je Schritt, Hochrechnung, bewusste Auslassungen

**Auf dem Zweig `rl-env` entwickelt (eigener Worktree), inzwischen per echtem Merge
zusammengefuehrt -- ein Fast-Forward war nicht moeglich, weil `master` sich
zwischenzeitlich bewegt hatte.** Auftrag:
Befund 52 (`scripts/study_operating_point_regret.py`,
`artifacts/runs/bb1_bb3_operating_point_regret.json`) als Gymnasium-Umgebung
(kontextueller Bandit, eine Episode = ein Schritt) nachbilden -- keine neue Physik, keine
neue Zielfunktion. Code: `src/twinlink_ml/rl/` (`env.py`, `evaluation.py`), Tests
`tests/test_operating_point_bandit_env.py`, `tests/test_bandit_env_evaluation.py`, Skript
`scripts/probe_bandit_env_sac_smoke.py`.

**a) Nebenbedingungsbehandlung (Auftrag VA1) -- Straffunktion, keine Projektion, und ein
Befund, der ueber die Implementierungsentscheidung hinausgeht.** `check_constraints`/
`normalized_violation` aus Befund 52 werden unveraendert per `importlib` wiederverwendet,
fuer alle drei Nebenbedingungen (Gutsfeuchte, Fluidisierungsfenster, Bettemperatur <=
80 °C), als Straffunktion in der Belohnung. Keine Projektion fuer das
Fluidisierungsfenster, obwohl das Muster dafuer existiert
(`GreyBoxFluidBedGranulator.project_action`) -- Grund ist kein Implementierungsdetail,
sondern ein Befund: in diesem Environment bindet das Fluidisierungsfenster STRUKTURELL
NIE. Der Beharrungsdurchmesser d50 (`steady_state.py`, Schritt 2) haengt nur von
`material.nucleus_diameter_m`/`material.nucleation_solid_fraction` ab -- Spruehrate und
Suspensionsfeststoffanteil kuerzen sich in der Herleitung heraus, und beide sind in diesem
Environment ohnehin fest (Spruehrate am Nominalwert, Suspensionsfeststoffanteil an seinem
Nominalwert, nicht Teil der zwei gelernten Aktionen). d50 ist damit ueber den GESAMTEN
Aktions- und Stoergroessenraum eine Konstante (750 µm bei den Platzhalterwerten). Eine
Sweep-Rechnung ueber Betttemperaturen 20-140 °C (weit ueber das plausible
Beharrungsfenster hinaus) und den vollen Stoergroessenbereich liefert worst-case
`u_mf` ≈ 0,236 m/s, worst-case `u_terminal` ≈ 3,42 m/s -- die `ActionBound` fuer
`gas_velocity_m_per_s` ([0,8; 2,5] m/s) liegt mit ≥ 0,56 m/s Marge unten und ≥ 0,92 m/s
Marge oben vollstaendig darin. AN BEFUND 52S OPTIMALPUNKTEN wirksam ist ausschliesslich die
Gutsfeuchte (Ergebnisgroesse, nicht projizierbar): bindet an 17 von 25 Befund-52-
Gitterpunkten, die Bettemperaturgrenze an 0/25.

**Korrektur (WA3, siehe unten): das gilt NICHT ueber die volle Aktionsbox.** Ein
systematischer 13×9-Gittersweep ueber `[60,120]°C × [0,8; 2,5] m/s` bei Nominalstoergroesse
zeigt: von 24/117 unzulaessigen Gitterpunkten sind 21 durch die 80-°C-Bettemperaturgrenze
getrieben, nur 3 durch die Gutsfeuchte. Befund 52s Optimierer muss nie in die heisse Ecke
der Box (hohe Zulufttemperatur), ein noch breit explorierender RL-Agent landet dort leicht
-- die Bettemperaturgrenze ist damit die praktisch dominante Sicherheitsschranke WAEHREND
des Trainings, obwohl sie am KONVERGIERTEN Optimum irrelevant ist. Beide Aussagen gelten
gleichzeitig, fuer verschiedene Teile des Aktionsraums.

MODELLEIGENSCHAFT, keine Aussage ueber die reale Anlage: das Vier-Zustands-Modell kennt
keine Agglomeration, keinen Bruch und keine Kopplung des Wachstums an die
Trocknungsbedingungen. An der realen Anlage und in einer echten Populationsbilanz (Dyssol)
haengt die Partikelgroesse sehr wohl von der Fahrweise ab; die zustandsabhaengige
Gasgeschwindigkeitsprojektion wird dort (und in diesem Modell in TRANSIENTEN, wenn d50 vom
stationaeren Wert abweicht -- Anfahrvorgaenge, Produktwechsel) wirksam. Im stationaeren
Fall hier ist sie ein fester Kasten. Dazu passt, ohne sich zu widersprechen: im
Anlagenmassstab (separate Untersuchung, RA3) scheiterten bei demselben d50-Platzhalter 15
Gitterpunkte an der Fluidisierungsobergrenze -- vermutlich hoehere Gasgeschwindigkeiten im
dortigen Anlagengitter oder andere Gaseigenschaften bei anderer Betttemperatur; nicht in
diesem Worktree nachgerechnet, hier nur zur Einordnung genannt.

`_PENALTY_WEIGHT_KW` (50 kW, Befund 52) wird unveraendert uebernommen -- fuer die VA2-
Bitgleichheit noetig, nie neu kalibriert. Als Hyperparameter hat es fuer einen RL-Agenten
eine andere Wirkung als fuer den Optimierer aus Befund 52: der Agent sitzt bei bindender
Gutsfeuchte direkt an der Grenze, statt sie nur als Suchheuristik zu durchqueren. Deshalb
werden Zulaessigkeitsrate und mittlere Feuchteueberschreitung unzulaessiger Aktionen als
eigene Groessen berichtet (VA3), nicht nur die Regret-Metrik.

**b) Orakel-Pruefung (Auftrag VA2) -- bitgleich, mit einem numerisch begruendeten
Praezisions-Fund.** Alle 25 Gitterpunkte: Orakelaktion -> Belohnung exakt `-energy_target`
(max. Abweichung 3,3·10⁻¹⁶ kW, Gleitkommarauschen), fester Betriebspunkt -> Regret-Median
16,03 %/P95 22,20 %/Max 25,80 %, exakt wie dokumentiert, Zulaessigkeit an allen 25 Punkten
fuer beide Aktionsmengen. Test dazu: `tests/test_operating_point_bandit_env.py`
(`test_va2a_...`, `test_va2b_...`, `test_va2c_...`).

Fund unterwegs: mit den Loeser-DEFAULTS (`solve_steady_state` ohne Argumente, 100
Bisektionsschritte/Toleranz 1e-13) scheitern 6 der 25 Gitterpunkte an einem
Bracket-Vorzeichenfehler in der aeusseren Temperatur-Bisektion -- exakt das in
`evaluate_point`s Docstring (`study_operating_point_regret.py`) beschriebene Phaenomen:
an der Bracket-Untergrenze (`T_amb + 10`) liegt die wahre Nullstelle knapp ausserhalb, und
ob der dort ausgewertete Residuenwert numerisches Rauschen mit positivem oder negativem
Vorzeichen zeigt, haengt von der Bisektionstiefe der INNEREN Feuchte-Bisektion an genau
diesem Punkt ab. Mit der SUCH-Praezision aus Befund 52 (`DEFAULT_SEARCH_MAX_ITERATIONS`/
`DEFAULT_SEARCH_TOLERANCE`, 30/1e-7) -- die Befund 52 auch fuer seine berichteten Punkte
verwendet, nicht nur fuer die Suche selbst -- loesen alle 25 Punkte bitgleich. Die Umgebung
verwendet deshalb durchgehend die Such-Praezision, nicht die Loeser-Defaults (`env.py`,
`evaluate_operating_point`).

Massstabsfrage ("Achtung Massstab", Auftrag VA2): Duesendruck ist in der Umgebung fest
(`FIXED_NOZZLE_PRESSURE_BAR` = 1,002634656115166 bar, Befund 52s fester Betriebspunkt), in
den Befund-52-Optima frei (Spanne 1,0002-1,0111 bar). Empirisch: die Abweichung ist NICHT
nur klein, sondern numerisch nicht von null unterscheidbar (max. 3,3·10⁻¹⁶ kW ueber alle
25 Punkte, `test_fixed_nozzle_pressure_scale_is_negligible`). Grund gefunden, nicht nur
gemessen: in `energy.power_breakdown_kw` clamped `max(pressure_ratio, 1 + 1e-12)` den
Kompressor-Druckverhaeltnis-Term, solange `p_nozzle <= gas.pressure_pa / 1e5 = 1,01325`
bar -- die gesamte Befund-52-Spanne liegt unterhalb dieser Schwelle, der Atomiser-Anteil an
`energy_target` ist dort exakt konstant (≈ 0). Die Zwei-Aktionen-Vereinfachung ist damit
nicht nur "innerhalb einer Toleranz", sondern in diesem Fenster strukturell exakt.

**Nachtrag (WA1, nur lesend, keine Code-Aenderung).** Der Clamp ist keine Nebenbeobachtung,
sondern das in `energy.power_breakdown_kw`s eigenem Docstring bereits benannte
Modellartefakt ("ein optimierender Agent faehrt [`p_nozzle`] sofort ans Minimum [...] sollte
`p_nozzle` im RL-Teil entweder fixiert oder mit einer Qualitaetsstrafe versehen werden") --
diese Umgebung setzt genau die dort empfohlene Fixierung um. Praezisiert: `nozzle_pressure_
bar` ist im Modell ABSOLUTDRUCK (Formelstruktur `ln(p_nozzle/p_amb)` ohne Offset gegen
`gas.pressure_pa`, das im gesamten `thermo.py` durchgehend als absoluter Gasphasendruck
verwendet wird -- kein `_rise`-Suffix wie bei `pump_pressure_rise_pa`, das explizit eine
Differenzgroesse traegt; keine explizite Absolut/Ueberdruck-Festlegung in
`docs/03_code_konventionen.md`). Befund 52s fester Betriebspunkt (1,002634656115166 bar)
liegt bei `ActionBound = [1,0; 4,0]` bar nur 0,26 % des Abstands zur Clamp-Schwelle
(1,01325 bar) oberhalb der UNTEREN Aktionsschranke -- der Optimierer sitzt praktisch am
Kastenrand, in der Zone, in der Zerstaeuberluft nichts kostet.

Vergleich zu Anlage F-5 (Anwendungspartner, siehe Befund 56 fuer den bisherigen
Validierungsstand -- Zahlen hier als Vergleichsgroesse aus dem Auftrag, keine Rohdaten):
gemessene Duesendruecke 1,42-1,81 bar. Ob das Ueberdruck ist (in der Anlagenmesstechnik fuer
Zerstaeuberluft/Druckluft die verbreitete Konvention -- entspraeche dann ≈2,43-2,81 bar
absolut, weit oberhalb der Clamp-Schwelle, also in einem Bereich, in dem die
Zerstaeuberleistung im Modell tatsaechlich variiert) oder Absolutdruck (dann nahe am
Modell-Fixpunkt), ist NICHT ermittelt. Offene Frage, nicht Ergebnis: ob Befund 52s
Optimum an der unteren Aktionsschranke einen physikalisch zulaessigen Betriebspunkt
beschreibt oder ein Artefakt des Clamps ist, der die reale Zerstaeubungsguete nicht
kennt und ihr Fehlen nicht bestraft.

**c) Bewertungsmetrik (Auftrag VA3) -- implementiert, mit einer bekannten Pathologie
bei kleinem Nenner.** `twinlink_ml.rl.evaluation.evaluate_policy` berechnet den Anteil
geschlossenen Regrets (`(E_fest - E_Agent) / (E_fest - E_Orakel)`) an den 25 Befund-52-
Punkten, plus Zulaessigkeitsrate und mittlere Feuchteueberschreitung unzulaessiger
Aktionen. Zwei Grenzfaelle validieren die Formel selbst: eine Policy, die immer den festen
Betriebspunkt liefert, schliesst ueberall exakt 0 % Regret; eine Policy, die immer die
Orakelaktion liefert, schliesst ueberall exakt 100 % (`tests/test_bandit_env_evaluation.py`).
Disjunktheit von Trainings- und Testmenge: bei stetiger Gleichverteilung (Standard-`reset()`)
Mass-Null-Ereignis, zusaetzlich als Laufzeitpruefung ausgefuehrt
(`assert_disjoint_from_test_grid`, im Rauchtest an 200 Ziehungen gepruefte, keine Kollision).

Waehrend des Rauchtests aufgefallen: der Nenner `E_fest - E_Orakel` ist an manchen
Gitterpunkten sehr klein (der Regret selbst schwankt zwischen 16 % und 26 % relativ, aber
absolut zwischen 0,17 und 0,24 kW) -- bei einer schlechten Policy kann der geschlossene
Anteil dadurch stark negativ ausreissen (im Rauchtest: Minimum bei einer un-trainierten
Policy im vierstelligen negativen Bereich). Median/P95 sind robuster als das Minimum;
letzteres ist als Diagnosegroesse zu lesen, nicht als Kernkennzahl.

**d) Rauchtest (Auftrag VA4) -- Kette steht, Kosten gemessen, kein Trainingsergebnis.**
`scripts/probe_bandit_env_sac_smoke.py`, SAC ueber `stable-baselines3` (neu als optionales
Extra `rl` in `pyproject.toml`: `gymnasium>=0.29`, `stable-baselines3>=2.3`,
`torch>=2.4` -- installiert ohne bestehende Pakete anzufassen, siehe Provenienz unten),
`total_timesteps=64`, ein Seed. Reine `env.step()`-Kosten (30 Messungen, MKL_NUM_THREADS=
OMP_NUM_THREADS=1, Such-Praezision): Median 164 ms. Hochrechnung: 10⁴ Schritte ≈ 0,46 h,
10⁵ Schritte ≈ 4,6 h, einstraengig. SB3-Overhead ist klein gegen die Loeserkosten (168 ms/
Schritt inkl. SB3 vs. 164 ms reine Loeserzeit). Damit ist `solve_steady_state` als
Uebergang fuer ein ernsthaftes Training (10⁴-10⁵ Schritte) auf EINEM Kern nicht praktikabel
(Groessenordnung Stunden) -- mit Parallelisierung ueber mehrere Umgebungsinstanzen
(SB3-VecEnv, mehrere Kerne, analog zum Gitterpunkte-Pool in Befund 52) sinkt die Wandzeit
proportional zur Kernzahl, aendert aber nichts an den reinen Solver-Kosten pro Schritt; ob
das ausreicht oder ein schnelleres Ersatzmodell (Surrogat, siehe `simulation/base.py`s
geplante `SurrogateSimulator`-Implementierung) noetig wird, ist eine Entscheidung ueber das
verfuegbare Kernbudget des tatsaechlichen Trainingslaufs, hier nicht getroffen. Die
VA3-Metrik wurde am Ende einmal berechnet (Zulaessigkeitsrate 1,0, stark negativer
geschlossener Regret-Anteil) -- das ist eine Pipeline-Pruefung, KEIN Trainingsergebnis
(Auftrag VA4: ob der Agent im Rauchtest schon lernt, ist nicht die Frage). Artefakt:
`artifacts/runs/probe_bandit_env_sac_smoke.json`.

**e) Bewusst ausgelassene Erweiterungen (Auftrag), mit Blocker:**

- **Speicher.** Kein Zustand ueber Episoden hinweg -- der Beharrungszustands-Loeser
  rechnet je Schritt einen neuen, unabhaengigen Fixpunkt, keine Speicherfuellstands-
  Dynamik. Blocker: erfordert eine mehrschrittige (nicht Bandit-)Formulierung mit
  Speicherzustand als zusaetzlicher Zustandsgroesse -- ausserhalb dieses Auftrags.
- **Mehrere Linien.** Eine Umgebungsinstanz = eine Anlage/ein `ProcessConfig`. Blocker:
  braucht eine Multi-Agent- oder eine gemeinsame Ressourcen-Formulierung (z. B. geteilte
  Dampf-/Stromkapazitaet ueber Linien), die hier nicht spezifiziert ist.
- **Netto-Zielfunktion.** Belohnung ist `-energy_target` (Bruttoleistung, `energy.target
  = power_kw`), nicht die MA4-Netto-Variante (`study_operating_point_regret_net.py`,
  `q_rec_kw`/`recovery_credit_kw`). Blocker: keiner technischer Art -- der Override-
  Mechanismus existiert bereits (`_OBJECTIVE_VALUE_FN`) und liesse sich einhaengen; nicht
  eingebaut, weil der Auftrag explizit "keine neue Zielfunktion" verlangt und die
  Netto-Frage laut Modul-Docstring von `study_operating_point_regret_net.py` fuer AP 1
  ohnehin offen ist.
- **Anlagenmassstab.** Environment rechnet am Laborplatzhalter (`conf/process/
  greybox_gab_placeholder.yaml`), nicht am Anlagenmassstab. Blocker: der Anlagenmassstab
  lebt in einer separaten Untersuchung (RA3, siehe (a)) mit abweichenden Fluidisierungs-
  Randbedingungen -- eine RL-Umgebung dafuer braeuchte deren Konfiguration und muesste die
  in (a) gefundene Slack-Eigenschaft des Fluidisierungsfensters neu pruefen, da sie dort
  nicht gilt.

**f) WA2 -- solve_steady_state-Falle abgesichert.** ``max_iterations``/``tolerance`` waren
bislang undokumentierte Parameter; der Docstring vermerkt jetzt explizit, dass die
Voreinstellung (100/1e-13) Befund 52 NICHT bitgleich reproduziert (siehe (b)). Neuer
Regressionstest (`tests/test_steady_state.py::test_solve_steady_state_defaults_do_not_
reproduce_befund_52`) belegt den Kontrast an Gitterpunkt 15 exekutierbar: Loeser-Defaults
werfen `RuntimeError`, Such-Praezision reproduziert `energy_target` bitgleich. Nur
Docstring-Ergaenzung und Test, keine Verhaltensaenderung -- `OperatingPointBanditEnv` hing
bereits vorher nicht von den Defaults ab (uebergibt die Praezision explizit, siehe (b)).

**g) WA3 -- erste RL-Lernkurve (SAC, 10⁴ Schritte, ein Seed) -- widerlegt die vorab
festgelegte Erwartung, mit gefundener Ursache.** `scripts/run_rl_bandit_learning_curve.py`,
Hyperparameter durchgehend SB3-Default (keine Abstimmung), Zwischenauswertung bei
1000/2500/5000/10000 Schritten auf den 25 Befund-52-Testpunkten. Artefakt:
`artifacts/runs/rl_bandit_sac_learning_curve.jsonl` (ein `RunRecord` je Checkpoint,
dasselbe JSONL-Format wie die Architektur-Lernkurven).

| Schritte | Zulaessigkeitsrate | geschl. Regret Median | P95 | Min |
|---|---|---|---|---|
| 1000 | 0,00 | -19,94 | -15,19 | -78786 |
| 2500 | 0,00 | -19,98 | -15,22 | -79000 |
| 5000 | 0,40 | -16,89 | -8,73 | -79116 |
| 10000 | 0,56 | -13,41 | -8,62 | -78480 |

**Kernfrage (Auftrag, ein Satz): NEIN.** Nach 10⁴ Schritten schliesst der Agent keinen
positiven Anteil des Regrets (Median durchgehend tief negativ -- schlechter als der feste
Betriebspunkt) und erreicht nur 56 % Zulaessigkeit. Das widerspricht der vorab
festgelegten Erwartung ("bei zweidimensionalem Kontext/Aktion sollte ein Grossteil des
Regrets schliessbar sein") deutlich, nicht nur graduell.

Ursache gefunden, nicht nur vermutet -- und eine KORREKTUR von (a)/(e) oben: die dortige
Aussage "wirksam ist ausschliesslich die Gutsfeuchte" galt nur AN BEFUND 52S
OPTIMALPUNKTEN, nicht ueber die volle Aktionsbox, die ein Agent waehrend des Trainings
explorieren muss. Indiz: `mean_moisture_violation_of_infeasible_kg_per_kg` steht bei allen
vier Checkpoints exakt auf 0,0 -- obwohl bei Schritt 1000/2500 ALLE 25 Testpunkte
unzulaessig sind. Stichprobe und systematischer 13×9-Gittersweep ueber die volle Aktionsbox
(Nominalstoergroesse) bestaetigen: von 24/117 unzulaessigen Gitterpunkten sind 21 durch die
80-°C-Bettemperaturgrenze getrieben, nur 3 durch die Gutsfeuchte. Beispiel:
`t_in=108,8 °C, u_gas=2,352 m/s` -> `bed_temperature=86,97 °C` (7 K ueber der Grenze),
Feuchte dabei unauffaellig (0,0124 kg/kg, weit unter kritisch). Befund 52s Optimierer muss
nie in die heisse Ecke der Box (hohe Zulufttemperatur nahe 120 °C), ein noch breit
explorierender RL-Agent landet dort leicht. Die Bettemperaturgrenze -- ein laut Befund 46
unbelegter Platzhalter -- ist damit die praktisch dominante Sicherheitsschranke WAEHREND
des Trainings, obwohl sie am konvergierten Optimum irrelevant ist (0/25 bindend dort).

Nicht weiter diagnostiziert in dieser Session (kein Modell-Checkpoint gespeichert, kein
Re-Training): OB die Zulaessigkeitsverbesserung (0,00 -> 0,56) allein aus dem Wegdriften von
der 80-°C-Grenze kommt oder ob spaeter auch die Gutsfeuchte relevant wuerde, sobald der
Agent naeher an ein Energieoptimum heranrueckt (dort bindet sie an 17/25 Punkten, siehe
oben). Der nahezu konstante `closed_min`-Ausreisser (~-79000 ueber alle vier Checkpoints)
ist vermutlich derselbe kleine-Nenner-Effekt, der in VA3 (c) bereits dokumentiert ist,
nicht weiter isoliert. KEINE Seeds-Sweeps, kein 10^5-Lauf in dieser Session (Auftrag).

**h) XA1 -- VA3-Metrik korrigiert.** Die urspruengliche Metrik (Median/P95/Min ueber
PER-PUNKT-Quotienten aller 25 Punkte) mischte Zulaessigkeit und Regret in einer Zahl und
war anfaellig fuer kleine Nenner (WA3s Minimum -78786). Neue Berichtsform, IMMER
zusammen gelesen (`twinlink_ml.rl.evaluation`):

a) `feasibility_rate` (unveraendert).
b) `pooled_closed_regret_fraction` -- GEPOOLT (Summe im Zaehler/Nenner ueber alle
   zulaessigen Punkte, kein Einzelquotient), nur ueber zulaessige Aktionen, rohe Energie
   ohne Strafterm. Eine Summe ueber 25 Punkte ist gegen einen einzelnen kleinen Nenner
   robust, ein Einzelquotient nicht.
c) `infeasible_violations` -- je unzulaessiger/loeserfehlgeschlagener Aktion, welche
   Nebenbedingung (Feuchte, Bettemperatur, Fluidisierungsfenster, Loeserfehler) um wie viel.

Begruendung, warum (b) nie ohne (a): eine Politik, die nur an zwei leichten Punkten
zulaessig ist und dort zufaellig nah am Orakel liegt, zeigt ein perfektes (b), obwohl sie
an 23 von 25 Punkten unbrauchbar ist -- (a) macht diesen Fall sichtbar. Median/P95/Min
bleiben als nennerempfindliche Zusatzgroesse erhalten, nicht mehr primaer. Getestet: die
VA3-Grenzfaelle (fest -> (b)=0, Orakel -> (b)=1) gelten weiter; ein Testfall mit genau
einer unzulaessigen Aktion (konstruiert: hohe Zulufttemperatur verletzt die
80-°C-Bettemperaturgrenze) laesst (b) bitgleich, aendert nur (a) und (c) --
`tests/test_bandit_env_evaluation.py::test_single_infeasible_action_does_not_affect_
pooled_fraction`.

**i) XA2 -- Checkpoints ab jetzt gespeichert.** Der WA3-Lauf liess sich nicht weiter
diagnostizieren, weil keine Politik gespeichert war. Ab diesem Lauf sichert
`scripts/run_rl_bandit_learning_curve.py` an jedem Auswertungspunkt die Politik
(SB3-natives `model.save()`, `.zip`) unter
`artifacts/checkpoints/rl_bandit_sac_learning_curve/` -- per `.gitignore`
(`artifacts/checkpoints/`, bereits vorhanden fuer Torch-Checkpoints,
`utils/checkpoints.py`) ausgeschlossen, nicht committet. Pfad im RunRecord unter
`checkpoint/path`, analog zur bestehenden Provenienz-Konvention.

**j) XA4 -- Einordnung der 80-°C-Bettemperaturgrenze, KEINE Aenderung.** Die Grenze bleibt
unveraendert -- sie zu entfernen, damit RL besser aussieht, waere das Falsche. Zur
Einordnung:

- Sie ist ein Platzhalter ohne Quelle (Befund 46, `BED_TEMPERATURE_LIMIT_C = 80.0` in
  `study_operating_point_regret.py`, frei gesetzt).
- Sie bindet an KEINEM Befund-52-Optimum (0/25), verursacht aber 21 von 24 unzulaessigen
  Punkten im vollen Aktionsraum (siehe (g)/WA3) -- an den Optima irrelevant, im
  Explorationsraum dominant.
- Allgemeines Muster, nicht spezifisch fuer diese eine Grenze: ein Optimierer (Befund 52s
  `differential_evolution`+Nelder-Mead) sieht im Wesentlichen nur die Nebenbedingungen NAHE
  DEM Optimum -- er konvergiert dorthin und die uebrige Box bleibt unbesucht. Ein
  explorierender RL-Agent sieht die GESAMTE Box, insbesondere frueh im Training. Ein
  Platzhalter, der fuer die Optimierung harmlos ist (weil das Optimum ihn nie beruehrt),
  wird fuer RL wirksam, sobald die Exploration ihn erreicht. Diese Diskrepanz ist bei jedem
  weiteren unbelegten Platzhalter in `ProcessConfig` zu erwarten, nicht nur bei dieser
  einen Grenze.
- An Anlage F-5 (Anwendungspartner) tragen die Betttemperatur-Messstellen das Kuerzel SAH
  (Alarm hoch, ISA-Notation) -- ein realer, produktspezifischer Grenzwert existiert also
  und ist beim Anwendungspartner erfragbar. Keine Rohdaten hier, nur die Tag-Konvention als
  Beleg, dass die Groesse an der Anlage ueberwacht wird, siehe Befund 56 fuer den
  bisherigen Validierungsstand an F-5.

**k) XA3 -- WA3 wiederholt, gleiche Einstellungen, korrigierte Auswertung.** SAC, 10^4
Schritte, Seed 0, SB3-Default-Hyperparameter -- identisch zu WA3. Artefakt:
`artifacts/runs/rl_bandit_sac_learning_curve_xa3.jsonl`, Checkpoints unter
`artifacts/checkpoints/rl_bandit_sac_learning_curve/` (nicht committet, siehe (i)).

| Schritte | (a) Zulaessigkeit | (b) gepoolt, nur zulaessig | (b) Innere (n=2) | (b) Rand (n=23) | (c) Bett-Verl. |
|---|---|---|---|---|---|
| 1000 | 0,00 | n. def. (0 zulaessig) | n. def. | n. def. | 25/25 |
| 2500 | 0,00 | n. def. (0 zulaessig) | n. def. | n. def. | 25/25 |
| 5000 | 0,40 | -10,59 | n. def. (0/2 zulaessig) | -10,59 (10/23 zulaessig) | 15/25 |
| 10000 | 0,56 | -10,83 | -11,83 (1/2 zulaessig) | -10,74 (13/23 zulaessig) | 11/25 |

`(c)`-Spalte: ALLE Verletzungen ueber alle vier Checkpoints sind Bettemperatur (25, 25, 15,
11 von 25) -- Feuchte 0, Fluidisierungsfenster 0, Loeserfehler 0, an JEDEM Checkpoint. Das
bestaetigt (g)/WA3s Befund mit exakten Zaehlungen aus der tatsaechlich trainierten Politik,
nicht nur aus einem generischen Gittersweep.

**Kernfrage (Auftrag, ein Satz): BEIDES, aber der Agentenanteil ist der groessere.** Die
Zulaessigkeit (a) ist unveraendert (0,00/0,00/0,40/0,56 -- identisch zu WA3, weil (a) von
der Metrikkorrektur nicht betroffen ist) und bleibt ein reines Agentenversagen: er verlaesst
die 80-°C-Grenze auch nach 10^4 Schritten in 44 % der Faelle. Die korrigierte, NUR ueber
zulaessige Aktionen gepoolte Regret-Metrik (b) bleibt trotz Korrektur deutlich negativ
(-10,6 bei 5000, -10,8 bei 10000 Schritten) -- also NICHT nahe 0. Das zeigt: WA3s
Extremwerte (Median bis -20, Minimum -78786) waren GROSSTENTEILS Metrikartefakt (Pooling
unzulaessiger/kleiner-Nenner-Punkte), aber selbst bereinigt bleibt ein substanzieller,
NICHT wegerklaerbarer Agentenanteil -- unter den zulaessigen Aktionen ist die rohe
Energie regelmaessig ums Zehnfache schlechter als die Orakel-Fest-Differenz vorgibt.

Die XA3-Erwartung (Rand- vs. Innenpunkte trennen, weil ein Anteil von 1.0 an
Randoptima fuer eine tanh-Politik strukturell nicht erreichbar ist) traegt HIER NICHT als
Erklaerung: Innen- und Randpunkte liegen bei 10000 Schritten nah beieinander (-11,83 vs.
-10,74) -- die schlechte Performance ist nicht auf die Randpunkte konzentriert, sondern
gleichmaessig ueber beide Gruppen verteilt. Die tanh-Asymptotik ist als grundsaetzliche
Einschraenkung weiterhin korrekt (siehe Auftrag), erklaert aber NICHT den beobachteten
Rueckstand bei diesem Trainingsstand.

Bei Schritt 1000/2500 ist (b) an KEINEM der beiden Teilmengen definiert (0 zulaessige
Punkte insgesamt) -- das ist keine fehlende Messung, sondern Folge der Definition ("nur
ueber zulaessige Aktionen"): wenn nichts zulaessig ist, gibt es nichts zu pooligen. `n.
def.` ist hier keine 0 und kein Fehlschlag, sondern unbestimmt.

**l) YA1 -- Zufallsbaseline: die erwartete Lesart trifft NICHT zu, der Agent liegt auf
BEIDEN Achsen unter Zufall.** `scripts/probe_bandit_env_random_baseline.py`, 20
Wiederholungen (Seeds 0-19), gleichverteilte Aktionen in `[-1, 1]^2`, an denselben 25
XA3-Testpunkten, XA1-korrigierte Metrik. Artefakt:
`artifacts/runs/rl_bandit_random_baseline.jsonl`. Reine Auswertung, kein Training.

| | (a) Zulaessigkeit | (b) gepoolt, nur zulaessig |
|---|---|---|
| Zufall (Median [Min, Max], n=20) | 0,80 [0,68; 0,96] | -6,55 [-8,96; -5,08] |
| XA3, 1000 Schritte | 0,00 | n. def. (0 zulaessig) |
| XA3, 2500 Schritte | 0,00 | n. def. (0 zulaessig) |
| XA3, 5000 Schritte | 0,40 | -10,59 |
| XA3, 10000 Schritte | 0,56 | -10,83 |

**Kernfrage, getrennt beantwortet:**

- **Zulaessigkeit: der Agent liegt nach 10^4 Schritten UNTER dem Zufallsniveau**, nicht
  darueber (0,56 gegen Zufall-Median 0,80, sogar unter dem Zufall-MINIMUM 0,68 ueber alle
  20 Wiederholungen). Eine gleichverteilte Aktion respektiert die 80-°C-Grenze haeufiger
  als die trainierte Politik.
- **Anteil ueber zulaessige Aktionen: der Agent liegt ebenfalls UNTER dem Zufallsniveau**
  (-10,83 gegen Zufall-Median -6,55), nicht "etwa gleichauf" wie erwartet.

Die im Auftrag genannte erwartete Lesart ("Agent ueber Zufall bei Zulaessigkeit, auf
Zufallsniveau bei Energie -- er haette gelernt, der Strafe auszuweichen, aber noch nichts
ueber die Energie") TRIFFT NICHT ZU. Der tatsaechliche Befund ist staerker: nach 10^4
Schritten hat der Agent auf KEINER der beiden Achsen einen messbaren Vorteil gegenueber
reinem Zufall erarbeitet -- eher im Gegenteil. Das ist ein staerkeres negatives Ergebnis
als WA3/XA3 allein nahelegten (dort fehlte der Referenzpunkt "was leistet gar keine
gelernte Politik"). Ursache nicht in diesem Abschnitt geklaert -- siehe YA2 fuer die
Belohnungsskala als wahrscheinlichen Erklaerungsansatz.

Nebenbefund: `n_violations_solve_failed` bei Zufallsaktionen median 1, max 3 von 25 --
gleichverteilte Aktionen ueber die GESAMTE Box treffen gelegentlich Kombinationen, an denen
`solve_steady_state` selbst an einem Bracket-Vorzeichenfehler scheitert (die Such-Praezision
aus WA2 verhindert das nicht vollstaendig, sie macht es nur seltener als mit den
Loeser-Defaults). Bei XA3s trainierter Politik trat das ueber alle vier Checkpoints kein
einziges Mal auf -- die Politik bleibt, obwohl sie oft unzulaessig ist, in einem numerisch
loesbaren Bereich, den reiner Zufall gelegentlich verlaesst.

**m) YA2 -- Belohnungsskala beziffert: Strafe uebertrifft das Energiesignal um Faktor
~500.** `scripts/probe_bandit_env_reward_scale.py`, reine Messung, keine Aenderung.
Normierung wortgleich aus `study_operating_point_regret.py::check_constraints`:

```python
normalized_violation = (
    moisture_violation / max(critical_moisture_kg_per_kg, 1e-6)
    + gas_lower_violation / window
    + gas_upper_violation / window
    + bed_temperature_violation / 10.0
    + (0.0 if solution.feasible else 1.0)
)
```

Fuer eine REINE Bettemperaturverletzung (laut XA3 ALLE 76 beobachteten Verletzungen ueber
alle vier Checkpoints) reduziert sich das auf `normalized_violation = bed_temperature_
violation_k / 10.0`, also `Strafe_kW = _PENALTY_WEIGHT_KW * bed_temperature_violation_k /
10.0 = 5.0 * bed_temperature_violation_k` (bei `_PENALTY_WEIGHT_KW=50`). Ein Beispiel aus
WA1 (7 K ueber der Grenze) ergaebe 35 kW Strafe -- das Zahlenbeispiel unten ist aus den
tatsaechlich beobachteten XA3-Verletzungen gemessen, nicht konstruiert:

| Groesse | Median | Spannweite |
|---|---|---|
| a) E_fest - E_Orakel je Testpunkt [kW] | 0,174 | [0,000; 0,242] |
| b) Energie ueber zulaessige Zufallsaktionen [kW] (409/500 Ziehungen zulaessig) | 2,145 | [0,946; 4,145] (Spanne 3,20 kW) |
| c) Strafe fuer XA3s beobachtete Bettemperaturverletzungen [kW] (76 Verletzungen) | 89,81 | [4,82; 94,10] |

**Verhaeltnis (c)/(a) = 516** -- deutlich mehr als eine Groessenordnung (der Auftrag
nennt eine Groessenordnung als Schwelle; der gemessene Faktor liegt mehr als zwei
Groessenordnungen darueber). Wahrscheinliche Ursache, wie im Auftrag vermerkt: der
Gradient wird von der Strafe dominiert, das Energiesignal geht darin unter -- ein Agent
lernt (wenn ueberhaupt) zuerst Zulaessigkeit, dann Energie. YA1 zeigt aber, dass nach
10^4 Schritten selbst die Zulaessigkeit noch nicht gelernt ist (unter Zufallsniveau) --
konsistent mit einem Gradienten, der so stark von seltenen, extremen Strafausschlaegen
dominiert wird, dass er kein verlaessliches Signal in Richtung "vermeide die 80-°C-Grenze"
liefert, geschweige denn in Richtung Energieoptimierung.

Zusaetzlicher, eigenstaendiger Befund aus (b): SELBST OHNE Strafe ist das rohe
Energiesignal verrauscht relativ zum zu schliessenden Regret -- die Spanne zulaessiger
Zufallsaktionen (3,20 kW) ist bereits das ~18-Fache des Median-Regrets (0,174 kW). Der
Regret, den Befund 52 als Obergrenze eines Nutzens fixiert, ist eine kleine Differenz
innerhalb einer viel groesseren Energie-Landschaft -- selbst eine durchgehend zulaessige
Politik muesste dieses kleine Signal gegen ein deutlich groesseres Grundrauschen aufloesen,
mit SB3-Default-Hyperparametern (keine Abstimmung, siehe WA3).

Vier Optionen, OHNE eine auszuwaehlen (Auftrag), je mit Wirkung auf die VA2-Bitgleichheit
(`_PENALTY_WEIGHT_KW=50` ist bewusst aus Befund 52 uebernommen, weil die Orakelpruefung
Bitgleichheit verlangt -- jede Aenderung beruehrt das):

1. **Belohnung normieren** (Energie relativ zu einem Bezugswert). Aendert die
   Belohnungs-SKALA, nicht die VA2-Bitgleichheit selbst (die prueft `reward ==
   -energy_target` am Orakelpunkt, wo `normalized_violation` ohnehin ~0 ist) -- aber jede
   spaetere Lernkurve waere nicht mehr direkt in kW gegen Befund 52 lesbar, ein neuer
   Bezugswert muesste dokumentiert und versioniert werden.
2. **Strafgewicht fuer RL getrennt vom Optimierer setzen.** Bricht die Bitgleichheit VA2
   NICHT direkt (der Orakeltest bleibt exakt, weil Befund 52s Optimum ohnehin zulaessig
   ist und `normalized_violation` dort ~0), entkoppelt aber jede kuenftige Kalibrierung
   von Befund 52 als gemeinsamer Referenz -- ein anderes Gewicht fuer RL als fuer den
   Optimierer bedeutet, beide sind nicht mehr direkt vergleichbar.
3. **Lagrange-Verfahren** (Nebenbedingung als eigene Groesse statt Strafe). Groesster
   Eingriff: ersetzt die Straf-Formulierung komplett, damit auch die VA1-Entscheidung
   ("Straffunktion fuer alle drei Nebenbedingungen"). Reward waere nicht mehr direkt mit
   Befund 52s Optimierer-Zielfunktion vergleichbar, weil dieser selbst eine Straffunktion
   nutzt.
4. **Laengeres Training, Formulierung unveraendert.** Aendert an der Vergleichbarkeit
   NICHTS (exakt dieselbe Belohnung, derselbe Fixpunkt) -- aber YA1 zeigt, dass 10^4
   Schritte nicht einmal Zufallsniveau erreichen; ob mehr Schritte das Skalenproblem
   ueberwinden oder nur langsamer an derselben Verzerrung vorbeitrainieren, ist ungeklaert.

Artefakt: `artifacts/runs/rl_bandit_reward_scale.json`.

**n) AB1-AB3 -- "unter Zufall" praezise erklaert: Politikkollaps auf die heisse Ecke, kein
Code-Fehler.** Anlass: der Agent liegt bei der Zulaessigkeit UNTER dem Zufallsniveau (YA1),
was zur YA2-Erklaerung ("strafdominierter Gradient") nicht passt -- eine Strafe, die das
Energiesignal um Faktor 516 uebertrifft, ist das staerkste Signal im Problem; ein Agent
sollte sie zuerst vermeiden lernen und damit UEBER Zufall liegen, nicht darunter. Drei
Pruefungen, VOR jeder Aenderung an Belohnung oder Strafgewicht.

**AB1 (Vorzeichen der Strafe): kein Fehler.** VA2 hatte nur zulaessige Aktionen geprueft.
Bekannt unzulaessige Aktion (XA3-Beispiel, 108,8 °C/2,352 m/s, Bettemperatur 86,97 °C, 6,97
K ueber der Grenze) gegen bekannt zulaessige Aktion (90 °C/1,6 m/s) am selben Kontext:
Belohnung -38,74 gegen -2,23 -- deutlich negativer, Strafanteil korrekt positiv (+34,85 kW,
passt zu YA2s Formel). Test: `tests/test_bandit_env_reward_sign_and_normalisation.py::
test_infeasible_action_reward_is_much_more_negative_than_feasible`.

**AB2 (normierter Aktionspfad): kein Fehler.** VA2 hatte Aktionen immer physikalisch direkt
eingespeist, nie durch die Normierung. Die vier Ecken von `[-1, 1]^2` ergeben die
erwarteten physikalischen Extreme (`[1, 1]` = maximale Zulufttemperatur UND maximale
Gasgeschwindigkeit, keine vertauschte Kombination); alle 25 Orakelaktionen durch den
normierten `step()`-Pfad eingespeist reproduzieren VA2 bitgleich (max. Abweichung 3,3·10⁻¹⁶,
Gleitkommarauschen). Tests:
`test_action_space_corners_map_to_expected_physical_extremes`,
`test_oracle_actions_through_normalised_step_path_reproduce_va2`.

**AB3 (was die Politik tatsaechlich tut): KEIN Code-Fehler, aber die eigentliche
Erklaerung.** `scripts/probe_bandit_env_policy_inspection.py`, laedt die XA2-Checkpoints,
berichtet die deterministische Aktion (`model.predict(obs, deterministic=True)` --
Mittelwert der tanh-Politik, KEINE stochastische Ziehung, wie ueberall in dieser
Auswertung) fuer alle 25 Testkontexte. Artefakt:
`artifacts/runs/rl_bandit_policy_inspection.json`.

| Schritt | t_in Mittelwert (Streuung) | u_gas Mittelwert (Streuung) | Kollaps? | Heisse Ecke? |
|---|---|---|---|---|
| 1000 | +0,9907 (0,0029) | +0,9938 (0,0020) | JA | JA |
| 10000 | +0,2575 (0,6110) | +0,9295 (0,0538) | nein (t_in) | nein (t_in entkommen, u_gas weiterhin gesaettigt) |

Bei Schritt 1000 gibt die Politik fuer ALLE 25 Kontexte praktisch dieselbe Aktion aus
(Streuung 0,003/0,002 auf einer Skala von `[-1, 1]`) -- Kontext wird vollstaendig
ignoriert, KOLLAPS. Diese eine Aktion liegt an BEIDEN Schranken gleichzeitig, beide "hoch"
(t_in≈119,5 °C, u_gas≈2,49 m/s) -- die HEISSE ECKE: maximale Energie UND maximale
Bettemperatur gleichzeitig, das genaue Gegenteil dessen, was ein Energieminimierer waehlen
wuerde. Das erklaert Schritt 1000/2500 aus WA3/XA3 vollstaendig: 25/25 unzulaessig, weil
die Politik buchstaeblich immer denselben, garantiert bettemperatur-verletzenden Punkt
anfaehrt.

Bei Schritt 10000 hat sich `t_in` vom Kollaps geloest (Streuung 0,61, Spanne ueber die
Kontexte 69-120 °C) -- die Politik reagiert jetzt auf den Kontext. `u_gas` bleibt jedoch
weiterhin an der oberen Schranke gesaettigt (Mittelwert 0,93, Streuung nur 0,054) ueber
ALLE Kontexte -- diese Achse hat sich in 10^4 Schritten nicht vom Rand geloest. Das erklaert
den weiterhin schwachen Stand bei Schritt 10000 (Zulaessigkeit 0,56, unter Zufalls-Median
0,80): eine teilweise entkollabierte Politik, die auf einer von zwei Achsen weiterhin an
der schlechtesten moeglichen Schranke sitzt, bleibt strukturell im Nachteil gegenueber
einer Politik, die (wie der Zufall) auf BEIDEN Achsen frei streut und damit auch oft genug
in guenstige Regionen faellt.

**Einordnung (AUSWERTUNG laut Auftrag).** AB1/AB2 finden keinen Fehler -- Vorzeichen und
Normierung sind korrekt, keine Aenderung an Code, Belohnung oder Strafgewicht ausgeloest.
AB3 findet ebenfalls keinen CODE-Fehler, aber eine praezisere Erklaerung als "strafdominierter
Gradient" (YA2): der Agent liegt nicht deshalb unter Zufall, weil die Strafe generell
dominiert, sondern weil die Politik frueh auf die EINE schlechteste Ecke des Aktionsraums
kollabiert und sich davon nur teilweise (eine von zwei Achsen) und langsam loest. YA2s
Skalenbefund (Strafe/Signal = 516) bleibt die plausibelste Ursache DAFUER, dass der
Kollaps ueberhaupt dorthin fuehrt -- warum ein auf Reward-Maximierung trainierter Gradient
in eine derart schlechte Ecke laeuft statt sie sofort zu verlassen, ist damit aber selbst
noch nicht mechanistisch geklaert (z. B. ob eine instabile Kritiker-Schaetzung unter
extremen, unnormierten Strafauschlaegen die Ursache ist) -- NICHT in dieser Session
untersucht (haette einen Blick in die SAC-Kritiker-Schaetzung selbst gebraucht, ausserhalb
des Auftragsumfangs). Die Formulierung "kein grundsaetzliches Politik- oder
Umgebungsversagen" aus frueheren Entwuerfen ist bis zur Klaerung dieses Mechanismus als
OFFEN zu kennzeichnen, nicht als Ergebnis.

**o) BB1 -- Mechanismus des Kollaps: Kritikerfehleinschaetzung klar bestaetigt, tanh-
Saettigung als Verstaerker, "Optimismus durch Initialisierung" in der einfachen Form NICHT
bestaetigt.** `scripts/probe_bandit_env_critic_diagnosis.py`, nur mit den XA2-Checkpoints
(1000/10000 Schritte), kein Training. Artefakt:
`artifacts/runs/rl_bandit_critic_diagnosis.json`.

**(a) tanh-Saettigung -- teilweise bestaetigt, nuancierter als "waechst unbegrenzt".**
Voraktivierung `mu` (vor tanh) bei Schritt 1000: Mittelwert `[2,70; 2,91]`, Maximum
`[2,90; 3,07]` -- 9 von 25 Kontexten ueberschreiten die Schwelle `|mu|>3`
(`tanh'<0,01`) auf der Gasgeschwindigkeits-Achse, 0 von 25 auf der Temperatur-Achse. Bei
Schritt 10000 ist KEIN Kontext mehr ueber der Schwelle (`mu`-Mittelwert `[0,58; 1,88]`) --
`mu` ist also nicht monoton gewachsen, sondern fuer die Temperatur-Achse deutlich
zurueckgegangen (2,70 -> 0,58), fuer die Gasgeschwindigkeits-Achse nur wenig (2,91 -> 1,88).
Das erklaert praezise, warum sich AB3s Kollaps NUR auf der Temperatur-Achse loest: `tanh(1,9)
= 0,95`, `tanh(0,6) = 0,54` -- ein `mu`-Rueckgang von ~2,9 auf ~1,9 reicht nicht, um die
tanh-Ausgabe spuerbar von der Schranke wegzubewegen, ein Rueckgang auf ~0,6 schon. Die
Saettigung ist damit kein "Wegdriften", sondern eine Haftreibung: sobald `mu` in den flachen
Bereich von tanh laeuft, braucht es einen ueberproportional grossen weiteren Rueckgang, um
die AUSGABE messbar zu bewegen -- genau das im Auftrag beschriebene "der Akteur kommt nicht
zurueck", nur graduell statt kategorisch.

**(b) Kritikerfehleinschaetzung -- klar bestaetigt, an ALLEN 25 Kontexten, an BEIDEN
Checkpoints, und WIRD SCHLIMMER statt besser.** `Q(Ecke)` gegen `Q(Orakel)`, jeweils am
selben Kontext, normiert eingespeist (wie ein Agent sie saehe):

| Schritt | Q(Ecke) Mittelwert (2 Kritiker) | Q(Orakel) Mittelwert | Kontexte Q(Ecke)>Q(Orakel) |
|---|---|---|---|
| 1000 | -1286,6 / -1272,3 | -1603,0 / -1576,7 | 25 / 25 |
| 10000 | -14,0 / -32,6 | -9856,9 / -8878,2 | 25 / 25 |

Der Kritiker bewertet die unzulaessige Eckaktion an JEDEM einzelnen der 25 Testkontexte
hoeher (weniger negativ) als die tatsaechlich optimale Orakelaktion -- und die Luecke
VERGROESSERT sich zwischen Schritt 1000 und 10000 (Faktor ~1,2 auf Faktor ~300-700), statt
sich mit mehr Trainingsdaten zu schliessen. Das ist die zentrale, eindeutige Bestaetigung:
der Akteur folgt einem Kritiker, der die heisse Ecke systematisch ueberschaetzt und das
wahre Optimum systematisch unterschaetzt, und dieser Fehler heilt sich in 10^4 Schritten
nicht selbst.

**(c) "Optimismus durch Initialisierung" -- NICHT in der einfachen Form bestaetigt.**
Erwartung war: unbesuchte Aktionen nahe null, besuchte deutlich negativ. Gemessen (Proxy-
Aktionen -- ACHTUNG, siehe Einschraenkung unten): Mitte, Gegenecke und gemischte Ecken
liegen bei Schritt 1000 durchgehend zwischen -293 und -505, NICHT nahe null, und bei
Schritt 10000 zwischen -4535 und **+156** (ein Kritiker-Netz schaetzt eine gemischte
Aktion positiv!) -- keine saubere "nahe null"-Signatur, sondern hohe, teils widerspruechliche
Streuung zwischen den beiden Kritiker-Netzen und zwischen Aktionen. Einschraenkung, die die
Aussagekraft von (c) begrenzt: der Replay-Buffer wurde in XA2 NICHT gesichert (nur die
Politik-Gewichte) -- "tatsaechlich nie besucht" ist nicht direkt pruefbar, die vier
Proxy-Aktionen sind nur "weit von der ausgegebenen Politik entfernt", keine verifizierte
Negativliste. Die einfache Optimismus-Erklaerung ist damit widerlegt; das tatsaechliche
Bild ist eine generell instabile, teils widerspruechliche Kritiker-Schaetzung (konsistent
mit YA2s extremer, unnormierter Belohnungsskala als Ursache fuer die Instabilitaet selbst),
nicht ein einfaches "unbesucht = optimistisch, nahe null".

**Einordnung: Mechanismus insgesamt bestaetigt genug fuer BB2.** Der entscheidende, robuste
Befund ist (b): der Kritiker bewertet die heisse Ecke an jedem Kontext hoeher als das
Optimum, durchgehend und sich verschlechternd. (a) liefert eine plausible Haftreibung, die
erklaert, warum sich der Fehler nicht von selbst korrigiert. (c) widerlegt die spezifische
"Optimismus"-Lesart, aendert aber nichts an (b)s Kernaussage. Siehe BB2 fuer den naechsten
Schritt (zentrierte Belohnung) -- ausdruecklich NICHT, weil (c) das ETABLIERT haette,
sondern weil (b) unabhaengig davon steht.

**p) BB2 -- zentrierte Belohnung: der Kollaps verschwindet NICHT.** `RewardCentering`
implementiert (siehe Commit zuvor), `--reward-centering`-Flag im Lernkurven-Skript,
identischer Lauf wie XA3 (SAC, 10^4 Schritte, Seed 0, SB3-Default-Hyperparameter), nur mit
`r' = (r - r_ref) / 0,2297 kW`. Artefakte:
`artifacts/runs/rl_bandit_sac_learning_curve_bb2_centered.jsonl`,
`artifacts/runs/rl_bandit_policy_inspection_bb2_centered.json`.

| Schritt | XA3 (a) | XA3 (b) | BB2 (a) | BB2 (b) |
|---|---|---|---|---|
| 1000 | 0,00 | n. def. | 0,00 | n. def. |
| 2500 | 0,00 | n. def. | 0,00 | n. def. |
| 5000 | 0,40 | -10,59 | 0,20 | -10,22 |
| 10000 | 0,56 | -10,83 | 0,40 | -10,32 |

Zum Vergleich Zufall (YA1, Median [Min, Max]): (a) 0,80 [0,68; 0,96], (b) -6,55
[-8,96; -5,08].

**Kernfrage, ein Satz: NEIN, der Kollaps verschwindet nicht.** BB2 (zentriert) liegt bei
5000/10000 Schritten in der Zulaessigkeit SOGAR UNTER XA3 (unzentriert) -- 0,20 gegen 0,40
bei 5000 Schritten, 0,40 gegen 0,56 bei 10000 Schritten (ein Seed, keine Aussage ueber
statistische Signifikanz dieses Unterschieds moeglich, aber klar KEINE Verbesserung). Der
gepoolte Anteil (b) bleibt in derselben Groessenordnung wie unzentriert (-10,2 bis -10,3
gegen -10,2 bis -10,8). Beide Laeufe bleiben weit unter Zufallsniveau.

Politikinspektion (identisches Verfahren wie AB3) bestaetigt: bei Schritt 1000 kollabiert
die zentrierte Politik GENAUSO vollstaendig auf die heisse Ecke wie die unzentrierte
(Streuung 0,003/0,002, Mittelwert +0,992/+0,995 -- sogar noch naeher an der Schranke als
XA3s +0,991/+0,994). Bei Schritt 10000 loest sich wieder nur `t_in` (Streuung 0,66, Spanne
70,6-119,6 °C), waehrend `u_gas` weiterhin an der oberen Schranke haengt (Mittelwert 0,962,
Streuung 0,029 -- SOGAR STAERKER gesaettigt als XA3s 0,930/0,054). Die deterministischen
Aktionen streuen also NICHT staerker mit dem Kontext als zuvor.

**Einordnung, konsistent mit RewardCentering-Punkt 3 (siehe Docstring, dort vorab
angekuendigt).** Das Ergebnis ist keine Ueberraschung im Nachhinein: BB1 identifizierte die
Kritikerfehleinschaetzung (Q(Ecke) > Q(Orakel)) als zentralen Mechanismus, und diese
Fehleinschaetzung haengt plausibel an der EXTREMEN, schlecht konditionierten Groessenordnung
der Strafausschlaege (YA2, bis zu ~90 kW gegenueber einem Signal von ~0,2 kW) -- nicht an der
kontextabhaengigen Rohenergie-Grundskala, die BB2 entfernt hat. Zentrierung verschiebt und
skaliert affin, veraendert aber das VERHAELTNIS von Strafe zu Signal nicht (Faktor 516 bleibt
516) und damit auch nicht die Konditionierung der Kritiker-Zielwerte an strafenden
Transitionen. Der naechste sinnvolle Test waere eine der YA2-Optionen, die dieses Verhaeltnis
TATSAECHLICH aendern (Strafgewicht fuer RL getrennt setzen, oder Lagrange-Verfahren) --
NICHT in dieser Session (Auftrag begrenzt BB2 auf die zentrierte Belohnung, keine weitere
Aenderung an Belohnung oder Strafgewicht ohne neue Anweisung).

**q) CB1 -- Episodenende geprueft: KEIN Bootstrapping ueber das Episodenende, entgegen der
Ausgangsvermutung.** Anlass: die Kritikerluecke aus BB1 waechst mit dem Training (Faktor
1,2 -> 300-700) -- in einem Einschritt-Bandit ist das Kritikerziel reine Regression
`Q(s,a) = r(s,a)`, eine Regression divergiert nicht von selbst; ein wachsender Fehler legt
nahe, dass trotzdem gebootstrapt wird (`r + gamma * Q(s', a')` statt `r`). Vier Pruefungen,
NUR lesend, Belege aus dem Quelltext der installierten Version (stable-baselines3 2.9.0),
nicht aus Erinnerung.

**(a) `step()`-Rueckgabe, woertlich** (`src/twinlink_ml/rl/env.py:346-348`):
```python
terminated = True
truncated = False
return observation, reward, terminated, truncated, info
```
Korrekt fuer einen Einschritt-Bandit.

**(b) TimeLimit-Wrapper?** Keiner. Weder `TimeLimit` noch `max_episode_steps` noch
`gym.make`/`register()` kommen im gesamten `src/twinlink_ml/rl/`, den Trainings- oder
Auswertungsskripten vor (Volltextsuche, keine Treffer). Der Trainings-Env laeuft nur durch
`stable_baselines3.common.monitor.Monitor`, dessen `step()` `terminated`/`truncated`
unveraendert durchreicht (`Monitor.step`, installierte Quelle):
```python
observation, reward, terminated, truncated, info = self.env.step(action)
...
return observation, reward, terminated, truncated, info
```

**(c) gamma im Trainingsskript?** `scripts/run_rl_bandit_learning_curve.py:204`:
```python
model = algorithm_cls("MlpPolicy", train_env, seed=seed, verbose=0)
```
Kein `gamma=`-Argument -- SB3s SAC-Default gilt, `gamma: float = 0.99`
(`stable_baselines3/sac/sac.py`, Konstruktorsignatur, installierte Version 2.9.0).

**(d) Behandlung von `truncated` beim Bootstrapping, Beleg aus dem Quelltext der
installierten Version:**

1. `DummyVecEnv.step_wait` (`stable_baselines3/common/vec_env/dummy_vec_env.py:59-66`)
   kombiniert die beiden Flags zu einem einzigen `done` und einem separaten Timeout-Merker:
   ```python
   self.buf_dones[env_idx] = terminated or truncated
   self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated
   ```
   Mit `terminated=True, truncated=False`: `done=True`, `TimeLimit.truncated=False`.
2. `ReplayBuffer.add`/`_get_samples` (`stable_baselines3/common/buffers.py:277-278,
   320-322`) speichert diesen Timeout-Merker separat und maskiert beim Sampling:
   ```python
   self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])
   ...
   # Only use dones that are not due to timeouts
   (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
   ```
   Mit `dones=True, timeouts=False`: der an SAC gelieferte `dones`-Wert ist `True * (1-0) =
   True`.
3. `SAC.train` (`stable_baselines3/sac/sac.py:220,260`) bildet das Ziel:
   ```python
   discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
   ...
   target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
   ```
   Mit `replay_data.dones=True`: `(1 - 1) = 0`, der gesamte Bootstrap-Term
   `discounts * next_q_values` wird mit 0 multipliziert -- UNABHAENGIG vom Wert von
   `discounts`/`gamma`. Das Ziel ist exakt `replay_data.rewards`, kein Bootstrapping.

**Kernfrage, ein Satz: NEIN, es wird nicht ueber das Episodenende hinweg gebootstrapt** --
die vier Glieder der Kette (Umgebung -> `DummyVecEnv` -> `ReplayBuffer` -> `SAC.train`)
setzen `terminated`/`truncated` durchgehend korrekt fort, und die Zielformel maskiert den
Bootstrap-Term exakt auf null. CB1 findet KEINEN Fehler im Episodenende -- die
Ausgangsvermutung ist durch den Quelltext widerlegt. Die wachsende Kritikerluecke aus BB1
braucht eine andere Erklaerung; naheliegend (nicht in dieser Session geprueft): ein
neuronales Netz als Funktionsapproximator kann bei extremer Konzentration der Ziehungen auf
eine schmale Region (die kollabierte Ecke) an SELTEN besuchten Punkten (wie der Orakelaktion)
beliebig schlecht extrapolieren, unabhaengig davon, ob innerhalb der besuchten Region
korrekt (nicht-bootstrappend) regressiert wird -- ein Generalisierungsfehler, keine
Bootstrapping-Divergenz. CB2 prueft das empirisch, unabhaengig von dieser Lesart.

**r) CB2 -- gamma=0 bestaetigt CB1 empirisch: das Ergebnis ist BITIDENTISCH zu XA3.**
Identischer Lauf wie XA3 (SAC, 10^4 Schritte, Seed 0, rohe Belohnung, SB3-Default sonst),
einzige Aenderung `gamma=0.0` statt SB3s Default `0.99`. Artefakte:
`artifacts/runs/rl_bandit_sac_learning_curve_cb2_gamma0.jsonl`,
`artifacts/runs/rl_bandit_policy_inspection_cb2_gamma0.json`,
`artifacts/runs/rl_bandit_critic_diagnosis_cb2_gamma0.json`.

| Schritt | XA3 (a) | XA3 (b) | CB2 (a) | CB2 (b) |
|---|---|---|---|---|
| 1000 | 0,00 | n. def. | 0,00 | n. def. |
| 2500 | 0,00 | n. def. | 0,00 | n. def. |
| 5000 | 0,40 | -10,5879 | 0,40 | -10,5879 |
| 10000 | 0,56 | -10,8269 | 0,56 | -10,8269 |

Nicht nur "kein deutlicher Unterschied" -- die Zahlen stimmen auf ALLE berichteten
Nachkommastellen ueberein, bei (a), (b), der nennerempfindlichen Median/P95/Min-Metrik UND
den Verletzungszaehlern je Checkpoint (25/25/15/11 -- exakt wie XA3). Die
Politikinspektion (identisches Verfahren wie AB3/BB2) bestaetigt dasselbe fuer die
deterministischen Aktionen (Schritt 1000: `t_in=+0,9907 u_gas=+0,9938`, Streuung
`0,0029/0,0020`, Kollaps auf die heisse Ecke; Schritt 10000: `t_in=+0,2575 u_gas=+0,9295`,
Streuung `0,6110/0,0538`, nur `t_in` geloest) -- Werte bitgleich zu XA3s AB3-Ergebnis. Die
Kritikerluecke (identisches Verfahren wie BB1) ist ebenfalls bitgleich: `Q(Ecke)`/`Q(Orakel)`
Mittelwerte `[-1286,64; -1272,25]`/`[-1603,02; -1576,72]` bei Schritt 1000 und
`[-14,0; -32,6]`/`[-9856,9; -8878,2]` bei Schritt 10000, 25/25 Kontexte mit `Q(Ecke) >
Q(Orakel)` an BEIDEN Checkpoints -- identisch zu BB1.

**Kernfrage: der Kollaps verschwindet NICHT, und der Kritikerfehler waechst GENAUSO wie
zuvor** (Faktor ~1,2 bei Schritt 1000 auf Faktor ~300-700 bei Schritt 10000, exakt wie in
BB1 berichtet) -- weil es, wie CB1 durch Quelltextanalyse zeigte, gar keinen messbaren
Unterschied zwischen den beiden Laeufen gibt: `gamma` geht in `SAC.train`s Zielformel nur
als Faktor von `(1 - dones) * discounts * next_q_values` ein, und `(1 - dones) = 0` fuer
JEDE Transition dieses Environments (jede Episode endet nach einem Schritt, `dones=True`
immer). Der Bootstrap-Term ist unabhaengig vom Wert von `gamma` exakt null -- ein Lauf mit
`gamma=0` und einer mit `gamma=0,99` durchlaufen deshalb IDENTISCHE Gradientenschritte, von
der Seed-Initialisierung bis zum letzten Update. CB2 ist damit die staerkste moegliche
Bestaetigung von CB1: nicht nur "kein Bootstrapping-Bug gefunden", sondern empirisch belegt,
dass die Aenderung, die einen Bootstrapping-Fehler sichtbar gemacht haette, NULL Wirkung
zeigt.

**Einordnung.** Die urspruengliche Vermutung (wachsender Kritikerfehler deutet auf
Bootstrapping ueber das Episodenende hin) ist damit doppelt widerlegt -- einmal durch
Quelltextanalyse (CB1), einmal durch ein bitgenau reproduzierendes Experiment (CB2). Die
wachsende Kritikerfehleinschaetzung aus BB1 hat eine andere Ursache, die in dieser Session
NICHT geklaert wurde (CB1s Hypothese: Generalisierungsfehler eines Funktionsapproximators
bei stark konzentrierter/kollabierter Aktionsverteilung -- die Politik besucht die
Orakelregion praktisch nie, der Kritiker sieht dort kaum Trainingsdaten und kann dort
beliebig falsch liegen, ohne dass eine korrekte, nicht-bootstrappende Regression auf den
tatsaechlich besuchten Punkten das verhindert). Keine Aenderung an Belohnung, Strafgewicht
oder Umgebung vorgenommen.

**s) DA0 -- die SB3-Schnittstelle: kein Fehler, aber die Praemisse "heisse Ecke = globales
Belohnungsminimum" ist zu praezisieren.** Vier Pruefungen, vor jeder weiteren Aenderung.

**(a) 13x9-Gitter (Nominalstoergroesse) neu ausgewertet (reine Auswertung, keine
Trainingsaenderung):** die heisse Ecke (120 °C, 2,5 m/s) ist bestaetigt das ARGMAX der
ROHEN ENERGIE im Gitter (4,50 kW) -- aber NICHT das globale Argmin der Belohnung. Unter den
Gitterpunkten, die ueberhaupt loesen, liegt die schlechteste Belohnung (-155,58) bei (65 °C,
0,8 m/s) -- kalt UND langsam, mit einer GUTSFEUCHTE-Verletzung von 0,419 kg/kg (Feuchte
0,619 gegen kritisch 0,20), nicht Bettemperatur. Am EXAKTEN kalten/langsamen Eckpunkt
(60 °C, 0,8 m/s) scheitert der Loeser sogar ganz (Bracket-Vorzeichenfehler) -- dieser Punkt
erhaelt die Sentinel-Strafe `_SOLVE_FAILURE_PENALTY_KW = 1.000.000`, vier Groessenordnungen
unter jedem anderen Wert im Gitter. Die vier exakten Eckpunkte im Vergleich:

| Ecke | reward | energy | zulaessig | Grund |
|---|---|---|---|---|
| kalt/langsam (60, 0,8) | -1.000.000 (Loeserfehler) | -- | -- | Bracket-Vorzeichenfehler |
| kalt/schnell (60, 2,5) | -2,21 | 2,21 | JA | -- |
| heiss/langsam (120, 0,8) | -1,55 | 1,55 | JA | -- |
| heiss/schnell (120, 2,5) | -94,93 | 4,50 | NEIN | Bettemperatur +18,09 K |

Praezisierung: von den VIER Ecken ist die heisse/schnelle tatsaechlich die schlechteste
(einzige unzulaessige feste Ecke) -- aber im GESAMTEN Gitter (nicht nur den vier Ecken) gibt
es einen noch schlechteren Bereich am kalten/langsamen Rand, angetrieben von der
Gutsfeuchte, nicht der Bettemperatur, und am exakten Eckpunkt dort eine numerische
Loeserfalle mit einer um vier Groessenordnungen extremeren Sentinel-Strafe. Die Politik
konvergiert NICHT zu diesem objektiv schlechtesten Bereich, sondern zur heissen Ecke --
"praezise am globalen Argmin" (Auftrag) trifft damit NICHT woertlich zu, auch wenn die
heisse Ecke unter den VIER Ecken die schlechteste ist.

Neuer, eigenstaendiger Befund: die Sentinel-Strafe (-1.000.000) am kalten/langsamen
Eckbereich ist ein moeglicher ZUSAETZLICHER Verzerrungsfaktor, nicht in DA0 selbst
untersucht -- trifft die Zufalls-Aufwaermphase (`learning_starts=100`, gleichverteilt ueber
die GESAMTE Box) diesen numerisch fragilen Bereich auch nur einmal, geht ein Ziel-Wert vier
Groessenordnungen groesser als alles sonst in den Replay-Puffer und damit in die fruehe
Kritiker-Regression ein.

**(b) `env.action_space`, woertlich** (`src/twinlink_ml/rl/env.py:271`):
```python
self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
```
Exakt `[-1, 1]` -- dieselbe Spanne, in die SACs Politik intern quetscht (tanh).

**(c)/(d) SB3-Aktionsabbildung, Beleg aus dem Quelltext der installierten Version (2.9.0),
keine Doppelabbildung gefunden.** `BasePolicy.predict` (`stable_baselines3/common/
policies.py:372-375`):
```python
if self.squash_output:
    # Rescale to proper domain when using squashing
    actions = self.unscale_action(actions)
```
`unscale_action`/`scale_action` (dieselbe Datei) bilden `[-1,1] <-> [low, high]` linear ab.
Da `low=-1, high=1` EXAKT, ist diese Abbildung fuer unseren `action_space` rechnerisch die
IDENTITAET (`unscale_action(a) = -1 + 0.5*(a+1)*2 = a`) -- unabhaengig davon, wie oft sie
verkettet wird (`OffPolicyAlgorithm._sample_action`,
`stable_baselines3/common/off_policy_algorithm.py:398-410`, verkettet `predict` ->
`scale_action` -> `unscale_action` waehrend des Trainings, alles Identitaet fuer uns). Die
an `step()` uebergebene Aktion ist bitgleich der rohen tanh-Politikausgabe -- keine
Umkehrung, keine Verschiebung, kein Skalierungsfehler.

Nebenfund, kein Bug: `_sample_action` ruft `self.predict(obs, deterministic=False)`
waehrend des Trainings (Zeile 398) -- SAC sampelt WAEHREND DES TRAININGS stochastisch
(explorierend um den aktuellen Mittelwert), nicht deterministisch. Alle
Politikinspektionen dieser Session (AB3, BB1, BB2, CB2) fragen bewusst `deterministic=True`
ab (den MITTELWERT) -- das ist Standard-SAC-Verhalten (Training explorativ, Auswertung
deterministisch), keine Diskrepanz zwischen Inspektion und tatsaechlichem Verhalten, aber
die waehrend des Trainings tatsaechlich BESUCHTEN Aktionen streuen um diesen Mittelwert
(Groessenordnung aus BB1s `log_std`, dort ca. -0,64 bei Schritt 1000, `std = exp(-0,64) ~=
0,53` im Voraktivierungsraum).

**Kernfrage, ein Satz: JA, die Aktion, die SAC waehlt, kommt in `step()` genauso an, wie
die Politik sie meint** -- keine Doppelabbildung oder Umkehrung gefunden, `env.action_space
= [-1,1]` macht jede SB3-interne Rescale-Operation zur Identitaet. DA0 findet KEINEN Fehler
im Uebergabeweg; DA1 kann wie geplant folgen. Die Praemisse "heisse Ecke = globales
Belohnungsminimum" ist mit (a) zu praezisieren: die heisse Ecke ist das Argmax der rohen
Energie und die schlechteste der vier Ecken, aber NICHT das globale Minimum der Belohnung
im gesamten Aktionsraum -- dieses liegt (Gutsfeuchte-getrieben, mit einer numerischen
Loeserfalle am exakten Eckpunkt) am entgegengesetzten, kalten/langsamen Rand.

**t) DA1 -- 2x2-Matrix: SAC kann diesen Bandit mit Voreinstellungen grundsaetzlich nicht,
die Strafklippe ist NICHT die Ursache.** Vier Zellen, sonst identisch zu XA3 (SAC, 10^4
Schritte, Seed 0, SB3-Default-Hyperparameter, gamma unveraendert). Zelle 1 = XA3
(wiederverwendet). Artefakte je neue Zelle: `rl_bandit_sac_learning_curve_da1_*.jsonl`,
`rl_bandit_policy_inspection_da1_*.json`, `rl_bandit_critic_diagnosis_da1_*.json`.

| Zelle | learning_starts=100 | learning_starts=2000 |
|---|---|---|
| **mit Strafe** | Zelle 1 = XA3 | Zelle 2 (neu) |
| **ohne Strafe** | Zelle 3 (neu) | Zelle 4 (neu) |

Mittlere rohe Energie der deterministischen Politik an den 25 Testkontexten gegen den
festen Betriebspunkt (1,179 kW), UNABHAENGIG von Zulaessigkeit -- das primaere Kriterium
(Auftrag):

| Zelle | Energie Schritt 1000 | Energie Schritt 10000 | Zulaessigkeit 1000 | Zulaessigkeit 10000 |
|---|---|---|---|---|
| 1 (Strafe, LS=100, XA3) | 4,484 kW | 3,639 kW | 0,00 | 0,56 |
| 2 (Strafe, LS=2000) | 1,771 kW* | 3,523 kW | 1,00* | 0,48 |
| 3 (ohne Strafe, LS=100) | 4,483 kW | 3,636 kW | 0,00 | 0,28 |
| 4 (ohne Strafe, LS=2000) | 1,771 kW* | 3,412 kW | 1,00* | 0,40 |

\* Bei Schritt 1000 hat das Training noch nicht begonnen (`learning_starts=2000`) -- Zellen
2 und 4 sind an diesem Punkt noch die REIN ZUFAELLIGE Aufwaermphase, mit demselben Seed
bitgleich zueinander (siehe Politikinspektion unten). "Zulaessigkeit 1000" ist dort ein
Artefakt der Zufallsziehung, kein Politikergebnis.

**Bei Schritt 10000 konvergieren ALLE VIER Zellen auf nahezu dieselbe erhoehte Energie
(3,41-3,64 kW), unabhaengig von Strafe UND Aufwaermphasenlaenge** -- eine Spanne von nur
0,23 kW ueber alle vier Zellen, gegenueber einem Abstand von ueber 2,2 kW zum festen
Betriebspunkt. Die Zulaessigkeit variiert staerker (0,28-0,56), aber OHNE erkennbares
Strafe/Aufwaermphasen-Muster -- Zelle 3 (ohne Strafe, kurze Aufwaermphase) hat die
NIEDRIGSTE Zulaessigkeit (0,28) im gesamten Feld.

**Politikinspektion, WOHIN konvergiert die Politik:**

| Zelle | Schritt 1000: Region | Schritt 10000: Region |
|---|---|---|
| 1 (XA3) | HEISSE ECKE, Kollaps (Streuung 0,003/0,002) | t_in geloest (Streuung 0,61), u_gas gesaettigt hoch (0,93) |
| 2 | nicht kollabiert (Zufallsphase) | t_in=+0,40, u_gas=+0,72, keine Saettigung, aber "warme" Region |
| 3 | HEISSE ECKE, Kollaps (Streuung 0,003/0,002) -- BITGLEICH zu Zelle 1 | t_in=+0,70, u_gas=+0,61, keine Saettigung, "warme" Region |
| 4 | nicht kollabiert (Zufallsphase), BITGLEICH zu Zelle 2 | t_in=+0,56, u_gas=+0,53, "warme" Region |

Zelle 3 (KEINE Strafe, Standard-Aufwaermphase) kollabiert bei Schritt 1000 auf eine normierte
Aktion `[+0,9912; +0,9934]` -- bitgleich (auf vier Nachkommastellen) zu Zelle 1s
`[+0,9907; +0,9938]` MIT Strafe. Ohne jede Strafe, rein durch Energieminimierung motiviert,
waere die rationale Richtung KALT/LANGSAM (Energieminimum, siehe DA0(a): ~0,81 kW bei
65 °C/0,8 m/s) -- die Politik laeuft stattdessen in die exakt entgegengesetzte, energie-
MAXIMIERENDE Ecke. Das ist mit einer korrekt funktionierenden Politikverbesserung nicht
vereinbar.

**Kritikerdiagnose bestaetigt: die extreme Fehleinschaetzung tritt AUCH OHNE STRAFE auf.**
Zelle 3, Schritt 10000: `Q(Ecke)`/`Q(Orakel)` Mittelwerte `[6,34; 2,89]` / `[-9924,39;
-9386,51]`. Zelle 4: `[118,67; 90,26]` / `[-6406,24; -6190,25]`. Diese Groessenordnungen
(minus mehrere tausend) sind mit den tatsaechlich MOEGLICHEN rohen Belohnungen in diesem
Modus (roh, ohne Strafe: schlechtstenfalls ca. -4,5 kW fuer die teuerste Energie im
Gitter, siehe DA0(a)) UNVEREINBAR -- der Kritiker ist um mehrere Groessenordnungen von
jedem physikalisch moeglichen Wert entfernt, obwohl gar kein Strafterm im Spiel ist.

Wahrscheinlichste Erklaerung, NICHT in dieser Session verifiziert: der
Loeserfehlschlag-Sentinel (`_SOLVE_FAILURE_PENALTY_KW = 1.000.000`, siehe DA0(a)) bleibt
auch im `--no-penalty`-Modus AKTIV (bewusste Entscheidung, siehe `BanditEnvConfig.include_
penalty`-Docstring: kein Nebenbedingungs-Strafterm, aber ein Loeserfehlschlag hat schlicht
keine Energie, die zurueckgegeben werden koennte). Trifft die Zufalls-Aufwaermphase den in
DA0 gefundenen numerisch fragilen Bereich (kalt/langsam) auch nur einmal, geht ein
Ziel-Wert von -1.000.000 in die fruehe Kritiker-Regression ein -- sechs Groessenordnungen
ueber jedem sonst moeglichen Wert. Das wuerde erklaeren, warum selbst der straflose Modus
eine derart instabile Kritiker-Schaetzung zeigt. NICHT verifiziert (haette eine
Rekonstruktion der exakten SB3-Zufallsziehungssequenz gebraucht, ausserhalb des
Auftragsumfangs) -- als plausibelste Erklaerung vermerkt, nicht als Ergebnis.

**Kernfrage, vorab festgelegte Lesarten (Auftrag) bewertet:**

- ~~Kollabiert SAC auch ohne Strafe in die heisse Ecke: SAC kann diesen Bandit mit
  Voreinstellungen grundsaetzlich nicht, die Strafe ist nicht die Ursache~~ -- **BESTAETIGT.**
  Zelle 3 kollabiert bitgleich zu Zelle 1 trotz fehlender Strafe; alle vier Zellen
  konvergieren bei Schritt 10000 auf praktisch dieselbe erhoehte Energie.
- ~~Ohne Strafe Richtung kalte Ecke, mit Strafe heisse Ecke: die Strafklippe ist die
  Ursache~~ -- **WIDERLEGT.** Ohne Strafe waere die kalte Ecke die rationale Richtung
  (Energieminimum); beobachtet wird das Gegenteil.
- ~~Laengere Zufallsphase verhindert den Kollaps: fruehes Ausnutzen des Kritikers ist die
  Ursache~~ -- **WIDERLEGT, PRAEZISIERT.** `learning_starts=2000` verzoegert den sichtbaren
  Kollaps (Zellen 2/4 zeigen bei Schritt 1000 noch keine kollabierte Aktion, weil das
  Training dort noch nicht begonnen hat), verhindert ihn aber nicht -- bei Schritt 10000
  ist die Kritikerfehleinschaetzung mindestens ebenso extrem (Zelle 2: `Q(Ecke)=+56,4`
  gegen `Q(Orakel)=-14051`, staerker als Zelle 1) und die Energie praktisch identisch zu
  den kurzen Aufwaermphasen.

**Einordnung.** Die urspruengliche DA0/DA1-Praemisse ("heisse Ecke = Argmin, Strafe als
naheliegendste Erklaerung") war zu eng. Der tatsaechliche Befund ist staerker und
allgemeiner: SAC mit SB3-Default-Hyperparametern findet auf diesem Bandit -- mit oder ohne
Strafe, mit kurzer oder langer Aufwaermphase -- systematisch eine energiereiche,
ueberwiegend unzulaessige Region, nicht das Energieminimum und nicht das wahre Optimum.
Das ist ein Befund ueber DIESE Kombination aus Algorithmus-Voreinstellungen und
Problemstruktur (sehr kleiner, deterministischer Kontext-Aktions-Raum, extreme
Belohnungsausreisser durch Loeserfehlschlaege), nicht ueber die Strafgewichtung allein.
Keine Aenderung an Belohnung, Strafgewicht oder Umgebung ueber die bereits dokumentierten,
schaltbaren Modi hinaus vorgenommen.

**u) EA1 -- Nachweis aus vorhandenen Daten: der Sentinel-Wert passt quantitativ, exakte
Trefferzahlen nicht ermittelbar.** Vier Teilfragen, ohne Training.

**(a) Sentinel-Treffer im XA3-/DA1-Lauf -- NICHT exakt zaehlbar aus vorhandenen Daten.**
Weder `run_rl_bandit_learning_curve.py` noch die Checkpoints protokollieren die einzelnen
Schritte WAEHREND des Trainings (nur die vier Zwischenauswertungen an den 25 fixen
Testpunkten) -- eine exakte Zaehlung haette eine Rekonstruktion der SB3-internen
Zufallsziehungssequenz gebraucht, ausserhalb des Auftragsumfangs (siehe DA1s
Einschraenkung dazu). `n_violations_solve_failed` an den 25 Testpunkten ist bei ALLEN
Checkpoints alle Laeufe 0 -- das sagt nur, dass die KONVERGIERTE Politik keine
Fehlschlaege mehr trifft, nichts ueber die Zufalls-Aufwaermphase. Indirekter, aber klarer
Beleg stattdessen unter (c).

**(b) Fehlschlagsregion kartiert (25x18-Gitter, Nominalstoergroesse, dann verfeinert auf
21x26 im Nahbereich):** ein SCHMALER Keil an der kalten/langsamen Ecke, `t_in` in
`[60; ~60,8]` °C (von 60 °C Spanne im gesamten Aktionsraum -- ca. 1,3 % der `t_in`-Achse),
`u_gas` darin bis maximal `[0,8; 1,08]` m/s am aeussersten Rand (`t_in=60`), auf `[0,8;
0,82]` schrumpfend bei `t_in=60,5`, verschwunden ab `t_in=61`. Flaechenanteil an der
gesamten Aktionsbox grob 0,1-0,7 % (zwei Gitteraufloesungen, konsistente Groessenordnung).
Bei gleichverteilter Ziehung (`env.action_space.sample()` waehrend der SB3-Aufwaermphase)
ergibt das einen ERWARTETEN Trefferanteil von < 1 bei `learning_starts=100` (0,1-0,7
erwartete Treffer -- im Wesentlichen ein Muenzwurf, ob EIN Treffer ueberhaupt vorkommt) und
2-14 erwartete Treffer bei `learning_starts=2000` -- konsistent mit DA1s Beobachtung, dass
die laengere Aufwaermphase die Kritikerfehleinschaetzung NICHT verbessert, sondern (Zelle
2 gegenueber Zelle 1) eher verschlechtert.

**(c) Kritikerwerte an der kalten Ecke, alle vier Zellen, Schritt 10000** (aus den bereits
gesicherten Kritikerdiagnose-Artefakten, `proxy_unvisited_action_q_at_context_0.
gegenecke_kalt`):

| Zelle | Q(kalte Ecke) |
|---|---|
| 1 (XA3) | [-4504,0; -4535,2] |
| 2 | [-6907,0; -7590,2] |
| 3 | [-4444,0; -4480,2] |
| 4 | [-3031,1; -3000,1] |

Alle vier Zellen: Groessenordnung -3000 bis -7600 -- weit jenseits JEDER physikalisch
moeglichen rohen Belohnung in diesem Aktionsraum (schlechtester Energiewert im Gitter
~-4,5 kW, schlechtester beobachteter Strafwert ~-155 kW, siehe DA0/YA2), aber auch weit
UNTER dem Sentinel selbst (-1.000.000) -- ein Muster, das zu einer GEGLAETTETEN Regression
auf einen seltenen, extremen Zielwert passt (ein neuronales Netz lernt bei MSE-Verlust
keinen exakten Ausreisser, sondern eine im Eingaberaum geglaettete Naeherung, die in der
Naehe des Ausreissers stark, aber nicht bis zum vollen Betrag, absinkt). Das ist der
staerkste indirekte Beleg dieser Untersuchung: OHNE einen Sentinel-Treffer irgendwo in der
Naehe waere kein Mechanismus ersichtlich, der Q-Werte dieser Groessenordnung an gerade
DIESER Stelle erzeugen wuerde.

**(d) Warum der Loeser dort scheitert -- Modellphysik, nicht Bracket-Wahl.** Woertlich aus
`steady_state.py` (Kommentar vor der Bracket-Festlegung):

```
Untere Grenze bewusst nicht symmetrisch zur oberen: bei Betttemperaturen nahe
T_amb kippt das Modell für typische Zuluftfeuchten in die Kondensationsbranche
von evaporation_rate_kg_per_s (Y_in > Y_s(T_bed)) -- dort existiert im
Feuchtebracket [X_eq, moisture_high] kein Vorzeichenwechsel, weil die Bettmasse
bei jeder Feuchte netto zunimmt. Das ist keine numerische Schwäche, sondern die
Modellphysik (ein Bett, das kälter als die Zuluft ankommt, trocknet nicht,
sondern befeuchtet sich).
```

Praezise Antwort auf die gestellte Alternative (numerische Bracket-Grenze ODER kein
Beharrungszustand): WEDER/NOCH in Reinform. Bei niedrigem `t_in` UND niedrigem `u_gas`
(minimaler Waermeeintrag) kann sich das Bett nahe der Umgebungstemperatur einpendeln --
dort kippt die MODELLGLEICHUNG selbst von Verdunstung auf Kondensation um, ein Regime, das
dieses reduzierte Grey-Box-Modell strukturell nicht abbildet (keine Kondensationsgleichung
vorhanden). Ein "Beharrungszustand" in einem allgemeineren physikalischen Sinn (ein
befeuchtendes statt trocknendes Gleichgewicht) mag existieren, ist aber ausserhalb dessen,
was DIESES Modell berechnen kann -- die 10-K-Marge der Bracket-Untergrenze ist eine
BEWUSSTE Entscheidung, das Modell im Trocknungsregime zu halten, fuer das es gebaut ist,
nicht eine zu eng gewaehlte Zahl, die sich durch Verbreitern beheben liesse (der
Modul-Docstring bestaetigt das ausdruecklich an mehreren Stellen).

**v) EA2 -- der entscheidende Lauf: Sentinel-Hypothese BESTAETIGT, vollstaendig und auf
allen Ebenen.** DA1s Zelle "ohne Strafe, `learning_starts=100`" wiederholt, einzige
Aenderung: `solve_failure_penalty_kw=4,500067` (EA1s modusabhaengiger Grenzwert, siehe
`scripts/probe_bandit_env_solve_failure_bound.py`) statt Befund 52s Sentinel
(1.000.000). Seed 0, sonst identisch. Artefakte:
`rl_bandit_sac_learning_curve_ea2_nopenalty_bounded.jsonl`,
`rl_bandit_policy_inspection_ea2_nopenalty_bounded.json`,
`rl_bandit_critic_diagnosis_ea2_nopenalty_bounded.json`.

| Schritt | Zulaessigkeit | mittlere Energie | Delta zu fest (1,179 kW) | Verletzung |
|---|---|---|---|---|
| 1000 | 1,00 | 1,672 kW | +0,493 | keine |
| 2500 | 1,00 | 1,468 kW | +0,289 | keine |
| 5000 | 0,96 | 1,281 kW | +0,102 | 1x Feuchte |
| 10000 | 0,52 | 1,054 kW | **-0,125** | 12x Feuchte |

Die mittlere Energie FAELLT MONOTON ueber das Training und unterschreitet bei Schritt 10000
den FESTEN BETRIEBSPUNKT selbst (1,054 kW gegen 1,179 kW) -- zum ersten Mal in dieser
gesamten Untersuchungsreihe (WA3 bis DA1) bewegt sich eine trainierte Politik in Richtung
Energieoptimum statt davon weg. Die Verletzungen wechseln vollstaendig die Art: KEINE
Bettemperaturverletzung mehr an irgendeinem Checkpoint (`bett=0` durchgehend), stattdessen
Feuchteverletzungen (1 bei Schritt 5000, 12 bei Schritt 10000) -- physikalisch genau die
erwartete Nebenbedingung am kalten/energiesparenden Rand.

**Politikinspektion bestaetigt die Richtung explizit.** Schritt 1000: normierte Aktion
`[-0,19; -0,44]` (schwach negativ, beide Achsen). Schritt 10000: `[-0,44; -0,93]` --
DEUTLICH negativ auf BEIDEN Achsen, `u_gas` an der UNTEREN Schranke gesaettigt (0,93,
"niedrig", nicht "hoch" wie in jeder frueheren Zelle). Physikalische Aktion bei Schritt
10000: `t_in≈63,4-63,0 °C`, `u_gas≈0,85-0,84 m/s` -- das liegt UNMITTELBAR am Rand der in
EA1(b) kartierten Loeserfehlschlagsregion (`t_in` ab 61 °C loesbar, `u_gas`-Untergrenze
0,8 m/s). Die Politik bewegt sich also nicht nur "in die richtige Richtung", sondern bis
an den Rand des ueberhaupt loesbaren Bereichs -- exakt die im Auftrag vorab festgelegte
Lesart ("an den Rand des loesbaren Bereichs... das ist physikalisch der Rand, an dem ein
Beharrungszustand noch existiert, und eine sinnvolle Loesung").

**Kritikerdiagnose: keine Ausreisser mehr, an KEINEM Checkpoint.** `Q`-Werte (beide
Kritiker, alle gepruefte Aktionen inkl. Ecken und Kontext 0) liegen bei Schritt 1000
durchgehend in `[-3,15; -1,2]`, bei Schritt 10000 in `[-2,85; -0,9]` -- die GESAMTE
Kritiker-Wertlandschaft bleibt im physikalisch plausiblen Bereich (roh, ohne Strafe:
schlechtester Wert im Gitter `-4,5` kW). `Q(Ecke) > Q(Orakel)` gilt bei Schritt 1000 nur an
10 von 25 Kontexten (vorher immer 22-25/25) -- auch die Kritiker-Fehlrangfolge ist deutlich
schwaecher. Keine `|mu| > 3`-Saettigung an irgendeinem Checkpoint (vorher regelmaessig 5-9
von 25).

**Kernfrage, vorab festgelegte Lesart: BESTAETIGT.** "Konvergiert die Politik jetzt Richtung
kalte, energiesparende Ecke (bzw. an den Rand des loesbaren Bereichs): Sentinel bestaetigt
als Ursache" -- exakt das ist eingetreten, auf allen vier unabhaengig geprueften Ebenen
(Zulaessigkeit/Energie, deterministische Aktionen, Kritikerwerte, Saettigung). Die
Gegenprobe (weiterhin heisse Ecke) trat NICHT ein.

**w) EA3 -- Einordnung, allgemeines Muster.** Neben dem in XA4 dokumentierten Muster zur
80-°C-Grenze (ein fuer die Optimierung harmloser Platzhalter wird fuer RL wirksam, weil ein
Optimierer nur die Umgebung des Optimums sieht, ein Agent die ganze Box) steht ein zweites,
unabhaengiges Muster: **ein Rueckgabewert, der fuer einen KLASSISCHEN OPTIMIERER richtig
gesetzt ist, kann eine REGRESSIONSBASIERTE WERTFUNKTION vergiften.**

`differential_evolution` (Befund 52s eigener Optimierer) vergleicht Kandidatenwerte nur
ORDINAL -- ein Rueckgabewert von `-1.000.000` bedeutet dort ausschliesslich "schlechter als
jeder andere gefundene Kandidat", unabhaengig vom genauen Betrag. Fuer diesen Zweck ist ein
beliebig grosser Sentinel nicht nur unschaedlich, sondern die SICHERE Wahl (siehe Befund 52s
eigene Begruendung fuer `_SOLVE_FAILURE_PENALTY_KW`: "draengt den Optimierer weg, ohne den
Lauf abzubrechen").

Ein neuronaler Kritiker in einem Actor-Critic-Verfahren tut etwas kategorisch anderes: er
REGREDIERT auf den tatsaechlichen Zahlenwert (`Q(s,a) ≈ r(s,a)` in diesem bootstrap-freien
Bandit, siehe CB1/CB2). Eine MSE-Regression gewichtet grosse Residuen QUADRATISCH -- ein
einzelner Ausreisser von `10^6` inmitten von Zielwerten der Groessenordnung `1-150` dominiert
den Gradienten so stark, dass er die gesamte gelernte Wertlandschaft verzerrt (EA1(c): Q-Werte
an einer numerisch fragilen, aber ansonsten irrelevanten Ecke lagen bei -3000 bis -7600,
obwohl kein anderer Punkt im Aktionsraum jemals unter -155 lag). Dieselbe Zahl, die fuer
Befund 52s Optimierer eine bewusste, richtige Entscheidung war, ist fuer einen
regressionsbasierten RL-Kritiker eine strukturelle Gefahr -- unabhaengig von der
Nebenbedingungs-Strafgewichtung (YA2) und unabhaengig vom Episodenende (CB1/CB2). Wer eine
bestehende Straf-/Fehlschlagskonstante aus einem klassischen Optimierer in ein
Actor-Critic-Verfahren uebernimmt, sollte diesen Unterschied zwischen ordinalem Vergleich und
Regressionsziel explizit pruefen, nicht stillschweigend annehmen, dass ein "ausreichend
schlechter" Wert in beiden Faellen gleich wirkt.

**Provenienz.** Branch `rl-env`, eigener Worktree (`git worktree add ../twinlink_rl -b
rl-env master`) -- Hauptverzeichnis mit laufendem Mehrstunden-Lauf nicht angefasst.
`pip install gymnasium stable-baselines3` lief nach `--dry-run`-Pruefung (nur 4 neue
Pakete, keine Aenderung an numpy/scipy/torch) gegen das globale, worktree-uebergreifend
geteilte Python -- es gibt kein Worktree-eigenes venv. Fund dabei: der editable
Paket-Install (`pip install -e .`) zeigt fest auf das Hauptverzeichnis; Code in diesem
Worktree wird ueber `PYTHONPATH` vorrangig gemacht, NICHT ueber einen erneuten `pip
install -e .` (der haette den globalen Zeiger auf diesen Worktree umgebogen und damit
`spawn`-Worker-Prozesse des Hauptlaufs -- siehe `study_operating_point_regret.py`s
Windows-`spawn`-Multiprocessing -- auf den falschen Quellcode verweisen lassen).

**x) FA1-FA4 -- Abschlusslauf: Sentinel-Pathologie behoben, Strafdominanz jetzt sichtbar
und dominant. Die RL-Voruntersuchung ist damit abgeschlossen.** Direkt auf `master`
gearbeitet (alle Entwicklungslinien zusammengefuehrt, kein Worktree noetig). Frage: findet
der Agent das BESCHRAENKTE Optimum, wenn Strafe UND begrenzter Sentinel gleichzeitig aktiv
sind -- EA2 hatte das nur ohne Strafe gezeigt.

**FA1 -- Sentinelwert fuer den Modus mit Strafe: 155.57896052362634 kW, nicht die erwartete
Groessenordnung -97 kW.** Der Wert war bereits in Commit 9925094 berechnet
(`artifacts/runs/rl_bandit_solve_failure_bound.json`, `with_penalty`) und hier bitgleich
reproduziert: schlechteste Summe aus Energie und Strafe unter den 115/117 loesbaren
13x9-Gitterpunkten, bei `t_in=65 °C, u_gas=0.8 m/s` -- demselben Punkt, den Befund 62 DA0(a)
bereits als schlechteste Belohnung im gesamten Gitter dokumentiert (Gutsfeuchte-getrieben,
kalter/langsamer Rand, nicht die Bettemperaturgrenze).

Ursache der Abweichung von der Erwartung (-97 kW), gefunden statt vermutet:
`evaluate_operating_point` an diesem Punkt liefert `energy_target=0.8097` kW,
`moisture_violation_kg_per_kg=0.419077` (kritisch `0.2` kg/kg) UND
`solver_feasible=False` (die Loesung selbst ist unphysikalisch, nicht nur die Gutsfeuchte
zu hoch). `check_constraints`s Formel zaehlt BEIDES: `normalized_violation =
0.419077/0.2 + 1.0 = 3.09539` (der zweite Term ist der feste `+1.0`-Zuschlag fuer
`solver_feasible=False`), Strafe `= 50 * 3.09539 = 154.77` kW, plus Energie ergibt
155.579 kW. Zum Vergleich die heisse Ecke (`120 °C, 2.5 m/s`, Befund 62s bisheriger
Referenzpunkt fuer Strafgroessenordnungen): NUR eine Bettemperaturverletzung
(`bed_temperature_violation_k=18.086`, `normalized_violation=1.8086`), Strafe 90.43 kW,
Gesamt -94.93 kW -- exakt der in DA0(a) dokumentierte Wert. Die erwartete Groessenordnung
-97 kW passt zu EINER Verletzungsart (analog der heissen Ecke); der tatsaechliche
Grenzwert stammt von einem Punkt, an dem ZWEI Straftermini gleichzeitig greifen (grosse
Gutsfeuchteueberschreitung UND der pauschale Loesungs-Unphysikalitaets-Zuschlag) --
deshalb deutlich hoeher, nicht weil ein Fehler vorliegt.

Der Modus "mit Strafe" (`include_penalty=True`, Default) nutzte diesen Wert noch NICHT --
`BanditEnvConfig(solve_failure_penalty_kw=None)` faellt weiterhin auf Befund 52s Sentinel
(-1.000.000) zurueck. Kein Code-Fehler: der Override-Mechanismus ist bereits
modusunabhaengig (`test_bounded_solve_failure_penalty_overrides_sentinel` nutzt
`include_penalty=True` implizit), nur noch nicht mit diesem Wert *verwendet* worden. Neuer
Test (`test_fa1_bounded_solve_failure_penalty_with_penalty_mode`) pinnt den Wert. Der
Reproduktionsmodus (VA2, Sentinel-Default) bleibt unveraendert bitgleich
(`test_default_config_still_reproduces_va2_bitgleich`, unveraendert gruen).

**FA2 -- der Abschlusslauf.** SAC, `total_timesteps=10_000`, SB3-Default-Hyperparameter
(`learning_starts=100`), MIT Strafe (Default), Sentinel begrenzt auf FA1s
155.57896052362634 kW, drei Seeds (0, 1, 2), je einstraengig
(`MKL_NUM_THREADS=OMP_NUM_THREADS=1`), parallel. Artefakte:
`artifacts/runs/rl_bandit_sac_learning_curve_fa2_seed{0,1,2}.jsonl`, Checkpoints an jedem
Auswertungspunkt unter `artifacts/checkpoints/rl_bandit_sac_learning_curve/` (nicht
committet, siehe XA2). Belohnungsmodus und Sentinelwert stehen in jedem Record
(`include_penalty`, `solve_failure_penalty_kw`).

**FA3 -- Auswertung (`scripts/probe_bandit_env_fa3_seed_evaluation.py`,
`artifacts/runs/rl_bandit_fa3_seed_evaluation.json`).** Zulaessigkeit (a) ist an JEDEM der
12 Kombinationen aus 3 Seeds x 4 Checkpoints exakt 1.00 -- keine einzige Verletzung
irgendeiner Art (Bettemperatur/Feuchte/Gasfenster/Loeserfehler durchgehend 0), auch nicht
bei Schritt 1000. Der gepoolte Anteil (b) bleibt dagegen an JEDEM Checkpoint, JEDEM Seed
klar NEGATIV:

| Schritt | (b) Seed 0 | (b) Seed 1 | (b) Seed 2 | (b) Median | Median innen (n=2) | Median rand (n=23) |
|---|---|---|---|---|---|---|
| 1000 | -8.038 | -8.254 | -7.829 | -8.038 | -5.314 | -8.413 |
| 2500 | -8.602 | -3.072 | -3.184 | -3.184 | -2.280 | -3.311 |
| 5000 | -6.927 | -5.187 | -4.997 | -5.187 | -3.118 | -5.526 |
| 10000 | -1.999 | -2.797 | -3.012 | -2.797 | -1.712 | -2.920 |

Innen- UND Randpunkte liegen bei Schritt 10000 in aehnlicher Groessenordnung (-1.712 vs.
-2.920, beide klar negativ) -- die tanh-Randasymptote (XA3s strukturelle Erklaerung fuer
einen Rueckstand an Randoptima) ist NICHT die dominante Ursache, sonst muessten
Innenpunkte deutlich besser (idealerweise positiv) liegen. `n_interior=2` bleibt eine
duenne Stichprobe (Vorbehalt wie in XA3).

Zum Vergleich, an denselben 25 Punkten: fester Betriebspunkt (a=1.00, b=0 per
Konstruktion), Orakel (a=1.00, b=1.0000000000000004, Gleitkommarauschen), Zufall (YA1,
n=20, unveraendert wiederverwendet): a-Median 0.80 [0.68; 0.96], b-Median -6.5456
[-8.9555; -5.0803]. FA2 liegt bei Schritt 10000 auf BEIDEN Achsen klar ueber Zufall
(Zulaessigkeit 1.00 gegen Median 0.80, sogar ueber dem Zufalls-Maximum 0.96; gepoolter
Anteil -2.797 gegen Median -6.546, besser als sogar Zufalls bestes Einzelergebnis
-5.080) -- ANDERS als YA1s Befund fuer den alten Sentinel-Zustand (dort lag der Agent
UNTER Zufall auf beiden Achsen). Trotzdem bleibt der Anteil klar negativ, also schlechter
als der triviale feste Betriebspunkt.

Mittlere rohe Energie der deterministischen Politik gegen den festen Betriebspunkt
(1.179 kW, `mean_energy_agent_kw`/`_delta_kw` aus dem RunRecord, DA1-Kennzahl) bestaetigt
dasselbe Bild bei Schritt 10000: Seed 0 = 1.475 kW (+0.296), Seed 1 = 1.594 kW (+0.415),
Seed 2 = 1.625 kW (+0.446) -- Median-Delta +0.415 kW, durchgehend TEURER als der feste
Betriebspunkt, trotz perfekter Zulaessigkeit.

**Deterministische Aktionen -- Kollaps in eine ZULAESSIGE, aber teure Nische, seed-
abhaengig, mit betraechtlicher Streuung darin, wie stark der Kontext beruecksichtigt
wird.**

| Schritt | Seed | Aktion norm. (t_in, u_gas) | Streuung (t_in, u_gas) | mittl. norm. Abstand Orakel |
|---|---|---|---|---|
| 1000 | 0 | (-0.777, +0.847) | (0.037, 0.027) | 1.888 |
| 1000 | 1 | (-0.768, +0.871) | (0.053, 0.035) | 1.905 |
| 1000 | 2 | (-0.826, +0.884) | (0.020, 0.037) | 1.934 |
| 2500 | 0 | (-0.488, +0.602) | (0.019, 0.163) | 1.605 |
| 2500 | 1 | (+0.481, -0.787) | (0.144, 0.049) | 0.878 |
| 2500 | 2 | (+0.573, -0.811) | (0.108, 0.028) | 0.961 |
| 5000 | 0 | (-0.065, -0.023) | (0.129, 0.137) | 1.028 |
| 5000 | 1 | (+0.135, -0.386) | (0.418, 0.323) | 0.898 |
| 5000 | 2 | (+0.248, -0.467) | (0.417, 0.339) | 0.928 |
| 10000 | 0 | (+0.356, -0.873) | (0.181, 0.045) | 0.737 |
| 10000 | 1 | (+0.714, -0.890) | (0.041, 0.062) | 1.093 |
| 10000 | 2 | (+0.117, -0.652) | (0.414, 0.285) | 0.725 |

Bei Schritt 1000 kollabieren ALLE DREI Seeds (Streuung < 0.06 auf beiden Achsen) auf
nahezu denselben Punkt: kaltes `t_in` (~65-68 °C), hohes `u_gas` (~2.3 m/s) -- eine ANDERE
Ecke als die alte "heisse Ecke" (hoch/hoch), und -- entscheidend -- eine ZULAESSIGE: kaltes
`t_in` vermeidet die Bettemperaturgrenze, hohes `u_gas` vermeidet die Gutsfeuchte-Grenze
(die an genau diesem Aktionsrand bei niedrigem `u_gas` bindet, siehe FA1). Genau das ist
die erwartbare Kehrseite eines begrenzten statt vergiftenden Sentinels: der Agent findet
sehr frueh eine global sichere Nische und bleibt zunaechst dort, unabhaengig vom Kontext
(Streuung nahe 0) -- ein Kollaps, aber ein GUTARTIGER (in eine zulaessige, nicht in eine
unzulaessige Ecke). Ab Schritt 2500 loesen sich Seed 1/2 in eine ANDERE, ebenfalls sichere
Nische (warmes `t_in`, niedriges `u_gas`), waehrend Seed 0 laenger in der urspruenglichen
Region bleibt. Bei Schritt 10000 landet KEIN Seed-Paar in derselben Region, und die
erreichte Kontextsensitivitaet streut betraechtlich: Seed 1 bleibt praktisch kollabiert
(Streuung 0.041/0.062, kaum ueber dem Schritt-1000-Niveau), Seed 2 zeigt deutliche
Kontextabhaengigkeit (Streuung 0.414/0.285), Seed 0 liegt dazwischen. Eine Einzelseed-
Aussage (wie in WA3/XA3/BB2/CB2/DA1/EA2 durchgehend gemacht) waere hier irrefuehrend
gewesen -- drei Seeds zeigen bereits qualitativ verschiedene Endzustaende bei identischer
Konfiguration.

Der mittlere normierte Abstand zur Orakelaktion sinkt ueber das Training (Median 1.905 ->
0.961 -> 0.928 -> 0.737), bleibt aber bei Schritt 10000 immer noch gross (> 25 % der
maximal moeglichen Diagonale `2*sqrt(2)`). Physikalisch aufgeschluesselt (Median-Seed):
die Gasgeschwindigkeit naehert sich dem Orakel deutlich an (mittlere Abweichung sinkt auf
0.09-0.28 m/s, gegenueber 1.7 m/s Aktionsspanne), waehrend die Zulufttemperatur weit
entfernt bleibt (14.6-32.5 °C Abweichung, gegenueber 60 °C Spanne) -- konsistent mit einer
Politik, die eine sichere Nische besetzt statt den Kontext tatsaechlich auszuwerten.

**Kernfrage (Auftrag, ein Satz): NEIN, der Agent findet das beschraenkte Optimum weiterhin
nicht -- aber die Art der Verfehlung hat sich mit FA1 grundlegend geaendert.** Zulaessigkeit
ist kein Problem mehr (1.00 an allen 12 Kombinationen, ueber Zufall), Energieoptimierung
bleibt eines: der gepoolte Anteil ist an jedem Checkpoint jedes Seeds negativ, die
deterministische Politik sitzt in wechselnden, sicheren aber teuren Nischen statt am
Orakel.

**Vorab festgelegte Lesarten bewertet:**

- *Zulaessigkeit hoch, Anteil deutlich positiv, Aktionen nahe Orakel (Pipeline belegt)* --
  NICHT eingetreten (Anteil durchgehend negativ).
- *Zulaessigkeit hoch, Anteil nahe null (Strafdominanz, Faktor 516, bisher vom Sentinel
  verdeckt)* -- TEILWEISE, STAERKER ALS FORMULIERT: die Zulaessigkeit ist nicht nur hoch,
  sondern PERFEKT (12/12); der Anteil ist nicht nahe null, sondern klar negativ (schlechter
  als der triviale feste Betriebspunkt). YA2s Strafdominanz-Hypothese ist damit die einzige
  der drei Lesarten, die zur Richtung des Ergebnisses passt -- nur ausgepraegter: der Agent
  vermeidet die Strafe so gruendlich, dass er dabei eine Nische findet, die selbst in
  reiner Energie schlechter ist als der Referenzpunkt.
- *Kollaps in eine Ecke (weitere Ursache)* -- TEILWEISE, ABER GUTARTIG: bei Schritt 1000
  kollabieren alle drei Seeds auf (nahezu) denselben Punkt -- anders als vor FA1 ist diese
  Ecke aber ZULAESSIG, nicht gefaehrlich. Der Kollaps loest sich mit dem Training uneinheitlich
  (seedabhaengig) und nur teilweise.

Die vorab formulierte Erwartung ("Kollaps bleibt aus, Zulaessigkeit steigt deutlich, Anteil
positiv aber klar unter 1, an Randpunkten strukturell wegen tanh-Quetschung") trifft
TEILWEISE zu (kein Kollaps in eine unzulaessige Ecke mehr, Zulaessigkeit auf dem
bestmoeglichen Wert) und WIDERSPRICHT ihr TEILWEISE (der Anteil ist nicht positiv, sondern
negativ; Innen- und Randpunkte liegen aehnlich schlecht, die tanh-Asymptote erklaert den
Rueckstand nicht).

**FA4 -- Befund 62 abgeschlossen.**

*Was belegt ist:* Die Ein-Schritt-Bandit-Umgebung reproduziert Befund 52 bitgleich (VA2,
unveraendert). Die Kollapsursache (Sentinel-Vergiftung der Kritiker-Regression, EA3s
"ordinal vs. Regression"-Unterscheidung) ist mechanistisch belegt (BB1, DA0, EA1) UND durch
FA1/FA2 kausal behoben: ein modusabhaengiger, physikalisch begrenzter Sentinel beseitigt
den Kollaps in eine unzulaessige Ecke vollstaendig, auch MIT aktiver Strafe (zuvor nur ohne
Strafe gezeigt, EA2). Die in YA2 quantifizierte Strafdominanz (Faktor 516) ist jetzt, ohne
die Sentinel-Verdeckung, die sichtbare und alleinige verbleibende Erklaerung dafuer, dass
der Agent trotz perfekter Zulaessigkeit das Energieziel verfehlt. Die tanh-Randasymptote
ist NICHT die dominante Erklaerung fuer den verbleibenden Rueckstand (Innen- und
Randpunkte aehnlich schlecht).

*Was offen ist:* YA2s vier Optionen gegen die Strafdominanz (Normierung, getrenntes
Strafgewicht fuer RL, Lagrange-Verfahren, laengeres Training) -- keine gewaehlt, keine
getestet. Speicher, mehrere Linien, Netto-Zielfunktion, Anlagenmassstab bleiben unveraendert
seit (e) ausserhalb dieses Auftrags. Die in FA3 gefundene betraechtliche Seed-zu-Seed-
Streuung DARIN, welche sichere Nische ein Seed findet und wie stark er den Kontext
beruecksichtigt, ist neu und selbst nicht weiter untersucht -- ob drei Seeds ausreichen, um
sie zu charakterisieren, oder ob laengeres Training (YA2-Option 4) diese Streuung eher
vergroessert oder verkleinert, ist offen.

*Der Bandit als Pruefstand:* mit bekanntem Optimum (Befund 52), bitgleicher Reproduktion
(VA2) und einer Laufzeit von Minuten statt Stunden ist dieser Ein-Schritt-Bandit ein
guenstiger Pruefstand fuer jede kuenftige Aenderung an Belohnung oder Verfahren (Strafgewicht,
Normierung, Lagrange, Algorithmus) -- bevor eine solche Aenderung an einem teureren,
mehrschrittigen oder Anlagenmassstab-Environment erprobt wird, sollte sie hier zuerst
gegen das bekannte Optimum geprueft werden.

**y) GA1 -- JA: der Agent haelt eine grosse, konsistente Sicherheitsmarge zur
Feuchtegrenze und kauft sie mit hoeherer Gasgeschwindigkeit.** Reine Auswertung der
bereits gesicherten FA2-Checkpoints bei 10000 Schritten, kein Training
(`scripts/probe_bandit_env_ga1_safety_margin.py`,
`artifacts/runs/rl_bandit_ga1_safety_margin.json`). Das Orakel bindet die Gutsfeuchte-
grenze an 17 von 25 Kontexten (bestaetigt Befund 62(a) exakt). An genau diesen 17 Punkten,
je Seed:

| Seed | Median Feuchtemarge Agent (kg/kg, kritisch=0.2) | Agent bindet Feuchte selbst | Median Delta u_gas (Agent - Orakel) |
|---|---|---|---|
| 0 | 0.1299 | 0/25 | +0.1030 m/s |
| 1 | 0.1454 | 0/25 | +0.0449 m/s |
| 2 | 0.1395 | 0/25 | +0.5322 m/s |
| Median | 0.1395 | 0/25 (alle Seeds) | +0.1030 m/s |

An den 17 Orakel-bindenden Punkten liegt die Produktfeuchte des Agenten bei ALLEN DREI
Seeds bei ca. 0.055-0.070 kg/kg (kritisch 0.2 kg/kg) -- deutlich unter der Haelfte des
zulaessigen Werts, waehrend das Orakel per Definition exakt auf der Grenze sitzt
(Feuchtemarge ~5,2·10⁻⁸ kg/kg, Gleitkommarauschen). KEIN Seed bindet die Feuchtegrenze an
IRGENDEINEM der 25 Kontexte (0/25, durchgehend) -- der Agent naehert sich der Grenze nie,
selbst dort nicht, wo das Orakel exakt auf ihr sitzt. Die Richtung des "Erkaufens" stimmt
mit der Hypothese ueberein: an allen drei Seeds ist `u_gas` an den bindenden Punkten HOEHER
als beim Orakel (Median +0.1030 m/s ueber die Seeds, Spanne seedabhaengig +0.045 bis
+0.532 m/s) -- hoehere Gasgeschwindigkeit senkt die Gutsfeuchte (siehe Umgebungsdocstring),
der Agent zahlt also genau in der erwarteten Groesse.

Nebenbefund, praezisiert statt nur "konservativ": bei der Bettemperatur ist der Agent
NICHT durchgehend konservativer als das Orakel, sondern WENIGER konservativ -- mittlere
Marge des Agenten (alle 25 Punkte, Median ueber Seeds) 17.8-24.6 K, gegenueber dem Orakel
selbst mit 45.4 K (Median) -- der Agent laesst also, obwohl er niemals eine Verletzung
erzeugt (0 Verletzungen an jedem Checkpoint, FA3), spuerbar WENIGER Abstand zur 80-°C-
Grenze als das Orakel, waehrend er zur Feuchtegrenze MEHR Abstand haelt. Das ist mit dem
Mechanismus vereinbar, nicht dagegen: die Feuchtegrenze bindet an 17/25 Orakelpunkten,
die Bettemperaturgrenze an 0/25 (Befund 62(a)) -- ein strafvermeidender Agent lernt vor
allem, Abstand zu der Nebenbedingung zu halten, die am Optimum tatsaechlich aktiv ist,
nicht zu jeder Nebenbedingung gleichermassen.

**Einordnung (Auftrag): als Mechanismus festgehalten.** Das Optimum liegt an der Mehrheit
der Kontexte (17/25) exakt auf der Gutsfeuchte-Nebenbedingungsgrenze; ein strafbasierter
Agent (Straffunktion statt Projektion oder Lagrange, siehe VA1) haelt systematisch Abstand
zu genau dieser Grenze und bezahlt dafuer mit hoeherer Gasgeschwindigkeit -- der
verbleibende, in FA3 gemessene negative gepoolte Anteil ist damit zu einem wesentlichen
Teil der PREIS DIESER SICHERHEITSMARGE, nicht ein ungerichtetes Verfehlen. Das ist
praeziser als "Strafdominanz" (YA2) allein und zeigt konkret, warum ein Verfahren fuer
beschraenktes RL, das auf der Nebenbedingungsgrenze sitzen darf (Lagrange, YA2-Option 3),
der naheliegende naechste Schritt waere -- eine Straffunktion kann eine Grenze per
Konstruktion nur mit einem Sicherheitsabstand ungleich null respektieren, ein Lagrange-
Multiplikator kann exakt auf ihr balancieren.

**Einordnung der Metrik (Auftrag).** Der Nullpunkt von `pooled_closed_regret_fraction` ist
PER KONSTRUKTION der feste Befund-52-Betriebspunkt (siehe `evaluation.py`) -- und dessen
eigener Regret gegenueber dem Orakel ist bereits klein: Befund 52/62(b) berichtet einen
Median-Regret von 16,03 % relativ zum Orakel (P95 22,20 %, Max 25,80 %), der feste
Betriebspunkt ist also selbst schon nah am Optimum, nicht "keine Optimierung". Zulaessige
Zufallsaktionen (YA1) erreichen dagegen einen gepoolten Anteil von -6,55 (Median) -- eine
VIEL groessere Distanz zum Orakel als der feste Betriebspunkt selbst. Ein negativer
gepoolter Anteil bedeutet deshalb "noch nicht besser als ein bereits guter konstanter
Sollwert", nicht per se "schlechter als Zufall" oder allgemeines Versagen -- FA3s Agent
liegt bei Schritt 10000 (Median -2,80) klar zwischen diesen beiden Referenzen, naeher am
festen Betriebspunkt als am Zufall.

**z) HA1 -- Nachtrag: GA1s Sicherheitsmarge im Verhaeltnis zum Vorhersagefehler des
Feuchte-Softsensors, getrennt nach Trainingsart.** Anlass: bei der Vorbereitung der
M2-Synthese wurde nach einer Gegenueberstellung "Marge des Agenten gegen die durch den
Vorhersagefehler gerechtfertigte Marge" gefragt -- diese Rechnung stand bisher an keiner
Stelle im Laborbuch. Reine Auswertung, kein neuer Lauf, kein Training
(`scripts/probe_bandit_env_ha1_margin_vs_prediction_error.py`,
`artifacts/runs/rl_bandit_ha1_margin_vs_prediction_error.json`). Verknuepft drei bereits
committete Artefakte, ohne eines davon neu zu berechnen: GA1s Marge
(`rl_bandit_ga1_safety_margin.json`), den Vorhersagefehler des ANGEREGT trainierten
LSTM-Softsensors (Befund 51/58, der fuer den Feuchte-Softsensor empfohlene Trainingsfall,
`arch_learning_curve_projected.jsonl`) und den des KONSTANT trainierten LSTM-Softsensors
(Befund 55, reine Betriebsdaten, `arch_learning_curve_constant_trained.jsonl`).

**Konvention.** Sicherheitsfaktor = Marge / (3 x RMSE). Drei Standardabweichungen ist eine
in der Statistik gebraeuchliche, hier aber bewusst neu eingefuehrte Konvention -- keine
Anlagen- oder Prozessgrenze, kein Bezug zu einer Norm oder einer Vorgabe des
Anwendungspartners.

**Herkunft der Marge (0,1395 kg/kg).** `rl_bandit_ga1_safety_margin.json`,
`median_across_seeds/moisture_margin_agent_binding_subset_kg_per_kg`: je Seed der Median
der Agenten-Feuchtemarge (`critical_moisture - moisture`) ueber genau die 17 von 25
Befund-52-Kontexten, an denen das Orakel die Gutsfeuchtegrenze bindet; berichtet wird der
Median dieser drei Seed-Mediane (Seed 0: 0,1299; Seed 1: 0,1454; Seed 2: 0,1395 kg/kg) --
**0,139545 kg/kg**.

**Vorhersagefehler, `product_moisture_kg_per_kg`, LSTM, `n_train=64`,
`test_set=constant_holdout`, h=36, `rmse_run_median_physical`, Median ueber 5 Seeds:**

| Softsensor-Training | Einzelwerte ueber 5 Seeds [kg/kg] | Median [kg/kg] | 3x Median [kg/kg] | Marge/(3xRMSE) |
|---|---|---|---|---|
| Angeregt (Befund 51/58, empfohlener Fall) | 0,000664 / 0,000426 / 0,000543 / 0,000486 / 0,000782 | **0,000543** | 0,001628 | **85,7** |
| Betriebsdaten, konstant (Befund 55) | 0,000695 / 0,002319 / 0,001310 / 0,001488 / 0,001681 | **0,001488** | 0,004465 | **31,3** |

**Einordnung.** In beiden Faellen liegt die vom Agenten gehaltene Sicherheitsmarge weit
(Faktor 31 bis 86) ueber dem Dreifachen des jeweiligen Modellfehlers -- selbst mit dem
deutlich schwaecheren, aus reinen Betriebsdaten trainierten Softsensor (Befund 55) waere
die Marge des Agenten im Modell mehr als ausreichend, um den Vorhersagefehler abzudecken.
Das ist eine Aussage ueber DIESES Modell (Grey-Box-Simulator, Ein-Schritt-Bandit-Umgebung),
nicht ueber die reale Anlage -- an ihr ist die Produktfeuchte nicht gemessen (Befund 56),
ein realer Vorhersagefehler des Softsensors an der Anlage ist damit unbekannt.

**Vorbehalte.**
- `critical_moisture_kg_per_kg` (0,20 kg/kg) ist ein unbelegter Laborplatzhalter
  (Befund 52a) -- sowohl die Marge selbst als auch ihre Einordnung als "ausreichend" gelten
  nur relativ zu diesem Wert.
- Die 3xRMSE-Konvention ist eine in dieser Auswertung neu eingefuehrte, nicht aus dem
  Projektauftrag oder einer Anlagenanforderung abgeleitete Wahl.
- `rmse_run_median_physical` ist ein Median ueber 5 Seeds, kein Maximalfehler -- die
  Streuung ueber Seeds/Laeufe ist in Befund 51/55 dokumentiert, hier nicht erneut
  herangezogen.
- Die Marge selbst (GA1) stammt aus einem Ein-Schritt-Bandit mit den in Befund 62(e)/GA1
  genannten Einschraenkungen (kein Speicher, ein Prozess, Laborplatzhalter-Maßstab,
  Straf- statt Lagrange-Formulierung) -- keine Aussage ueber eine mehrschrittige oder
  Anlagen-RL-Politik.

### 63. Auflösungsvermögen der tatsächlich gefahrenen 5 Seeds auf den Achtergleich (Befund
51/58) angewendet -- GRU/LSTM durchgehend nicht unterscheidbar, Transformer-Parität bei
`bed_temperature_celsius` war nie ein auflösbarer Unterschied; LSTMs Geschwindigkeitsvorteil
gegen GRU durch Modellgröße nicht erklärt

**Anlass.** Zwei offene Punkte aus der Vorbereitung der M2-Synthese
(`docs/08_synthese_m2_bestand.md`): (1) Tabelle C dort zeigt LSTM durchgehend schneller als
GRU (182,8 gegen 323,1 µs/Fenster, n_train=64) -- unerklärt stehengelassen, hier zu prüfen,
ob die Modellgröße das erklärt. (2) Die M2-Definition selbst nennt ein Auflösungsvermögen
("mit drei Seeds sind Unterschiede unter 10-15 % nicht nachweisbar") als eigenständig zu
berichtenden Punkt -- bisher wurde diese Schwelle nur für die ursprüngliche
Varianzzerlegungs-Studie (Schritt 5, drei Seeds) berechnet, nie auf den tatsächlichen
Achtergleich (Befund 51/58, fünf Seeds) angewendet. Reine Auswertung, kein neuer Lauf, kein
Training (`scripts/analyze_m2_resolution_verdicts.py`,
`artifacts/runs/analyze_m2_resolution_verdicts.json`).

**a) Geschwindigkeitsfrage -- Modellgröße erklärt den Unterschied NICHT.**
`model/n_parameters`/`model/net_hidden_size` aus den Records gelesen, nicht geschätzt: GRU
und LSTM haben bei `n_train=64` dieselbe `hidden_size` (64), aber LSTM hat **33 % mehr
Parameter** (57027 gegen 42819, zweiter Speicherzustand plus vierter statt dritter
Gattersatz -- konsistent mit dem bereits in Befund 30 notierten "LSTMs zweiter Speicher
(Faktor 1,33 mehr Parameter)"). Eine größere Parameterzahl bei gleicher Kapazitätsstufe
würde eher eine LANGSAMERE Inferenz erwarten lassen, nicht die gemessene 1,77x schnellere
(182,8 gegen 323,1 µs/Fenster). **Die Modellgröße erklärt den Geschwindigkeitsunterschied
nicht -- im Gegenteil, sie legt das Gegenteil des gemessenen Ergebnisses nahe.** Als
unerklärte Beobachtung stehen gelassen, nicht weiter diagnostiziert (kein Profiling der
`nn.GRU`/`nn.LSTM`-CPU-Kernel in dieser Sitzung).

**b) Auflösungsvermögen aus der beobachteten 5-Seed-Streuung, je Kanal/Testmenge, bei
`n_train=64`.** Methodik: `SeedSpread.relative_minimum_detectable_difference`
(`evaluation/resolution.py`, spannweitenbasiert, Hartley-`d2(5)=2,326`, 95 % Konfidenz,
Faktor `sqrt(2/5)` für den Vergleich zweier unabhängig streuender Größen), angewendet auf
LSTMs eigene Seed-Streuung -- **nicht** auf Persistenz. Grund: auf dem
Konstantlauf-Holdout ist Persistenz über alle fünf Seeds bitgleich (der Holdout ist
unabhängig vom `master_seed` fest, Persistenz hat keine trainierbaren Parameter) -- ihre
Streuung ist dort konstruktionsbedingt Null und liefert keine Schwelle. LSTM ist zusätzlich
die Referenz in jeder hier geprüften Aussage. Primärmetrik `rmse_run_median_physical`
(nicht `nrmse_run_median`): alle geprüften Aussagen sind bereits in dieser Metrik
formuliert, ein Wechsel hätte eine neue Inkonsistenz eingeführt statt eine zu klären.

Auflösungsschwelle (relativ, aus LSTMs Streuung, n=5, 95 % Konfidenz):

| Kanal | angeregt | Konstantlauf |
|---|---|---|
| `energy_target` | 9,1 % | 4,9 % |
| `product_moisture_kg_per_kg` | 35,7 % | 35,0 % |
| `bed_temperature_celsius` | 5,1 % | 8,1 % |

Durchgehend in derselben Größenordnung wie die in der M2-Definition genannten 10-15 %
(mit drei Seeds) -- bei fünf statt drei Seeds erwartungsgemäß enger bei den beiden
gutmütigeren Kanälen, bei `product_moisture_kg_per_kg` deutlich weiter (hohe
Seed-zu-Seed-Streuung selbst bei LSTM, konsistent mit dem im ganzen Dokument als
schwierigstem Kanal beschriebenen Feuchteziel).

**c) GRU gegen LSTM: in ALLEN sechs geprüften Zellen (3 Kanäle × 2 Testmengen) unterhalb
der Auflösungsschwelle.**

| Kanal | Testmenge | GRU vs. LSTM | Schwelle | Über Auflösung? |
|---|---|---|---|---|
| `energy_target` | angeregt | −1,8 % | 9,1 % | **Nein** |
| `energy_target` | Konstantlauf | +2,0 % | 4,9 % | **Nein** |
| `product_moisture_kg_per_kg` | angeregt | −19,2 % | 35,7 % | **Nein** |
| `product_moisture_kg_per_kg` | Konstantlauf | −19,9 % | 35,0 % | **Nein** |
| `bed_temperature_celsius` | angeregt | +0,3 % | 5,1 % | **Nein** |
| `bed_temperature_celsius` | Konstantlauf | +3,5 % | 8,1 % | **Nein** |

**Kein einziger der sechs GRU/LSTM-Vergleiche ist mit fünf Seeds auflösbar.** Die im
gesamten Dokument verstreute Formulierung "GRU/LSTM bleiben die besten Modelle" bleibt
richtig (beide klar vor Transformer/xLSTM/Mamba, s. u.), aber eine Rangfolge GRU-vor-LSTM
oder LSTM-vor-GRU ist mit dieser Seedzahl nicht zu belegen. **Korrekte Formulierung:
"GRU oder LSTM", nicht eines von beiden als Empfehlung.**

**d) Transformer gegen LSTM: kanalabhängig, eine der beiden Prüfungen ändert die bisherige
Einordnung.**

| Kanal | Epochen | Faktor (Text/Befund 58f./59c) | Verdikt | Schwelle | Über Auflösung? |
|---|---|---|---|---|---|
| `bed_temperature_celsius`, Konstantlauf | 200 | 1,026 | −2,6 % | 8,1 % | **Nein** |
| `bed_temperature_celsius`, Konstantlauf | 600 | 1,000 | +0,0 % | 9,3 % | **Nein** |
| `energy_target`, Konstantlauf | 200 | 1,082 | −8,2 % | 4,9 % | **Ja, schlechter** |
| `energy_target`, Konstantlauf | 600 | 1,158 | −15,8 % | 10,1 % | **Ja, schlechter** |
| `bed_temperature_celsius`, angeregt | 200 | -- | −16,5 % | 5,1 % | **Ja, schlechter** |
| `product_moisture_kg_per_kg`, beide Testmengen | 200 | -- | −45,9 % / −98,6 % | 35,0 %/35,7 % | **Ja, schlechter** |

**Korrektur der Einordnung aus Befund 58/59c:** die dort als "Aufholen auf Parität" bei
`bed_temperature_celsius` beschriebene Entwicklung (Faktor 1,026 bei 200 Epochen → 1,000
bei 600 Epochen) war **zu keinem Zeitpunkt ein auflösbarer Unterschied** -- schon der
200-Epochen-Wert (−2,6 %) lag unterhalb der Schwelle (8,1 %). Die bei 600 Epochen erreichte
"exakte Parität" ist damit eine **Bestätigung der von Anfang an geltenden
Ununterscheidbarkeit, keine neue Erkenntnis durch das größere Epochenbudget** -- das
Epochenbudget hat nichts aufgelöst, was vorher unaufgelöst war, weil auf diesem Kanal nie
ein auflösbarer Unterschied bestand. Anders bei `energy_target`: dort ist der
Transformer-Rückstand **in beiden Konfigurationen klar auflösbar** (200 Epochen bereits
−8,2 % gegen 4,9 % Schwelle, 600 Epochen −15,8 % gegen 10,1 % Schwelle) -- die im
ursprünglichen Auftrag als "an der Grenze" vermutete 1,158-Zahl liegt tatsächlich **deutlich
über** der Auflösungsschwelle (Faktor ~1,6 über der Schwelle), nicht an ihr. Auf
`product_moisture_kg_per_kg` und bei `bed_temperature_celsius` auf der angeregten Testmenge
bleibt der Transformer-Rückstand ebenfalls klar auflösbar.

**e) "xLSTM und Mamba bleiben deutlich zurück" -- in allen zwölf geprüften Zellen bestätigt
auflösbar.** xLSTM vs. LSTM und Mamba vs. LSTM, je 3 Kanäle × 2 Testmengen (12 Zellen
gesamt): **12 von 12 nachweisbar schlechter**, mit teils extremen Verhältnissen (z. B.
xLSTM `energy_target` angeregt: LSTM-Median 0,0143 kW gegen xLSTM-Median 0,428 kW, mehr als
Faktor 29 -- getrieben von der bereits in Befund 58 (OA5) dokumentierten divergierenden
xLSTM-Optimierung auf diesem Kanal, kein neuer Befund, nur zahlenmäßig bestätigt). Die
Aussage "bleiben deutlich zurück" ist damit durchgehend belegt, nicht nur an Einzelpunkten.

**f) Vorbehalte.**
- Referenz für die Schwelle ist LSTMs eigene Seed-Streuung, nicht eine parameterfreie
  Baseline (Persistenz) wie in der ursprünglichen Methodik (`resolution.py`-Moduldocstring)
  vorgesehen -- Persistenz ist auf dem Konstantlauf-Holdout konstruktionsbedingt streuungsfrei
  und liefert dort keine Schwelle. Diese Abweichung von der Standardmethodik ist eine
  bewusste, hier offen benannte Entscheidung, kein Fehler.
- Fünf Seeds bleiben fünf Seeds: die Auflösungsschwelle selbst ist bei so wenigen Replikaten
  eine Schätzung mit einer Unsicherheit ähnlicher Größenordnung (siehe
  `resolution.py`-Moduldocstring) -- als Größenordnung zu lesen, nicht als exakter Grenzwert.
- Geprüft wurde ausschließlich `n_train=64` (die Stützstelle der bereits im Dokument
  zitierten Aussagen) -- keine Aussage über andere Budgetpunkte.
- Bei `product_moisture_kg_per_kg` ist die Schwelle selbst groß (35 %), weil bereits LSTM
  dort stark über die Seeds streut -- ein GRU/LSTM-Unterschied könnte auf diesem Kanal auch
  mit deutlich mehr Seeds schwer auflösbar bleiben, das ist hier nicht weiter untersucht.

**g) Nachtrag (Korrektur, extern angestoßen): Zwei-Stichproben-Schwelle statt allein aus
LSTMs Streuung -- ändert zwei Verdikte bei xLSTM.** Anlass: die Erstfassung von (b)-(e)
verwendete durchgehend LSTMs eigene Seed-Streuung als Schwelle
($\Delta_{\min}=z\,\hat\sigma_{\mathrm{LSTM}}\sqrt{2/n}$) -- das unterstellt implizit
gleiche Streuung auf Kandidaten- und Referenzseite. Bei einer Architektur mit deutlich
größerer eigener Streuung ist das nicht gegeben. Korrigiert auf die Zwei-Stichproben-Form

$$\Delta_{\min} = z_{0{,}95}\sqrt{\frac{\hat\sigma_{\mathrm{Kandidat}}^2+
\hat\sigma_{\mathrm{Referenz}}^2}{n}}$$

(`scripts/analyze_m2_resolution_verdicts.py`, Korrektur-Commit `47c5292`,
`artifacts/runs/analyze_m2_resolution_verdicts.json`, neu geschrieben).

**Ergebnis (c)/GRU vs. LSTM: unverändert.** Alle sechs Zellen bleiben unterhalb der
(jetzt etwas anders berechneten) Schwelle -- die Empfehlung "GRU oder LSTM" ist gegen
diese Korrektur robust.

**Ergebnis (d)/Transformer vs. LSTM: Kernaussage unverändert, eine Zelle jetzt knapper.**
`bed_temperature_celsius` bleibt bei 200 UND 600 Epochen nicht auflösbar (Schwellen jetzt
8,3 %/7,5 % statt 8,1 %/9,3 % -- Verdikt unverändert). `energy_target` bleibt in beiden
Konfigurationen klar auflösbar (200 Epochen: Schwelle 4,7 % statt 4,9 %; 600 Epochen:
Schwelle 9,8 % statt 10,1 % -- Verdikt unverändert). `product_moisture_kg_per_kg`,
Konstantlauf, ist mit der korrigierten Schwelle nur noch **knapp** auflösbar (−45,9 %
gegen 45,5 % Schwelle, vorher 35,0 %) -- Transformers eigene Streuung auf diesem Kanal ist
deutlich größer als LSTMs, was die Schwelle von 35,0 % auf 45,5 % anhebt. Verdikt bleibt
"auflösbar", aber knapper als zuvor berichtet.

**Ergebnis (e)/xLSTM und Mamba: 10 von 12 statt 12 von 12 -- zwei Zellen kippen, beide
xLSTM auf `product_moisture_kg_per_kg`.** Auf diesem Kanal ist xLSTMs eigene
Seed-zu-Seed-Streuung so groß (relative $\hat\sigma$ deutlich über LSTMs), dass der
Medianunterschied (−140,2 % angeregt, −113,6 % Konstantlauf) trotz seiner Größe **nicht**
mehr über der neuen, entsprechend breiteren Schwelle liegt (357,3 % bzw. 147,8 %). Mamba
bleibt in allen sechs Zellen auflösbar, xLSTM in den übrigen vier (`energy_target`,
`bed_temperature_celsius`, je beide Testmengen). **Korrigierte Formulierung:** "xLSTM und
Mamba bleiben deutlich zurück" gilt auflösbar belegt für 10 von 12 geprüften Zellen: Mamba
durchgehend, xLSTM auf `energy_target` und `bed_temperature_celsius`. Auf
`product_moisture_kg_per_kg` ist xLSTMs Rückstand selbst -- trotz des großen
Medianunterschieds -- **statistisch nicht belastbar**, weil die Architektur auf genau
diesem Kanal bereits in Befund 58 (OA5) als divergierend optimierend dokumentiert ist: die
extreme eigene Streuung ist ein Symptom des Optimierungsversagens, nicht eine unabhängige,
zusätzliche Information über die Kapazitätsgrenze der Architektur. Die Aussage ist deshalb
als Optimierungsversagen unter festem Protokoll zu lesen, nicht als belastbare
Kapazitätsaussage auf diesem einen Kanal.

**h) Wie viele Seeds bräuchte `product_moisture_kg_per_kg`, um die M2-Zielschwelle von
12,5 % zu erreichen?** `SeedSpread.required_replicates(0{,}125, confidence=0{,}95)`,
LSTM-Streuung, $n_{\mathrm{train}}=64$: **41 Seeds** (angeregt), **40 Seeds**
(Konstantlauf) -- Größenordnung, nicht exakter Wert (Skalierung mit $1/\sqrt{n}$, selbst
mit einer bei 5 Replikaten unsicheren Streuungsschätzung). Zum Vergleich, mit der
korrigierten Zwei-Stichproben-Schwelle aus (g): für die Vergleiche zwischen GRU, LSTM und
Transformer liegt die mit 5 Seeds bereits erreichte Schwelle bei `energy_target`
(3,9-10,5 %) und `bed_temperature_celsius` (5,5-11,1 %) **unterhalb oder nahe** der
12,5-%-Zielmarke -- dort ist die Auflösung der M2-Definition für diese drei Architekturen
im Wesentlichen erreicht. Für xLSTM/Mamba liegt die Schwelle auf denselben beiden Kanälen
deutlich höher (17,2-1189,7 % je nach Kandidat und Testmenge) -- ihre eigene Streuung ist
zu groß, um dieselbe Auflösung zu erreichen. Auf `product_moisture_kg_per_kg` liegt die
Schwelle selbst für GRU/LSTM/Transformer bei 34,8-91,0 %, für xLSTM/Mamba bei
142,0-357,3 % -- durchgehend über der Zielmarke.

**i) Notiz (Quelle: E-Mail, Ansprechpartner beim Anwendungspartner, 16.09.2026, extern angestoßen): Anlage
KR-1 des Anwendungspartners verfügt über eine kapazitive In-situ-Feuchtemessung.** Nicht
Anlage F-5 (Befund 56) -- eine andere Anlage desselben Partners. Keine Daten, keine
Messwerte vorliegend; Übertragbarkeit zwischen den Anlagen (Stoffsystem, Baugröße,
Messstellenposition) nicht geprüft. Als mögliche künftige Grundlage festgehalten, den
Feuchte-Soft-Sensor (Abschnitt 3 der M2-Synthese) gegen eine reale In-situ-Messung zu
prüfen -- an KR-1, nicht an F-5. Reine Notiz, kein Ergebnis, keine Berechnung.

### 64. Cost of a safety margin at the moisture limit -- most of Befund 52's control benefit survives at a realistic soft-sensor margin, but the RL agent's own margin falls outside what the model can even represent

**a) Purpose and scope.** Befund 62 (GA1/HA1) showed that a penalty-based RL agent holds a
large, consistent safety margin to the moisture limit (median 0.139545 kg/kg, GA1) -- 31 to
86 times larger than the 3x-RMSE margin that the moisture soft sensor's own prediction error
would justify (HA1). Open question: what does a safety margin like that cost in energy, and
how much of Befund 52's control benefit survives once a controller has to keep one at all?

No reinforcement learning, no network training. A controller that estimates moisture with
uncertainty sigma and keeps a margin `m = 3 sigma` operates the plant (under a perfect
process model) exactly like an optimiser working against a tightened constraint
`critical_moisture - m` -- so the energy at that tightened constraint is the per-instance
oracle there. Implemented in `scripts/probe_margin_energy_cost.py`
(`artifacts/runs/probe_margin_energy_cost.json`), reusing the unchanged optimiser mechanics
of `study_operating_point_regret.py` (via `importlib`, same pattern as Befund 54 DA1's
`probe_critical_moisture_sensitivity.py`) and the same 3x3 grid/budget/seed as DA1. Measured
runtime: 33.3 min for the five newly computed boundaries (well under the 1-2 h estimate,
because KA1, unlike DA1, only reruns the per-point oracle (BB2), not the fixed-point search
(BB3(a)) -- the fixed operating point never sees a moisture measurement and therefore never
adapts to a margin, so it is reused unchanged from Befund 52/54 at 0.20 kg/kg). `git/dirty`
was `False` throughout; script committed (`e7d17c2`) before the run.

**b) KA1 -- seven effective boundaries, two reused, five newly computed.** Seven values of
`critical_moisture_kg_per_kg`, descending from the true limit (0.20 kg/kg) toward the RL
agent's own margin:

| Boundary | `critical_moisture` [kg/kg] | margin [kg/kg] | source of the margin | status |
|---|---|---|---|---|
| oracle anchor | 0.20000 | 0 | -- | reused from Befund 54 DA1 (bit-identical) |
| excited-trained LSTM | 0.19837 | 0.001628 | 3x RMSE, Befund 62 HA1 (excited-trained soft sensor, Befund 51/58) | newly computed |
| constant-trained LSTM | 0.19553 | 0.004465 | 3x RMSE, Befund 62 HA1 (constant-trained soft sensor, Befund 55) | newly computed |
| grid point | 0.18000 | 0.02 | -- (interpolation point) | newly computed |
| grid point | 0.15000 | 0.05 | -- (interpolation point) | reused from Befund 54 DA1 (bit-identical) |
| grid point | 0.12000 | 0.08 | -- (interpolation point) | newly computed |
| RL-agent margin | 0.06045 | 0.139545 | Befund 62 GA1 (median agent margin over 3 seeds) | newly computed -- **infeasible at all 9 grid points** |

The two reused boundaries were checked, not assumed, reusable: the script asserts the base
config fingerprint, the search budget (`popsize`, `maxiter`, `de_tol`,
`search_max_iterations`, `search_tolerance`, `seed`), and the grid coordinates against the
committed DA1 artifact before skipping the recomputation; all three assertions passed
silently (no exception), and the run completed without error. The `oracle_anchor` row is
therefore not an independent reproduction of DA1 -- it is the same data object -- but the
assertions are the actual verification that reuse was valid.

The margins for the two LSTM cases and the RL agent were read programmatically from the
already-committed `rl_bandit_ha1_margin_vs_prediction_error.json` (which in turn cites
`rl_bandit_ga1_safety_margin.json` for the agent's margin), at full floating-point precision
-- not retyped from the rounded display values in the task (0.1984/0.1955/0.0605 kg/kg),
which are simply this artifact's numbers rounded to four decimals.

**RL-agent margin: infeasible everywhere, not a search failure.** All 9 grid points failed
in 0.1 s each, with the identical reason regardless of ambient temperature/humidity:

> `critical_moisture_kg_per_kg (0.06045...) liegt nicht über dem Maximum der GAB-Isotherme
> im Sicherheitsraster (0.08891 kg/kg, phi in [0.0008, 0.99], T in [-5.0, 130.0] °C) --
> normalised_moisture würde eine negative Spanne (X_cr - X_eq) sehen.`

This is `sorption.py::validate_isotherm_against_critical_moisture`, a pre-existing,
documented guard (unrelated to this study) that requires `critical_moisture_kg_per_kg` to
exceed the GAB isotherm's maximum equilibrium moisture over the model's entire safety
envelope (not just the current disturbance grid point) -- otherwise the normalised-moisture
variable used throughout the drying physics would see a non-positive span. The bound
(0.08891 kg/kg) is a property of the isotherm parameterisation and the action/disturbance
envelope, identical at all 9 grid points -- not something the DE/refine search discovered
per point. **The RL agent's own margin, taken literally as a tightened moisture ceiling, asks
for a value (0.06045 kg/kg) that this grey-box model cannot represent at all, anywhere in its
operating envelope** -- a stronger statement than "expensive," and the reason every one of
the 45 objective evaluations per grid point returned the solve-failure sentinel immediately
rather than after a real search.

**c) KA2 -- cost of the margin and remaining control benefit.** Per boundary, over the 9
grid points that are feasible at that boundary: cost = `E_oracle(boundary) -
E_oracle(0.20)` (absolute and relative), and remaining fraction of Befund 52's control
benefit = `(E_fixed - E_oracle(boundary)) / (E_fixed - E_oracle(0.20))`, with `E_fixed` the
unchanged Befund 52/54 fixed operating point at 0.20 kg/kg. Aggregated two ways, as in
Befund 62's `pooled_closed_regret_fraction`: the median of the per-point ratios, and pooled
(sum of numerators over sum of denominators) -- pooling matters here for the same reason as
in Befund 62: at points where the moisture limit does not bind at 0.20 kg/kg, `E_fixed -
E_oracle(0.20)` is small, and a single such point can otherwise dominate a median or a mean
of ratios.

| Boundary | feasible | pooled cost [%] | pooled remaining, all 9 | pooled remaining, 6 binding points | pooled remaining, 3 non-binding points |
|---|---|---|---|---|---|
| oracle anchor | 9/9 | 0.000 | 1.000 | 1.000 | 1.000 |
| excited-trained LSTM | 9/9 | 0.054 | 0.996 | 0.997 | 0.962 |
| constant-trained LSTM | 9/9 | 0.174 | 0.987 | 0.988 | 0.912 |
| grid 0.18 | 9/9 | 0.824 | 0.937 | 0.944 | 0.611 |
| grid 0.15 | 9/9 | 6.025 | 0.537 | 0.549 | 0.015 |
| grid 0.12 | 9/9 | 18.83 | -0.447 | -0.443 | -0.592 |
| RL-agent margin | 0/9 | -- | -- | -- | -- |

(Median-of-ratios values are in the artifact alongside the pooled ones; they agree with the
pooled numbers in sign and order of magnitude at every boundary except grid 0.12, where the
per-point ratios range from +0.87 to -3.87 within the 6 binding points and include one
non-binding point at -308.7 (a near-zero `E_fixed - E_oracle(0.20)` denominator there) --
median and pooled disagree in sign in both the "all 9" and the "binding" groups as a result.
Exactly the small-denominator sensitivity the task anticipated, and the reason pooled values
are reported as primary throughout.)

**Core question, one sentence each:** at the excited-trained soft sensor's own margin,
99.6% of Befund 52's control benefit survives (pooled, all 9 points); at the considerably
weaker constant-trained (operating-data-only) soft sensor, 98.7% survives -- both realistic
soft-sensor margins cost close to nothing. The benefit only degrades materially from the
0.18 kg/kg grid point onward (93.7% remaining) and turns negative somewhere between the
0.15 kg/kg (53.7% remaining) and 0.12 kg/kg (-44.7%, i.e. worse than the fixed point) grid
points -- both margins far larger than either soft sensor's, but still smaller than the RL
agent's own.

**Does a small margin cost nothing where moisture doesn't bind at 0.20 kg/kg? Checked --
mostly yes, but only while the margin stays small.** At the two smallest, soft-sensor-scale
margins, the non-binding-point cost is indeed far smaller than the binding-point cost
(excited: 0.038% vs. 0.060%; constant: 0.089% vs. 0.210%) -- consistent with the expectation.
By the 0.15 kg/kg grid point (margin 0.05 kg/kg, roughly 11-31x larger than the soft-sensor
margins), that stops holding: the non-binding pooled remaining fraction (0.015) has nearly
collapsed even though the pooled cost there (1.00%) is still an order of magnitude below the
binding group's (8.13%) -- the near-zero denominator at non-binding points (Befund 52c: the
fixed point already sits close to the unconstrained optimum there) makes the *ratio*
sensitive well before the *cost* becomes large. "Small margin, no cost" holds for margins at
the soft-sensor scale; it is not a general property of non-binding points once the margin
grows to grid-point scale.

**d) KA3 -- counter-check against the RL agent, pre-registered.** Expected: a clearly
negative pooled remaining fraction at the agent's own margin, matching the order of
magnitude of the -2.80 pooled `closed_regret_fraction` that Befund 62 FA3 measured for the
agent itself at 10,000 training steps (median over 3 seeds, 5x5 grid, 25 contexts). That
comparison cannot be made directly: at the agent's literal margin (0.06045 kg/kg), every one
of the 9 grid points is infeasible for the reason given in (b) -- the boundary lies below
what this model's isotherm can represent at all, not merely below what is economically
reachable. This is a stronger failure mode than a finite negative number, not a weaker one.

The nearest computable point is the 0.12 kg/kg grid point (margin 0.08 kg/kg -- still 43%
smaller than the agent's actual 0.139545 kg/kg): its pooled remaining fraction is already
clearly negative (-0.447, all 9 points; -0.443 restricted to the 6 points where moisture
binds at 0.20 kg/kg). That is the same sign and a comparable order of magnitude to FA3's
-2.80, though about 6x smaller in magnitude -- and this is at a margin still smaller than
the agent's own. **Reading (pre-registered in the task): the margin mechanism is directionally
sufficient to explain FA3's negative finding -- a margin well short of the agent's own already
produces a worse-than-fixed-point outcome -- but no precise magnitude match can be
established, because the agent's actual margin sits beyond the range in which this
oracle-based method can be evaluated at all.** The caveat noted in advance applies in full:
the agent does not hold a uniform margin at every point (GA1: seed range 0.1299-0.1454
kg/kg), so this was always an order-of-magnitude check, not an exact reconciliation.

**e) Caveats (as specified in the task, none newly discovered).**
- Perfect process-model knowledge assumed; only the moisture estimation error is
  represented as a margin. A real soft sensor also has other error sources (Befund 56: no
  plant moisture measurement exists to validate against).
- The prediction error behind the 3x-RMSE margins is assumed unbiased and Gaussian; 3 sigma
  is a convention introduced in Befund 62 HA1, not a plant or standard requirement.
- The fixed operating point holds no margin at all -- fair, because it is determined offline
  against the true model (Befund 52), but worth stating plainly: it is not a policy that
  could exist without perfect foresight of the disturbance grid either.
- `critical_moisture_kg_per_kg` (0.20 kg/kg) remains an unsourced placeholder (Befund 52a);
  every number in this entry is relative to that placeholder, including the GAB-isotherm
  floor (0.08891 kg/kg) that made the RL-agent-margin boundary infeasible.
- Laboratory scale, gross energy consumption (no heat-network recovery credit, same scope
  limitation as Befund 52).

**f) Addendum (2026-09-28): sections (b)-(d) above conflated the drying-kinetics parameter
with the quality-constraint threshold -- corrected below.** Raised externally, before any
of the numbers above were relied on elsewhere: Befund 62 GA1 measured the RL agent reaching
product moisture 0.055-0.070 kg/kg in this same model, yet (b) reported the RL-agent-margin
boundary (0.0605 kg/kg) as infeasible at all 9 grid points. Both cannot be true at once
unless KA1 changed more than the boundary -- and it did.

*What was wrong.* `material.critical_moisture_kg_per_kg` has a double role in this model,
confirmed by reading every use of the field:

- `thermo.normalised_moisture`: `span = X_cr - X_eq`, `η = clip((X_p - X_eq)/span, 0, 1)` --
  the drying-kinetics falling-rate-period transition, used inside `solve_steady_state`
  itself (`config.py`'s own comment on the field: "Trocknungskinetik: normierte
  Trocknungskurve, linear fallende Rate").
- `sorption.py::validate_isotherm_against_critical_moisture`: a config-construction guard
  requiring `X_cr` to exceed the GAB isotherm's maximum equilibrium moisture over the whole
  safety envelope, so that `span` in the point above cannot go non-positive.
- `study_operating_point_regret.py::check_constraints`: the quality-constraint threshold --
  here a plain, independent function argument, unrelated to the config.

`probe_margin_energy_cost.py` (KA1, section (b) above) built each boundary by replacing
`material.critical_moisture_kg_per_kg` in the config -- moving the kinetics parameter and
the quality threshold together, exactly like Befund 54 DA1's `probe_critical_moisture_
sensitivity.py` (same `build_variant_config` pattern; see the addendum there). At the
RL-agent-margin boundary, this made `validate_isotherm_against_critical_moisture` reject
the *config* before any point was solved -- a kinetics-validity failure, not a finding about
what boundary is economically or physically reachable.

*The fix.* `scripts/probe_margin_energy_cost_decoupled.py` (committed `dc5c1c4`) keeps
`material.critical_moisture_kg_per_kg` fixed at 0.20 kg/kg (same fingerprint as Befund
52/54) for every boundary, and instead composes `evaluate_point`/`check_constraints`/
`_optimize` (unchanged, via `importlib`, same as KA1) into a new objective that takes the
quality threshold as an explicit parameter -- `study_operating_point_regret.py` itself is
untouched. Smoke-tested first: at the RL-agent-margin threshold, all 9 points now solve,
moisture ≈0.0604 kg/kg at every point -- consistent with GA1's observed 0.055-0.070 kg/kg,
which the contaminated version could not explain. Full run: 51.9 min,
`artifacts/runs/probe_margin_energy_cost_decoupled.json`, `git/dirty=False`, six of the
seven boundaries recomputed (the 0.20 kg/kg anchor is unaffected -- at margin 0, kinetics
and threshold coincide, so it is unchanged from Befund 52/54 either way).

*Corrected KA2 table (decoupled; KA1's contaminated numbers in parentheses for reference):*

| Boundary | margin [kg/kg] | feasible | pooled cost [%] | pooled remaining, all 9 |
|---|---|---|---|---|
| oracle anchor | 0 | 9/9 | 0.000 (0.000) | 1.000 (1.000) |
| excited-trained LSTM | 0.001628 | 9/9 | 0.263 (0.054) | 0.980 (0.996) |
| constant-trained LSTM | 0.004465 | 9/9 | 0.605 (0.174) | 0.954 (0.987) |
| boundary 0.18 kg/kg | 0.02 | 9/9 | 2.815 (0.824) | 0.784 (0.937) |
| boundary 0.15 kg/kg | 0.05 | 9/9 | 8.362 (6.025) | 0.358 (0.537) |
| boundary 0.12 kg/kg | 0.08 | 9/9 | 15.59 (18.83) | -0.198 (-0.447) |
| RL-agent margin | 0.139545 | **9/9 (was 0/9)** | 42.13 (n/a) | **-2.237 (n/a)** |

The correction does not point the same direction throughout: at the four smaller margins
(up to and including the 0.05 kg/kg margin, boundary 0.15 kg/kg), the decoupled (correct)
cost is *higher* than KA1 reported and the remaining benefit *lower* -- the entangled
kinetics in KA1 had made drying artificially easier at a reduced `X_cr`, partly offsetting
the true constraint cost. At the 0.08 kg/kg margin (boundary 0.12 kg/kg) this reverses: the
corrected cost is *lower* (15.59% vs. 18.83%) and the remaining fraction *less negative*
(-0.198 vs. -0.447) than KA1 reported -- the entangled kinetics there had made the tightened
search harder, not easier, than the true constraint alone. At the RL agent's own margin, KA1
could not report a number at all; the corrected boundary is not only computable but shows
the *most* negative remaining fraction of any boundary tested, as expected. No simple rule
("KA1 always over-/understated") holds across the whole range -- the sign and size of the
contamination depend on where the entangled kinetics shift lands relative to the true
constraint at each boundary. The one qualitative finding that survives unchanged: the
remaining fraction turns negative between the 0.15 and 0.12 kg/kg boundaries in both the
contaminated and the corrected version -- the location of that crossover was not an artifact
of the bug, even though its exact value was.

*Corrected core question.* At the excited-trained soft sensor's own margin, 98.0% of Befund
52's control benefit survives (was 99.6%); at the constant-trained (operating-data-only)
soft sensor, 95.4% survives (was 98.7%) -- both still close to fully preserved, but
measurably less so than KA1 reported. The benefit degrades further at the 0.18 kg/kg
boundary (78.4% remaining, was 93.7%) and turns negative between the 0.15 kg/kg (35.8%
remaining, was 53.7%) and 0.12 kg/kg (-19.8%, was -44.7%) boundaries.

*Corrected non-binding-point check.* Section (c)'s finding needs revision, not just
rescaling: with kinetics fixed, a small margin costs **exactly** nothing at points where
moisture does not bind at 0.20 kg/kg, at every margin up to and including the 0.18 kg/kg
boundary (median AND pooled cost 0.0% at the excited- and constant-trained margins; pooled
cost 0.049% -- floating-point-scale -- at 0.18 kg/kg). This is a cleaner result than KA1's
smeared small-but-nonzero costs there, which were themselves a symptom of the entangled
kinetics shifting the whole optimisation surface, not a real property of the true
constraint. The 0.15 kg/kg boundary is where this genuinely starts to break down: 2 of the 3
non-binding points still cost exactly 0%, but the third now binds (pooled cost 2.03%,
pooled remaining fraction -1.009 -- the small-denominator sensitivity flagged as a caveat
throughout this entry, not a new problem).

*Corrected KA3.* The counter-check against the RL agent is now direct rather than
extrapolated: at the agent's own margin, this study's pooled remaining fraction is **-2.237**
against Befund 62 FA3's measured **-2.797** (10,000 steps, median over 3 seeds, 25-context
5x5 grid) -- about four-fifths of FA3's magnitude (ratio 0.80), same sign, same order of
magnitude, on a 9-context 3x3 grid. This is a substantially cleaner confirmation of the
pre-registered KA3 reading than the contaminated run allowed: the margin mechanism alone,
evaluated correctly, accounts for most of the RL agent's measured shortfall against the
fixed operating point. The caveat noted in advance still applies in full -- the agent does
not hold a uniform margin at every point (GA1: seed range 0.1299-0.1454 kg/kg), so a ~20%
gap between -2.237 and -2.797 is well within what that alone could explain, without invoking
any additional mechanism.

*Open model question, recorded, not to be fixed here.* The model represents the drying
kinetics' critical moisture and the product's quality limit as ONE parameter
(`material.critical_moisture_kg_per_kg`). Physically these are two different quantities --
a material/process property of the drying curve, and a specification choice about
acceptable product quality -- that happen to be numerically equal in this placeholder
configuration only by construction, not by necessity. Separating them would be a model
change with fingerprint consequences for every prior finding that touched this field
(Befund 52, 53, 54, 57, 62 among them); **not implemented**, recorded here as an open
question for whoever next revisits the material model.

### 65. Soft-sensor accuracy requirement at the moisture limit -- met, but with a 3.35x smaller safety reserve right at the optimum, because the boundary region is under-represented in training

**a) Purpose and scope.** Befund 64 (KA2, decoupled) gives the fraction of Befund 52's
control benefit that survives as a function of the safety margin held at the moisture
limit -- measured against ONE global soft-sensor error (Befund 62 HA1's 5-seed median
RMSE). Befund 62 itself shows the optimum sits exactly on `critical_moisture_kg_per_kg`
(0.20 kg/kg) at 17 of 25 disturbance grid points. A single global error says nothing about
whether the soft sensor is that accurate specifically where the optimum lives. This entry
answers two questions: (GA1) how accurate would the soft sensor need to be, read off
Befund 64's curve, to keep 90/50/0 % of the control benefit; and (GA2/GA3) is it that
accurate right at the boundary, or only on average.

**GA1 -- accuracy requirement, read from Befund 64.** Befund 64's decoupled KA2 result
(`artifacts/runs/probe_margin_energy_cost_decoupled.json`) gives seven (margin, pooled
remaining fraction) points. The 90 %-mark falls in the largest unsampled gap in that curve
-- between the constant-trained soft sensor's margin (0.0044651 kg/kg, 95.35 % remaining)
and the bare 0.18 kg/kg grid point (0.02 kg/kg margin, 78.37 % remaining) -- exactly the
range the task anticipated. `scripts/probe_soft_sensor_margin_requirement.py` computes two
more boundaries (`critical_moisture` 0.19/0.185 kg/kg, margins 0.01/0.015 kg/kg) with the
same decoupled mechanism as Befund 64 MA2 (kinetics fixed at 0.20 kg/kg, only the
quality-constraint threshold moved), then linearly interpolates for the 90/50/0 % marks
over all nine points. Runtime: 17.0 min (estimated 15-20 min beforehand from MA2's
per-boundary wall times), `git/dirty=False` throughout, script committed (`2616aa0`)
before the run.

| Margin [kg/kg] | Remaining fraction | Source |
|---|---|---|
| 0 | 1.0000 | oracle anchor (Befund 64) |
| 0.0016282 | 0.9798 | excited-trained LSTM margin (Befund 64) |
| 0.0044651 | 0.9535 | constant-trained LSTM margin (Befund 64) |
| 0.01 | 0.8952 | grid 0.19 (new, this entry) |
| 0.015 | 0.8407 | grid 0.185 (new, this entry) |
| 0.02 | 0.7837 | grid 0.18 (Befund 64) |
| 0.05 | 0.3575 | grid 0.15 (Befund 64) |
| 0.08 | -0.1980 | grid 0.12 (Befund 64) |
| 0.1395451 | -2.2371 | RL-agent margin (Befund 64) |

Linear interpolation, `m = 3*sigma` (Befund 62 HA1's convention):

| Remaining fraction | Margin [kg/kg] | `sigma_max` [kg/kg] | Reserve vs. excited-trained RMSE (0.000543) | Reserve vs. operating-data RMSE (0.001488) |
|---|---|---|---|---|
| 90 % | 0.009544 | 0.003181 | 5.86x | 2.14x |
| 50 % | 0.039970 | 0.013323 | 24.5x | 8.95x |
| 0 % | 0.069308 | 0.023103 | 42.6x | 15.5x |

Read against their own GLOBAL RMSE, both soft sensors comfortably clear the 90
%-preservation requirement -- consistent with Befund 64's own coarser reading (98.0 %/
95.4 % remaining at the two soft-sensor margins). The reserve factor is the number that
changes once GA2 supplies a LOCAL rather than global error (see GA3).

## Was Schritt 6, Teil 2 als Nächstes braucht

Torch läuft jetzt (Befund 24); der Rahmen ist bereit, gebraucht wird die eigentliche
Studie statt eines einzelnen Trainingslaufs.

1. ~~**Lernkurve für GRU und LSTM fahren**~~ **Erledigt** (Befund 28): `study_architectures.py`,
   fünf Seeds, Stützstellen 8/16/32/64, `artifacts/runs/arch_composite.jsonl`
   (Provenienzvorbehalt siehe Befund 33). Befund 24
   zeigte, dass ein einzelner Seed bei n = 8 nichts über die Architektur aussagt, nur über
   diesen einen Trainingslauf — Befund 28 schließt diese Lücke mit fünf Seeds und
   Auflösungsschwelle.
2. ~~**Lernkurven bis 128 und 256** auch für die nichtlinearen klassischen Modelle
   (Befund 16).~~ **Erledigt** (Befund 44): `study_learning_curve.py` erweitert (NARX
   ergänzt, aktuelle GAB-Config, `horizon=36`), 10 Seeds, Stützstellen 8-256,
   `artifacts/runs/arx_narx_learning_curve_gab_900s.jsonl`. Befund 16s scharfe
   Sättigungsgrenze bei `n=16` war ein Artefakt der alten Konfiguration -- unter aktuellem
   Standard eine sanfte, bis `n=128` auslaufende Kurve. NARX schließt die Lücke zu ARX
   drastisch (69 %→12 % bzw. 79 %→9 % je nach Zielgröße), überholt ihn aber nie: reine
   Nichtlinearität hilft hier nicht. **Kein** Vergleich mit den neuronalen Ergebnissen aus
   Punkt 1 (Befund 28) -- der lief unter dem alten Standard und ist nicht direkt
   vergleichbar (Befund 44c, siehe bereits Befund 43e). Ein neuronaler Vergleich bei
   `n=128/256` unter aktueller Konfiguration bliebe ein eigener, separat zu entscheidender
   Arbeitsstrang.
3. ~~**Δt-Vorstudie wiederholen** mit korrigiertem Feuchteniveau. Die Sequenzlänge aller
   Architekturen hängt daran.~~ **Erledigt** (Befund 35): gegen `greybox_gab_placeholder.yaml`,
   900-s-Horizont, Kandidaten {5,15,25,60} s, 10 Seeds. Δt=25 s bestätigt/empfohlen
   (`product_moisture`-Median robust bester Wert, `energy_target` kein Ausreißer); Δt=60 s
   nur bei `product_moisture` isoliert günstig, bei `energy_target` schlechtester Kandidat --
   keine generelle Alternative.
4. **Seed-Budget entscheiden.** Drei Seeds lösen 10–15 % auf; für 5 % sind 12–26 nötig.
5. ~~**Split-Vergleich messen** (fenster- gegen laufweise) — als Ergebnis für M2 und als
   Erklärung gegenüber dem Verbund.~~ **Erledigt** (Befund 39): Extra-Trees-Reproduktion
   von Kräuters Merkmalsraum, fenster-/punktweiser gegen laufweisen Split auf demselben
   96-Läufe-Pool, 5 Seeds. Punktweise sieht bei allen drei Zielgrößen optimistischer aus
   (Faktor 0.71-0.32 relativ zum laufweisen NRMSE, am stärksten bei
   `product_moisture_kg_per_kg`) — validiert nachträglich die im ganzen Projekt
   durchgehend verwendete laufweise Splitkonvention.
6. **Relevanzgrenze festlegen**, getrennt von der Auflösungsschwelle (Befund 19).
   **Ergänzung, verifiziert an `arch_gru.jsonl`:** die Auflösungsschwelle aus
   `climatology` im Soft-Sensor-Betrieb ist **nicht** wie die aus `persistence`
   konstant über `n_train`, sondern schrumpft mit ihm — auf `energy_target`,
   Primärmetrik `nrmse` (gepoolt): 5.36 % (`n=8`) → 2.42 % (`n=16`) → 0.88 % (`n=32`)
   → 0.33 % (`n=64`). Grund, geprüft in `models/baselines.py`,
   `ClimatologyModel.fit`: der Mittelwert wird aus **allen Fenstern des
   Trainings-Splits** geschätzt (`train.full_batch()`), nicht aus einer festen
   Zeitreihe wie bei `PersistenceModel` (das gar keinen Parameter schätzt, sondern nur
   die letzte Beobachtung jedes einzelnen Testfensters hält — konstruktionsbedingt
   unabhängig von `n_train`). Mehr Trainingsläufe → mehr Fenster → präzisere
   Mittelwertschätzung → weniger Streuung der resultierenden Climatology-NRMSE über die
   Daten-Seeds (gemessen: σ von 0.045 bei `n=8` auf 0.0027 bei `n=64`, Median konstant
   um 1.0–1.04). Das ist kein Bug, sondern die erwartete Konsequenz einer geschätzten
   statt einer parameterfreien Referenz — heißt aber: eine "5 % über Climatology"-Aussage
   bedeutet bei `n=8` etwas anderes als bei `n=64`, und die Relevanzgrenze darf diesen
   Unterschied nicht mit der (dort viel gröberen) Auflösungsschwelle verwechseln.
7. **Trainingsbudget für Punkt 1 kalibrieren** (Epochen, Lernrate, `batch_size` relativ zur
   Fensterzahl bei kleinem n): Befund 24 lief mit effektiv 40 Gradientenschritten insgesamt
   bei n = 8 — bevor die Lernkurve entsteht, ist zu prüfen, ob das für die kleinen
   Stützstellen ausreicht oder das Bild systematisch zu ihren Ungunsten verzerrt.
