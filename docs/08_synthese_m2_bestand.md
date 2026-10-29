# Aussagenbestand für die M2-Synthese (SY1)

> Stand: 2026-09-24, **2. Fassung nach Nutzerrückmeldung** (1. Fassung als
> `docs/07_bestand_synthese_m2.md`, umbenannt — `07_` war belegt). Arbeitsdokument für die
> Vorbereitung von `docs/08_synthese_m2.md` (Meilenstein M2, TUHH-B-AP1). Jede Zeile ist auf
> die **endgültige, korrigierte** Fassung eines Befunds in `docs/04_stand_ap1.md`
> zurückgeführt. Keine Zahl in diesem Dokument ist überschlagen — Mediane/Verhältnisse sind
> einfache, projektüblich verwendete Aggregationen direkt aus committeten Metrikwerten; wo
> eine Zahl neu verknüpft werden musste (F6), ist sie zuerst als eigener Nachtrag in
> `docs/04_stand_ap1.md` committet, bevor sie hier zitiert wird.
>
> **Noch nicht geprüft** (2. Fassung), Rückmeldungen zur 1. Fassung siehe Abschnitt am
> Dateiende. **Dies ist die Bestandsaufnahme nach SY1 — noch keine Prosa, keine Entscheidung,
> was in `08_synthese_m2.md` übernommen wird.**

## SY0 — was M2 laut Projektdokumenten verlangt (wörtlich)

Aus `docs/01_projektsteckbrief.md`:

- **M2 (Monat 21):** "Methodik der Simulations- und ML-Modellierung dokumentiert
  (Vorgehensweisen, Werkzeuge, Vorlagen; Reduzierung des Trainingsdatenbedarfs)."
- "M2 ist mein Zielmilestone. Das primäre Ergebnisformat dafür sind **Lernkurven**
  (Fehler über Anzahl Trainingsläufe), nicht Einzelwerte."
- Zwei Ergebnisse, die **eigenständig** in M2 gehören, nicht als Nebensache:
  1. **Auflösungsvermögen des Architekturvergleichs:** "Mit drei Seeds sind Unterschiede
     unter 10 bis 15 % nicht nachweisbar; für eine Aussage auf 5-Prozent-Ebene braucht es
     12 bis 26 Seeds."
  2. **Trainingsdatenbedarf ist eine Eigenschaft der Modellklasse, nicht des Prozesses:**
     "Der lineare ARX sättigt bei 16 Läufen, ein nichtlineares Modell mit identischer
     Verzögerungsstruktur läuft bei 64 noch weiter."

Aus `docs/05_arbeitsplan_ap1.md`:

- Leitidee: "Der Wert von AP 1 liegt nicht darin, xLSTM zum Laufen zu bringen, sondern in
  einem **belastbaren Vergleichsrahmen**."
- Zielgröße: `product_moisture_kg_per_kg` im Mehrschritt-Rollout, `energy_target` als
  Kontrollgröße, `d50_um` gestrichen (zu 84 % Rauschgrenze).
- Vorgaben für alle Architekturen: Residual-Parametrisierung an einem Anker (Soft-Sensor:
  direkte Parametrisierung), kein Teacher Forcing als Standard, Sequenzlänge als
  Modellentscheidung, Physik-Constraints als Output-Projektion/Monotonie.
- **Kein RL** in AP1 ("erst müssen stabile Mehrschritt-Rollouts stehen").

**Methodische Pflichten aus dem Projektauftrag (SY0-Vorgabe), die sich im Dokument
wiederfinden müssen:** Lernkurven als primäres Format; 3–5 Seeds mit Median und
Spannweite; Splits auf Laufebene; Baselines (Persistenz, statische Regression, linearer
ARX) vor jedem Architekturvergleich; Metriken Ein-Schritt-Fehler, Mehrschritt-Rollout-
Fehler, Inferenzzeit.

**Folgerung für die Synthese-Gliederung:** M2 verlangt eine **Methodik**-Dokumentation mit
Lernkurven als Kernformat, plus die zwei explizit genannten Eigenständigkeiten
(Auflösungsvermögen, Modellklassen-Abhängigkeit des Datenbedarfs). Das Laborbuch enthält
deutlich mehr Material (Wärmeverbund, RL-Voruntersuchung, Anlagenvalidierung) — laut
Steckbrief ist AP1 primär die ML-Methodik; die Wärmeverbund-/RL-Befunde gehören laut
Arbeitsplan in den Ausblick, nicht in den Hauptteil ("AP1 ist nicht RL").

---

## Tabelle 1 — Aussagenbestand für M2

Spalten: Aussage · Zahl(en) · Befund · Artefakt · Commit · Korrigiert/ergänzt?

### A. Hauptergebnis: Lernkurven aller acht Architekturen

| # | Aussage | Zahl(en) | Befund | Artefakt | Commit | Korrigiert? |
|---|---|---|---|---|---|---|
| A1 | GRU/LSTM bleiben über den gesamten getesteten Budgetbereich (n_train ≤ 64) auf allen drei Zielkanälen und beiden Testmengen die besten trainierten Modelle; Transformer folgt mit Aufholtendenz, xLSTM/Mamba bleiben durchgehend zurück. | `product_moisture_kg_per_kg`, angeregt, h=36, n=64: LSTM 0,00051, GRU 0,00061, Transformer 0,00101, xLSTM 0,00123, Mamba 0,00505 kg/kg (Faktor 9,9 bester/schlechtester trainierter Kanal) | 58 (OA5), Fazit | `arch_learning_curve_projected.jsonl` (51) + `arch_learning_curve_projected_extended.jsonl` (58) | 0b3eaaf (51), 256d644 (58) | **JA** — s. Korrekturtabelle K1 (Transformer-Aussage), K3 (energy_target-Einordnung) |
| A2 | Baselines: Persistenz ist testmengenabhängig extrem unterschiedlich stark (angeregt schwach, Konstantlauf trivial stark); statische Regression ist die testmengenrobustere Baseline. | `bed_temperature_celsius`, h=36, Persistenz: 6,408 (angeregt) vs. 0,630 (Konstantlauf), Faktor 10,2 | 58 (OA5a) | `arch_learning_curve_projected_extended.jsonl` | 256d644 | Nein, aber **Voraussetzung** für K1/K2 |
| A3 | ARX/NARX-Lernkurve unter aktueller Konfiguration (GAB-Isotherme, Δt=25s/900s): keine scharfe Sättigung bei n=16, sondern sanft auslaufend bis n≈128; NARX schließt die Lücke zu ARX drastisch, überholt ihn aber nie. | `product_moisture`, ARX: 0,546 (n=8) → 0,423 (n=128) → 0,428 (n=256, Rauschen); NARX/ARX-Lücke 69,1 % (n=8) → 12,4 % (n=256) | 44 | `arx_narx_learning_curve_gab_900s.jsonl` | im Befund nicht genannt (Fingerprint `82f65f576a2e5253` verifiziert) | **JA** — korrigiert Befund 16 (alte Config, s. Korrekturtabelle K4). Nicht direkt vergleichbar mit GRU/LSTM-Kurve (andere Konfiguration, Befund 43e/44c) |
| A4 | GRU/LSTM auf `bed_temperature_celsius` (Ablufttemperatur/Anwendungsgröße, eigenes Protokoll): klar vor statischer Baseline in Kelvin, trotz hoher R² der Baseline (R²_stat=0,768). | GRU 0,131–0,183 K RMSE vs. Statisch 5,88–6,85 K RMSE (Faktor ≈20) | 50 (e, j) | `outlet_temperature_gru_protocol.jsonl` | im Befund nicht genannt | **JA** — s. Korrekturtabelle K2 |

### B. Transformer-Frage (endgültige Fassung)

| # | Aussage | Zahl(en) | Befund | Artefakt | Commit | Korrigiert? |
|---|---|---|---|---|---|---|
| B1 | Auf `bed_temperature_celsius` (Konstantlauf, h=36) erreicht der Transformer **Parität** zu LSTM, sobald das Epochenbudget nicht mehr bindet (600 statt 200 Epochen) — bei n_train=64. | Faktor Transformer/LSTM: 1,026 (200 Epochen) → **1,000** (600 Epochen; Transformer 0,44881 K, LSTM 0,44901 K) | 58, Nachtrag (600 Epochen) | `arch_epoch_budget_probe600.jsonl` | 7e3fb1d (Skript), 0ed48e5 (Daten) | Ersetzt die Erstfassung von Befund 58 (s. K1) |
| B2 | Auf `energy_target` (Konstantlauf, h=36, n=64) wächst der Abstand Transformer↔LSTM unter 600 Epochen leicht statt gleich zu bleiben. | Faktor 1,082 (200 Ep.) → 1,158 (600 Ep.); LSTM −5,9 % relativ, Transformer +0,8 % relativ | 58, Nachtrag | `arch_epoch_budget_probe600.jsonl` | 7e3fb1d/0ed48e5 | Widerlegt die vorab formulierte Erwartung "Abstand bleibt gleich" |
| B3 | Auf der **angeregten** Testmenge bleibt der Transformer bei 600 Epochen auf allen drei Köpfen hinter LSTM zurück, am deutlichsten bei `product_moisture_kg_per_kg`. | Faktor 1,15–1,99 (Median über 5 Seeds) | 58, Nachtrag | `arch_epoch_budget_probe600.jsonl` | 7e3fb1d/0ed48e5 | Nein (kein Widerspruch zu B1/B2, andere Testmenge) |
| B4 | `n_train=32` bleibt für den Transformer **offen**: dort ist er bei 600 Epochen nicht nachgerechnet worden (nur n_train=64), war aber bei 200 Epochen ebenfalls an allen 5 Seeds epochenbegrenzt. | Epochenbindung bei n=32: Transformer 5/5, LSTM 3/5 | 59c | — | — | Kein Ergebnis, offene Frage |
| B5 | Transformer: 14–18x teurer als GRU/LSTM bei Training **und** Inferenz, strukturell durch fehlendes KV-Caching im Decoder-Loop (O(H²)), nicht nur durch den Backward-Pass. | Training Faktor ~14,2x (b) / 14,0–16,4x (d, unabhängig gemessen); Inferenz ~17,7x (2394,8 µs/Fenster vs. GRU ~135 µs) | 37 (b, c) | `transformer_reduced_comparison.jsonl`, `probe_new_arch_cost.py`-Ergebnis | im Befund nicht genannt | Nein |

### C. Inferenzzeit je Architektur (Pflichtmetrik) — **vollständig, aus den JSONL-Artefakten gelesen, nicht nachgemessen**

Korrektur gegenüber der Erstfassung dieser Tabelle: `inference_us_per_window` steht als
Feld in `metrics` für **alle acht Architekturen** in den committeten Artefakten — die
Erstfassung hatte nur die im Fließtext der Befunde 37/38 zitierten Einzelzahlen gesucht,
nicht die Records selbst. LSTM und xLSTM sind vollständig vorhanden, keine Lücke.

**Vergleichbarkeitsprüfung zuerst (wie gefordert):** `torch/num_threads`,
`env/mkl_num_threads`, `env/omp_num_threads` sind über `arch_learning_curve_projected.jsonl`
(Befund 51: persistence/static/arx/gru/lstm) und `arch_learning_curve_projected_extended.jsonl`
(Befund 58: transformer/xlstm/mamba) durchgehend identisch (`1`/`1`/`1`) — beide Dateien sind
laut Befund 58 (OA5) zusätzlich protokollgleich (gleicher Fingerprint, gleiche Splits/Seeds je
`master_seed`). Diese acht Architekturen sind damit **untereinander direkt vergleichbar**.

| # | Architektur | Median µs/Fenster (n_train=64, 10 Records: 5 Seeds × 2 Testmengen) | Spanne [µs] |
|---|---|---|---|
| C1 | Persistenz | 0,15 | [0,09; 0,70] |
| C2 | Statisch | 1,76 | [1,29; 5,08] |
| C3 | ARX | 5,25 | [2,94; 31,45] |
| C4 | LSTM | **182,8** | [140,3; 260,2] |
| C5 | Mamba | 494,1 | [316,3; 639,4] |
| C6 | GRU | **323,1** | [186,9; 349,3] |
| C7 | xLSTM | 1121,9 | [889,8; 1779,2] |
| C8 | Transformer | 3081,6 | [2385,5; 3369,6] |

Quelle: `arch_learning_curve_projected.jsonl` (Commit `0b3eaaf`) +
`arch_learning_curve_projected_extended.jsonl` (Commit `256d644`), Feld
`metrics.inference_us_per_window`, gefiltert auf `n_train=64` (Tag `n_train`).

**Unerwarteter Befund, hier zum ersten Mal explizit benannt (nicht im Fließtext des
Laborbuchs diskutiert):** In diesem Datensatz ist **LSTM durchgehend schneller als GRU**
(182,8 vs. 323,1 µs bei n=64; 362,5 vs. 465,8 µs bei n=8) — das Gegenteil der in Befund 37
zitierten Referenzgrößenordnung ("GRU-Referenzwert ~135 µs/Fenster"). Nicht weiter
diagnostiziert (keine Vermutung zur Ursache in dieser Bestandsaufnahme); für die Synthese
zu entscheiden, ob das als offene Beobachtung mitgeführt wird.

**Vorbehalt — zwei nicht poolbare Datenfamilien:** `arch_gab_constant_holdout.jsonl`
(Befund 36, Quelle der im Fließtext von Befund 37/38 zitierten Zahlen: GRU ~135 µs,
referenziert als "~135 µs/Fenster"), `transformer_reduced_comparison.jsonl`,
`xlstm_reduced_comparison.jsonl`, `mamba_reduced_comparison.jsonl` und
`mamba_gab_constant_holdout.jsonl` (Quelle von Transformer 2394,8–2422,3 µs und Mamba
365,2 µs) tragen **kein** Thread-Feld (`torch/num_threads` etc. = `None`/nicht geloggt).
Diese Zahlen sind intern (innerhalb derselben Datei, z. B. Transformer-gegen-GRU-Faktor in
Befund 37) weiterhin gültig, aber **nicht** mit der obigen Tabelle poolbar oder direkt
gegenzurechnen — genau der Grund, warum z. B. der dort zitierte GRU-Wert (~135 µs) von dem
hier gemessenen (323,1 µs bei n=64) abweicht. Für die Synthese: **nur die Tabelle oben
(Befund 51+58, thread-geprüft) als Pflichtmetrik-Quelle verwenden**, die Einzelwerte aus
Befund 37/38 bleiben Belege für die dortigen (jeweils intern konsistenten) Kostenfaktoren,
nicht für einen Achtervergleich.

### D. Methodische Befunde

| # | Aussage | Zahl(en) | Befund | Artefakt | Commit | Korrigiert? |
|---|---|---|---|---|---|---|
| D1 | Statische Regression ist die testmengenrobuste Baseline gegen Persistenz — der Faktor gegen sie ist keine Testmengen-Falle wie bei Persistenz/K5, sondern eine robuste Modellgüte-Aussage. | `energy_target`, n=64, h=36: Faktor gegen Statisch 7,68 (GRU)/7,82 (LSTM)/6,43 (Transformer) angeregt; 5,74/5,62/5,20 Konstantlauf | 59a (SA1) | `analyze_energy_target_static_factor.py`-Ergebnis | bc6b097 | Ersetzt Teile von Befund 51 (K5-Warnung) — s. K5 |
| D2 | K5-Falle als Lehrbeispiel: ein Faktor-Sprung zwischen zwei Log-Dateien war ein Testmengen-Wechsel (Persistenz-Baseline strukturell unterschiedlich stark), keine Modellverbesserung. | `product_moisture`, n=64, h=36: Faktor 1,31 (GRU)/1,70 (LSTM) auf Konstantlauf (konsistent zur alten Referenz 1,54/2,00); Faktor 4,4–24 auf `excited` (neu, kein alter Referenzwert) | 51, K5 | `arch_learning_curve_projected.jsonl` | 0b3eaaf | Selbstkorrektur innerhalb des Befunds (CA1-Nachtrag) |
| D3 | R² komprimiert den Entscheidungsbereich: Bei `bed_temperature_celsius` bedeutet ein R²-Sprung von 0,768 auf 0,965 (FIR k=43) einen physikalischen Restfehler von Faktor 20 gegenüber GRU (0,18 K) — R² allein zeigt das nicht. | FIR k=43: 3,62 K Residuum (30,2 kW-Äquivalent) vs. GRU 0,18 K (1,5 kW) → Faktor 20 | 50j | `probe_l1_clip_sensitivity.py`-Ergebnis (Befund 47 AB2/AC1) + `outlet_temperature_gru_protocol.jsonl` | im Befund nicht genannt | Korrektur der ursprünglichen "reine Anwendungsgröße"-Einordnung aus 50a |
| D4 | Dasselbe Muster bei `energy_target`: R²_stat=0,972 (bzw. historisch 0,987) sieht nahe 1 aus, verdeckt aber Faktor 5,2–7,8 im RMSE gegenüber den trainierten Modellen. | R² der trainierten Modelle ≥0,999 (angeregt)/≥0,9989 (Konstantlauf) gegen R²_stat=0,972 | 59b (SA2) | `02_prozess_setup.md` (R²-Tabelle) | im Befund nicht genannt | s. Korrekturtabelle K3 |
| D5 | Rolle der zwei Testmengen: angeregt misst Identifikationsgüte, Konstantlauf-Holdout die Betriebsrelevanz — Verwechslung hat im Projekt bereits einmal eine falsche Zahl erzeugt (K5). | — (qualitativ) | 58 (OA5a), 51 (K5) | — | — | Nein, aber zentrale Lehre |
| D6 | Fensterweiser/punktweiser Split gegenüber laufweisem Split ist optimistisch verzerrt — rechtfertigt nachträglich die im Projekt durchgehend verwendete laufweise Splitkonvention. | Verhältnis punktweise/laufweise, h=1: `energy_target` 0,71; `product_moisture` 0,32; `d50_um` 0,48 | 39 | `run_extra_trees_split_comparison.py`-Ergebnis | im Befund nicht genannt | Nein |

### E. Datenbedarf und Anlage

| # | Aussage | Zahl(en) | Befund | Artefakt | Commit | Korrigiert? |
|---|---|---|---|---|---|---|
| E1 | Betriebsdaten (rein konstante Läufe) genügen für `product_moisture_kg_per_kg` im getesteten Budgetbereich (n≤64) nicht — die Lücke zum angeregten Training bleibt bei Faktor 2,2–3,1 flach, kein Aufholen. | NRMSE-Verhältnis konstant/angeregt: 2,2–3,1 (GRU), 2,4–3,1 (LSTM), über n=8..64 | 55 | `arch_learning_curve_constant_trained.jsonl` | 1b85ce4 | Nein |
| E2 | Konstant trainierte Modelle erschöpfen ihr lernbares Signal fast sofort (Early Stopping bei ~24 statt ~170 Epochen) — das ist eine Messung des Informationsgehalts, keine Recheneffizienz. | GRU: 24,3 Epochen (konstant) vs. 166,3 (angeregt) bei n=8 | 55 | `arch_learning_curve_constant_trained.jsonl` | 1b85ce4 | Nein |
| E3 | Reale Anlagendaten (Anwendungspartner) bestätigen das synthetische Ergebnis: Stellgrößen variieren innerhalb eines Betriebspunkts kaum. | Variationskoeffizient 0,12–2,01 % (Luftmenge, Zulufttemperatur, Sprührate, P1/P2) | 56 (56a) | (Rohdaten nicht im Repo, Auswertung `study_plant_validation.py`) | e3efc19/88c0746 | Nein |
| E4 | Anlagenvalidierung der Betttemperatur: ein Parametersatz trifft beide Betriebspunkte auf 0,4–2,1 K — validiert die Enthalpiebilanz, **nicht** die Trocknungskinetik oder Produktfeuchte (unvalidiert, da an der Anlage nicht gemessen; Fluidisierungsprüfung schlägt an beiden Punkten fehl). | P1: −0,4 K, P2: −2,1 K | 56 (NA5) | `study_plant_validation.py`-Ergebnis | 88c0746 | Nein, aber Vorbehalt zwingend mitzuführen |
| E5 | Warum die Modelle gebraucht werden: der Anlagenzustand ist nicht messbar (Softsensor-Notwendigkeit), Produktfeuchte bindet am energetischen Optimum, ist an der Anlage nicht gemessen. | Feuchtegrenze bindet an 17/25 Optimalpunkten (Befund 52c); keine Produktfeuchtemessung in Anlagenexporten (Befund 56) | 52c, 56 | `bb1_bb3_operating_point_regret.json`, Plant-Export-Auswertung | 7103b73 (Win-Repro) | Nein |

### F. Ausblick (knapp — AP1 ist nicht RL)

| # | Aussage | Zahl(en) | Befund | Artefakt | Commit | Korrigiert? |
|---|---|---|---|---|---|---|
| F1 | Ein fester Betriebspunkt hat Regret im Prozentbereich (nicht Promille) gegenüber der instanzoptimalen Lösung — je nach Störgrößenregime (Feuchtegrenze bindend vs. nicht). | Median 16,03 % (P95 22,20 %, Max 25,80 %), n=25 Gitterpunkte | 52 | `study_operating_point_regret_win_repro.json` | 7103b73 | Nein (bitgleich Linux↔Windows reproduziert) |
| F2 | Regret übersteht den Übergang Beharrungszustand→Charge praktisch unverändert. | Median 16,25 % (Charge) vs. 16,03 % (Fixpunkt) | 53 (CB3) | `study_operating_point_transient.json` | dirty (Provenienzvorbehalt, s. Befund 53g) | Nein |
| F3 | Ein RL-Agent (SAC, Ein-Schritt-Bandit) hält an den 17/25 Kontexten, wo das Orakel exakt auf der Gutsfeuchtegrenze sitzt, eine deutliche Sicherheitsmarge — und bezahlt dafür mit höherer Gasgeschwindigkeit. | Median Feuchtemarge Agent 0,1395 kg/kg (kritisch 0,20 kg/kg) vs. Orakel ~0 kg/kg; Median Δu_gas +0,103 m/s | 62 (GA1) | `rl_bandit_ga1_safety_margin.json` | 111cfb1 (GA1-Befund), 11e0cf0 (GA1-Skript) | Ersetzt frühere Zwischenstände desselben Befunds (Sentinel-Fix, s. u.) |
| F4 | Pinch-Lücke der sechs Partner-Verschaltungsfälle zum thermodynamischen Optimum ist klein (2,3–4,2 %), nicht die "mehreren Faktoren", die die 2.-Hauptsatz-Verletzungen (Befund 57) nahelegten. | Fall 1 (beste Verschaltung): 2,3–4,2 % über dem Pinch-Ziel, je nach `T_kalt`/`dT_min` | 60 (PA3) | `run_pinch_benchmark.json` | (Branch `lp-benchmark`, gemergt — Einzelcommit im Befund nicht genannt) | Nein |
| F5 | Der Standort-Pinch springt zwischen zwei Engpassstellen (Wirbelschicht-Ablufttemperatur vs. Sprühturm-Taupunkt), abhängig von der Fahrweise der Wirbelschichtlinie — Bindeglied zwischen Prozess- und Standortebene. | Pinch wandert 44,94–73,50 °C über 3 zulässige Gitterpunkte; Standort-Brennstoff ändert sich dabei nur 3,8 % relativ | 61 (QA3) | `run_process_pinch_sensitivity.json` | (Branch `lp-benchmark`, gemergt) | Nein |

| F6 | Die Sicherheitsmarge des Agenten zur Feuchtegrenze ist im Modell weit größer als der durch den Vorhersagefehler des empfohlenen Feuchte-Softsensors gerechtfertigte Wert — und bleibt auch gegen den schwächeren, aus reinen Betriebsdaten trainierten Softsensor deutlich ausreichend. | Marge/(3×RMSE): **85,7** (Softsensor aus Anregungsdaten, empfohlener Fall) bzw. **31,3** (Softsensor aus Betriebsdaten) | 62 (HA1-Nachtrag) | `rl_bandit_ha1_margin_vs_prediction_error.json` | 7e9ff67 (Skript), f99c7f4 (Nachtrag+Artefakt) | Neu — war zuvor keine Aussage im Laborbuch, s. u. |

**Herkunft von F6 (Korrektur gegenüber der Erstfassung dieser Bestandsaufnahme):** Diese
Gegenüberstellung existierte ursprünglich an keiner Stelle im Laborbuch — nach der Regel
dieses Auftrags durfte sie deshalb nicht in die Synthese. Sie ist jetzt als Nachtrag **z)
HA1** zu Befund 62 in `docs/04_stand_ap1.md` nachgetragen (reine Auswertung dreier bereits
committeter Artefakte, kein neuer Lauf) und kann ab hier zitiert werden. Zwei Fälle, nicht
einer: der ursprünglich herangezogene Vorhersagefehler (0,001488 kg/kg) ist der des
**konstant** trainierten LSTM (Befund 55, reine Betriebsdaten) — der für den
Feuchte-Softsensor tatsächlich empfohlene Fall ist der **angeregt** trainierte LSTM
(Befund 51/58), dessen Fehler mit 0,000543 kg/kg fast dreimal kleiner ist. Konvention:
Sicherheitsfaktor = Marge / (3×RMSE), in der HA1-Auswertung neu eingeführt, keine
Anlagen- oder Normvorgabe. Vorbehalt: `critical_moisture_kg_per_kg` (0,20 kg/kg) ist ein
unbelegter Platzhalter — die Aussage gilt im Modell, relativ zu diesem Wert.

---

## Tabelle 2 — Korrekturen (dürfen in der Synthese NICHT in der alten Fassung erscheinen)

| # | Thema | Alte/falsche Fassung | Korrigierte Fassung | Korrigiert in | Kommentar |
|---|---|---|---|---|---|
| K1 | Transformer "überholt nicht" | Befund 58 (Erstfassung): "deutliches Aufholen, kein Überholen im getesteten Bereich" — unkommentiert als generelles Ergebnis | Gilt nur **innerhalb des konfigurierten Epochenbudgets** (200); bei n_train=32/64 war der Transformer an praktisch allen Seeds epochenbegrenzt. Mit 600 Epochen: exakte Parität zu LSTM bei `bed_temperature_celsius`, aber wachsender Abstand bei `energy_target` | 59c (SA3), abschließend Nachtrag zu 58 (600-Epochen-Lauf) | **Zwingend mit Epochenbudget-Vorbehalt zitieren**, nicht als reines "holt auf, überholt nicht" |
| K2 | `energy_target`/`bed_temperature_celsius` als "fast statisch"/schwacher Vergleichskanal | Mehrfach im Dokument (Befund 39e, 41c/d): "`energy_target` zu 98,7 % statisch erklärbar" als Begründung für geringe architektonische Aussagekraft | R² nahe 1 verdeckt einen physikalisch bedeutsamen Restfehler: Faktor 5,2–7,8 (energy_target) bzw. Faktor 20 (bed_temperature) im RMSE. Beide Kanäle sind **keine** schwachen Architekturvergleichskanäle | 59b (SA2), 50j | Betroffene Stellen: Befund 39e (Zeile 2223–2225), Befund 41c/d (Zeile 2438/2455). **Nicht** zu korrigieren: Befund 50e (kontrastiert nur, kein Korrekturziel), Befund 58 OA5 (bereits korrekt), `05_arbeitsplan_ap1.md` Zeile 105f. (andere Aussage) |
| K3 | R² = 98,7 % (statische Regression, `energy_target`) | Wird im Dokument mehrfach als aktuelle Zahl zitiert (Befund 2, referenziert in 39e/41c/50e) | 98,7 % ist die **historische, laut `02_prozess_setup.md` nicht reproduzierbare** Zahl; die aktuell reproduzierte liegt bei **97,2 %** | 59b (SA2) | Beide Zahlen sagen nichts über den physikalischen Restfehler (s. K2) — Faktor-Vergleich in D4 nutzt bewusst die *ungünstigere* (höhere, 97,2 %→0,972) reproduzierte Zahl |
| K4a | K5-Faktor (`product_moisture`, Persistenz-Baseline) | Vor der Erstfassung von Befund 51 stand ein **Faktor 4,4–23 ohne Erklärung**, unkommentiert einer Größenordnung über der bisherigen Referenz (Faktor 1,5–2) gegenübergestellt — tatsächlich fälschlich auf der **angeregten** Testmenge gemessen, aber gegen eine auf dem **Konstantlauf-Holdout** gemessene alte Referenz verglichen (zwei verschiedene Testmengen, keine gültige Gegenüberstellung) | Auf dem Konstantlauf-Holdout (derselben Testmenge wie die alte Referenz) liegt der Faktor bei **1,31 (GRU)/1,70 (LSTM)**, n_train=64 — konsistent mit der alten Referenz (1,54/2,00), kein Einbruch. Der Faktor **4,4–24** (Median über 5 Seeds, n_train=8..64) ist ein eigenständiges, gültiges Ergebnis auf der `excited`-Testmenge, für die es keine alte Referenz gibt — beide Zahlen bleiben gültig, solange sie mit Testmenge zitiert werden | 51 (K5, Selbstkorrektur CA1) | Der Auftragstext nannte "19 bis 24" — laut Nutzerrückmeldung ein Ausschnitt aus der CA1-Einordnung, nicht die K5-Aussage selbst; im Laborbuch steht durchgehend 4,4–23 (überholt)/4,4–24 (gültig) |
| K4b | ARX-Sättigung bei n=16 | Befund 16 (alte Config, `greybox_placeholder.yaml`, `horizon=24`): "der ARX ist bei 16 Läufen fertig" | Unter aktueller Konfiguration (GAB-Isotherme, Δt=25s/900s-Horizont) ist die Sättigung eine **sanfte, bis n≈128 auslaufende** Kurve (größte relative Einzelverbesserung bei n=32, nicht n=16), kein scharfer Abbruch | 44 | Alte Zahl war unter anderer Physik gemessen, kein direkt übertragbarer Referenzpunkt (analog Befund 24/43a) |
| K5 | Taupunkt-Werte Q1/Q2 (Wärmeverbund) | Befund 57h nannte 47,9 °C (Q1) und 27,93 °C (Q2) als Taupunkte | Waren **keine** Taupunkte, sondern Taupunkt PLUS 3-K-Abkühlgrenze aus Model_v2 (Zellen-Verwechslung). Korrekte Taupunkte (Buck-Korrelation): **Q1 = 44,94 °C, Q2 = 24,92 °C** | 60 (PA3-Nachtrag), rückwirkend in 57h korrigiert | Die bereits in 57h stehende `Q_rec`-über-`T_kalt`-Tabelle selbst war nicht betroffen — nur die textliche Gegenüberstellung |
| K6 | Transformer "überholt ARX nie" vs. NARX "überholt ARX nie" | Zwei unterschiedliche Aussagen im Dokument, die beim flüchtigen Lesen verwechselbar sind: Befund 44 (NARX überholt ARX nie) und Befund 59c (Transformer überholt LSTM nicht) | Beides bleibt bestehen, sind aber **unterschiedliche Vergleiche** (NARX↔ARX bei Befund 44, Transformer↔LSTM bei Befund 58/59) — in der Synthese eindeutig benennen, welcher Vergleich gemeint ist | — (kein Korrekturbefund, nur Verwechslungsgefahr) | Aufgenommen, weil der Auftragstext "Transformer 'überholt nicht'" fordert — das ist Befund 59c, nicht Befund 44 |
| K7 | Sechs-Fälle-Modell des Partnermodells (`Model_v3.xlsx`): Einheitenfehler | Blattformel `Balances!E20/E27` verrechnete Verdampfungsenthalpie ohne Einheitenumrechnung (2501 statt 2,501·10⁶ J/kg, Faktor 1000) → Angebot/Bedarf-Deckung 48,7 % | Korrigiert: **92,9 %** Deckung; bester Fall wechselt von "Fall 2" (auf Basis der fehlerhaften Zahl) zu **"Fall 1"** (getrennte Verschaltung) | 57 (d, MA5) | Rohdatei (`Model_v3.xlsx`) nicht im Repo — Korrektur nur über Befund 57 und `probe_six_case_reconciliation.py` nachvollziehbar |
| K8 | RL-Agent kollabiert in unzulässige "heiße Ecke" | Erstlauf (WA3/XA3, Sentinel = −1.000.000 kW, Befund-52-Wert): Agent unter Zufallsniveau bei Zulässigkeit UND Energie, Politik kollabiert auf maximale Temperatur+Geschwindigkeit | Mechanismus identifiziert (Sentinel vergiftet die Kritiker-**Regression**, nicht nur den ordinalen Vergleich eines klassischen Optimierers) und behoben: begrenzter, modusabhängiger Sentinel (155,58 kW) → Zulässigkeit perfekt (12/12), kein Kollaps in unzulässige Region mehr; verbleibender negativer gepoolter Regret-Anteil ist die **Sicherheitsmarge zur Feuchtegrenze** (F3), nicht ungerichtetes Versagen | 62 (EA2/FA1–FA4/GA1) | Mehrstufiger Korrekturverlauf innerhalb *eines* Befunds — für die Synthese zählt nur der **Endstand** (FA4/GA1), nicht die Zwischenstände (WA3/XA3/DA1) |

---

## Tabelle 3 — Platzhalter ohne Quelle (schränken jede Aussage ein, die sie berührt)

| Platzhalter | Wert | Fundstelle(n) | Betrifft | Status/Empfindlichkeit |
|---|---|---|---|---|
| `critical_moisture_kg_per_kg` (kritische Gutsfeuchte) | 0,200 kg/kg, Laborniveau | `greybox_gab_placeholder.yaml`-Dateikopf, genannt u. a. in Befund 52a, 54, 56, 62 | Regret (F1), Regime-Umschlag (52c), RL-Sicherheitsmarge (F3) | Regret-Größenordnung **nicht** robust (Faktor 2–3 zwischen 0,10 und 0,20 kg/kg, Befund 54 DA1); aktueller Wert liegt am günstigen Rand des geprüften Bereichs |
| `BED_TEMPERATURE_LIMIT_C` (80-°C-Grenze) | 80,0 °C, frei gesetzt | Befund 46, 52a, 62j (XA4) | Nebenbedingung in Regret-Optimierung und RL; bindet an 0/25 Befund-52-Optima, aber dominant im RL-Explorationsraum (21/24 Verletzungen) | An Anlage F-5 existiert ein realer Grenzwert (SAH-Messstellenkürzel, Befund 62j) — erfragbar, nicht übernommen |
| `d50` (Beharrungsdurchmesser) | ≈750 µm, Laborplatzhalter (aus `nucleus_diameter_m`/`nucleation_solid_fraction`) | Befund 46, 56, 61 (QA2), 62a | Fluidisierungsfenster-Prüfung (u_terminal), Gasgeschwindigkeitsprojektion | Reale Industriegranulate "vermutlich größer/dichter" (Befund 56) — nicht angepasst; schneidet im Anlagenmaßstab 24/27 Gitterpunkte weg (Befund 61 QA2) |
| `T_kalt` (Kaltseite Wärmeübertrager) | 15 °C Default (Modell); real eher ~29,1 °C (Kühlwassertemperatur Kompressorstation) | Befund 57h (NB1) | MA2-Vorzeichenwechsel der Feuchteempfindlichkeit (gilt nur solange Linie kondensiert — bei 29,1 °C kondensiert die Wirbelschichtlinie nicht mehr) | Optimistischer Platzhalter — Partnermodell fährt zudem zwei unterschiedliche `T_kalt` (10/15 °C), Modell nur einen netzweiten Wert |
| `outlet_duct_heat_loss_k` (Leitungsverlust Abluftkanal) | 0,0 K Default (Identität) | Befund 50c, 56 (NA4) | Rekonstruktion `outlet_temperature_celsius` aus `bed_temperature_celsius` | Real eher betriebspunktabhängig als konstant (P1 6,59±0,19 K, P2 8,87±0,59 K, Differenz korreliert mit Sprührate) — kein Wert übernommen |
| `bed_mass_solid_setpoint_kg` (Bettmasse-Sollwert) | 2,000 kg (→ τ₄≈16000 s) | Befund 54 (DA2) | Chargenmittel-`Q_rec`-Abweichung (nicht Regret selbst, der robust ist) | Skalenparameter, nicht isolierbar (koppelt τ₄, Wärmekapazität, Druckverlust gleichzeitig) |
| Relative Feuchte Ansaugluft | 75 % angenommen, keine Quelle | Befund 56 (NA5) | Anlagenvalidierung Betttemperatur | Teil der "ungetunten Annahmen"-Liste, Befund 56 |
| `max_driving_humidity_kg_per_kg` | 0,50 kg/kg Default | Befund 46 (L1) | Verdunstungsrate, Temperaturspitzen | Klippungsanteil variiert stark mit Wert (12,83 %/4,08 %/0,00 % bei 0,50/1,00/5,00 kg/kg) |

---

## Rückmeldungen vom 2026-09-24 — umgesetzt

1. **Inferenzzeit:** aus den committeten JSONL-Artefakten gelesen (Feld
   `inference_us_per_window`), nicht nachgemessen — Tabelle C oben vollständig neu erstellt,
   Thread-Einstellungen vorab geprüft. Keine Lücke mehr; stattdessen zwei nicht poolbare
   Datenfamilien erkannt und getrennt gehalten.
2. **K4:** aufgeteilt in K4a (K5-Faktor, korrekt 4,4–23/überholt bzw. 4,4–24/gültig,
   Testmengen-Verwechslung) und K4b (ARX-Sättigung, Befund 16 vs. 44). "19 bis 24" war laut
   Rückmeldung ein Ausschnitt aus der CA1-Einordnung, keine eigene Laborbuch-Aussage.
3. **Ausblick-Umfang (Planungsnotiz für SY3, noch nicht umgesetzt):** höchstens eine Seite,
   drei kurze Absätze — (a) warum die Modelle nachgelagert gebraucht werden (Zustand nicht
   messbar, Feuchte bindet am Optimum), (b) Befund 52 als Motivation für Regelung, (c) F6 als
   Brücke. Wärmeverbund (57/60/61) in einem Satz. Die Diagnosekette zum Lösersentinel
   (K8/WA3–DA1) gehört **nicht** in den Ausblick. Befund 55/56 sind bereits korrekt unter
   Abschnitt E (Datenbedarf/Anlage) geführt, nicht F (Ausblick) — keine Änderung nötig.
4. **F6:** wie vermutet gemeint, aber die Verknüpfung stand noch nicht im Laborbuch — jetzt
   als Nachtrag **Befund 62(z) HA1** in `docs/04_stand_ap1.md` nachgetragen (Skript
   `scripts/probe_bandit_env_ha1_margin_vs_prediction_error.py`, Ergebnis
   `artifacts/runs/rl_bandit_ha1_margin_vs_prediction_error.json`), dabei die eigene
   Zahlenangabe korrigiert: der ursprünglich herangezogene Fehler (0,001488 kg/kg) ist der
   des konstant trainierten, nicht des empfohlenen angeregt trainierten Softsensors — beide
   Fälle jetzt getrennt in F6 und im Nachtrag ausgewiesen.

## Commits dieser Überarbeitung

- `7e9ff67` — HA1: Skript (vor Ausführung committet, Provenienzregel)
- `f99c7f4` — HA1: Befund-62-Nachtrag in `docs/04_stand_ap1.md` + Ergebnisartefakt
  `artifacts/runs/rl_bandit_ha1_margin_vs_prediction_error.json` (Skriptausgabe bitgleich
  zu den hier zitierten Zahlen: Marge 0,139545 kg/kg, Faktor 85,70/31,25)
- Diese Bestandsdatei (`docs/08_synthese_m2_bestand.md`) selbst wird als Bestandsdatei
  (SY1-Artefakt, kein Fachinhalt) mitcommittet, damit der Arbeitsbaum sauber bleibt — sie
  bleibt trotzdem der **Prüfstand**, nicht die Freigabe: SY2/SY3 beginnen erst nach Ihrer
  Rückmeldung zu dieser 2. Fassung.

**Datei-Umbenennung:** `docs/07_bestand_synthese_m2.md` → `docs/08_synthese_m2_bestand.md`
(`07_` war durch `07_synchronisation.md` belegt, nie committet). Das spätere Hauptdokument
bleibt `docs/08_synthese_m2.md`.

**Unverändert offen:** der unerwartete Befund aus Tabelle C (LSTM schneller als GRU in
diesem Datensatz) ist neu aufgefallen und nirgends im Laborbuch diskutiert — für die
Synthese ist zu entscheiden, ob das als offene Beobachtung mitgeführt wird.

**SY2/SY3 weiterhin erst nach Ihrer Freigabe. STOPP.**
