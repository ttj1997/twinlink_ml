# Prüfung 1: Blinde Nachableitung von F1–F3

Stand: master `c7c9621`, 2026-10-09. Repo unverändert. Teilbare Fassung: https://claude.ai/artifact/LMATkyR4LMMvXwaceWXzG2

**Ergebnis in einem Satz:** Die Zahlen im Laborbuch stimmen in allen drei Fragen bis auf Rundungen; die
wichtigste Abweichung betrifft die Deutung: Bei 1.26 m/s setzt die untere Suchgrenze des
Steady-State-Lösers (T_bed ≥ T_amb + 10 K, `src/twinlink_ml/simulation/steady_state.py:300`) den
Fixpunkt und mehrere Optima, und das Laborbuch nennt sie nirgends. Das betrifft F2 deutlich und F3
teilweise.

**Vorgehen.** Eigene Ableitung mit Zahlen zuerst schriftlich festgehalten, danach Befunde 58, 63, 69,
71, 73, 76, 77 und docs/08 gelesen; docs/09 und 11 nur gezielt durchsucht, um Vorab-Festlegungen zu
prüfen. Die Blindheit war nicht vollständig: Einige Docstrings der erzeugenden Skripte zitieren
Befund-Aussagen (z. B. `scripts/analyze_m2_resolution_verdicts.py`). Zwei Löser-Proben liefen außerhalb
des Repos.

Schwere-Skala: **hoch** = trägt eine Kernaussage; **mittel** = Deutung oder Formulierung zu stark;
**gering** = Präzisierung; **trivial** = Tipp- oder Rundungsfehler.

---

## Querschnittsbefund (F2 und F3): die Bracket-Grenze bindet bei warmer Umgebung

Der Löser sucht T_bed nur ab T_amb + 10 K. Liegt die Lösung darunter, scheitert er, und das Oracle
wertet den Punkt als Lösungsfehler. Bei 1.26 m/s liegt das M0-Optimum an 8 von 25 Punkten exakt auf
ΔT = 10.000 K (27.5 °C/φ 0.25–0.55, alle 35 °C-Punkte); der Fixpunkt F (74.25 °C) liegt bei
35 °C/0.25 genau dort (F = M0). `constraint_check` meldet das an keinem Punkt als bindend.

Probe mit erweitertem Bracket (T_amb + 3), u = 1.26, p = 1.0, Konfiguration
`conf/process/greybox_gab_placeholder.yaml`:

| Punkt (T_amb/φ) | T_in [°C] | Default-Löser | Bracket T_amb+3: T_bed [°C], Residuum | X [kg/kg] |
|---|---|---|---|---|
| 35.0 / 0.25 | 74.25 | T_bed = 45.00 (Rand) | 45.00 | 0.0842 |
| 35.0 / 0.25 | 72 | scheitert | 43.20, 1.7e-10 | 0.0918 |
| 35.0 / 0.25 | 70 | scheitert | 41.61, 3.7e-11 | 0.0994 |
| 35.0 / 0.25 | 66 | scheitert | 38.46, 3.6e-10 | 0.1171 |
| 27.5 / 0.25 | 62 | scheitert | 34.75, 6.6e-10 | 0.1312 |

Das sind gültige Beharrungszustände im Trocknungsregime (Feuchte unter 0.15, T_bed 3.5–8 K über
Umgebung). Die Begründung im Laborbuch (Z. 7055–7078, die 10-K-Marge halte das Modell im gültigen
Regime) trifft für die kalte, langsame Ecke zu, nicht aber hier.

---

## F1 Architekturvergleich (`arch_learning_curve_projected*.jsonl`)

**Meine Ableitung** (`rmse_run_median_physical`, 5 gepaarte Seeds):

- GRU oder LSTM ist in allen 48 Zellen (Kanal × Horizont × Testmenge × Budget) das beste trainierte
  Modell; dahinter Transformer, dann ARX, Mamba, xLSTM.
- Beispiel n=64, angeregt, Betttemperatur h36: GRU 0.428 K, LSTM 0.429 K, Transformer 0.499 K,
  xLSTM/Mamba 1.54 K, ARX 2.19 K, Persistenz 6.41 K.
- GRU gegen LSTM praktisch nirgends auflösbar (Zwei-Stichproben-Schwelle aus `resolution.py`).
  Transformer-Rückstand auf der angeregten Testmenge auflösbar (Faktor 1.11–1.99), auf der
  Betttemperatur im Konstantlauf nicht. xLSTM gegen LSTM auf der Feuchte h36 angeregt: 2.40×, nicht
  auflösbar (Schwelle 3.57).
- Parameterzahlen nicht abgeglichen (GRU 42 819, LSTM 57 027, Transformer 23 523, xLSTM 8 649,
  Mamba 8 139); Transformer erreicht bei n ≥ 16 meist die Epochengrenze 200.

**Abgleich:** Befund 58, 63 mit Nachtrag und docs/08 Kap. 3 decken sich damit. Monotonie-Zählung
(GRU 2/12, LSTM 1/12, Transformer 1/12, xLSTM 4/12, Mamba 4/12, static 5/12, ARX 7/12) exakt
reproduziert.

| # | Aussage | Ableitung und Beleg | Schwere |
|---|---|---|---|
| F1.1 | docs/08 Kap. 4: Faktor gegen Persistenz auf der Feuchte im Konstantlauf „1,31 (GRU) / 1,70 (LSTM)“ | Stammt aus `nrmse_run_median` (0.4466/0.3409 bzw. /0.2623). Diese Metrik ist auf Konstantläufen instabil (Nenner gegen null; `static` dort 7.7–18). Physikalisch (n=64, h36): 1.89 und 2.27. Widerspricht der Regel, in physikalischen Einheiten zu vergleichen. | gering–mittel |
| F1.2 | docs/08: „xLSTM und Mamba liegen zurück“ (gegen LSTM) | Auf dem Konstantlauf-Holdout, den Befund 58 OA5a als betriebsrelevant bezeichnet, sind bei n=64 mehrere Modelle schlechter als Persistenz: ARX 6/6 Zellen, Mamba 6/6, xLSTM 3/6, Transformer 1/6 (Feuchte h1: 1.43× Persistenz). GRU/LSTM bei h1 nur 7–28 % besser als Persistenz. In der Synthese nicht genannt. | gering–mittel |
| F1.3 | Befund 63: GRU/LSTM „durchgehend nicht unterscheidbar“ | Geprüft wurde nur h=36. Bei h=1, Betttemperatur, Konstantlauf ist GRU auflösbar besser (Verhältnis 0.96, Schwelle 2 %). Praktisch ohne Bedeutung, „durchgehend“ aber zu weit. | gering |
| F1.4 | Befund 58: „Konfiguration – bitgenau Befund 51“, „protokollgleich“ | Daten, Split, Seeds identisch. Training aber verschieden: GRU/LSTM Linux, torch 2.13+cu130, numpy 2.4.6, Commit `0b3eaaf`; Erweiterung Windows, torch cpu, numpy 2.5.1, Commit `256d644`. Nicht benannte Konfundierung zwischen den Modellfamilien. | gering |
| F1.5 | Befund 58 OA5b: „24 Zellen je Architektur“ | Es sind 12 (2 × 3 × 2); die Brüche „/12“ im selben Absatz stimmen. | trivial |

---

## F2 Kosten der Marge und Regret bei 1.26 m/s (`m1_margin_cost_l015_umin126`, `m1_gas_bound_sensitivity`)

**Meine Ableitung** (Linie A, Gaserhitzer, Grenze 0.15, m = 0.00596 kg/kg, 25 Gitterpunkte Befund 52):

- Regret F gegen M0 (CO2): Median 19.8 %, gepoolt 17.6 %, Maximum 27.9 %; absolut Median 0.070 kg/h.
- Margenkosten: Median über 25 Punkte 0; an den 7 feuchtebindenden Punkten Median 1.60 %, Maximum
  1.82 %; gepoolt 0.55 %, das sind 3.1 % des Regrets.

Was das M0-Optimum bei 1.26 setzt:

| Grenze | Punkte von 25 | im Laborbuch genannt |
|---|---|---|
| u_gas an der unteren Grenze 1.26 | 25 | ja |
| Feuchtegrenze bindet | 7 | ja |
| T_in am Box-Minimum 60 °C (Platzhalter; Labor 50–100 °C) | 10 | nein |
| T_bed − T_amb = 10.000 K (Löser-Bracket) | 8 | nein |

**Abgleich:** Zahlen stimmen mit Befund 69 und Addenda. Die Stabilitätsregel ist echt vorab
festgelegt (`2e42876`, der Lauf trägt diesen Commit); der Nachcommit `982c394` ändert nur die
Datenzuführung (Diff geprüft).

| # | Aussage | Ableitung und Beleg | Schwere |
|---|---|---|---|
| F2.1 | Befund 69 Addendum: „the margin costs nothing where it does not bind“; Laborbuch Z. 7075: die 10-K-Marge halte das Modell im gültigen Trocknungsregime | An den 18 nicht feuchtebindenden Punkten bindet etwas anderes: an 10 Punkten T_in = 60, an 8 Punkten die Löser-Grenze. Bei warmer Umgebung schneidet diese Grenze gültige Zustände ab (Probe oben). Lage von F und damit der Regret bei 1.26 hängen an einer Löser-Konvention. Größe des Effekts nicht quantifiziert; Richtung gemischt (F würde sinken, die heißen M0-Punkte ebenfalls). | **hoch** |
| F2.2 | Zweites Addendum: „The lower bound [u_gas] determines the optimum, independent of its value“ | Gilt nur für die u-Dimension. Die T_in-Dimension setzen bei 1.26 drei andere Grenzen: Box-Minimum 60 °C (Platzhalter), Feuchte, Löser-Grenze. Bindung an T_in ≥ 60 an 10 Punkten nicht genannt. | mittel |
| F2.3 | Addendum (c): „at the laboratory minimum the saving is larger (regret about 18-20 %) than at the placeholder (about 8 %), not smaller“ | Wegen F2.1 und F2.2 stärker formuliert, als die Daten tragen: der Wert hängt zusätzlich an T_in ≥ 60 und am Löser-Bracket. | mittel |
| F2.4 | Addendum, Tabelle: Regret p95 = 27.4 % | Artefakt: 0.27346, also 27.3 %. | trivial |

---

## F3 Kopplungsgewinn zweier Linien (`m3_*`, `m3x_*`, `m4c_*`)

**Meine Ableitung** (G = CO2(S2) − CO2(M3′) in kg CO2e/h):

- Typisch (Kontrolle, 25 Szenarien): Median G 0.38 (Hauptlauf) bzw. 0.56 (robuste Nachmessung), das
  sind 0.23–0.35 % der Linien-CO2; alle drei vorab festgelegten Kriterien verfehlt.
- Feucht-warm (18, 19, 22, 23, 24): 5.5–17.6, Minimum über 5 Seeds; 3.4–10.8 % der Linien-CO2.
  Bei T_max 100 °C: 4.1–11.4; die Größe hängt bei 22–24 am Box-Rand, die Existenz nicht.
- M4-C: K_lab 6/6 (Minimum 2.88–11.33), K_box 6/6 (3.76–15.77); Negativkontrollen N: −0.17 und
  −0.11 (≈ 0); H: 20/0.82 → 0.94, 25/0.82 → 5.77.

**Abgleich:** Zahlen stimmen mit Befund 73, 76, 77. Explorativer Status sauber deklariert, ebenso
die Verwechslung der Szenarien 16/17 mit 18/19 und die nachträglich ergänzte Spalte zum
Allokationsgewinn. M4-C echt vorab festgelegt (`d09f8ce` vor den Läufen).

| # | Aussage | Ableitung und Beleg | Schwere |
|---|---|---|---|
| F3.1 | Befund 76 e): „No solution approaches the validity boundary…“ (geprüft: Kinetik, 80 °C, Box) | Im S2-Arm sitzt Linie A auf der Löser-Grenze ΔT = 10.000 K: bei 22, 23, 24, in 5 von 6 W-Szenarien und in beiden N-Szenarien (eigene Nachrechnung der S2-Sollwerte). Nicht unter den geprüften Grenzen. Die Existenz des Gewinns hängt nicht daran (18, 19, H 25/0.82 nicht gepinnt: 5.5 / 9.3 / 5.8). Größe bei 35 °C und die Lesart der Negativkontrolle sind teils lösergebunden. | mittel |
| F3.2 | Überschrift Befund 77: „the gain follows the absolute humidity“ | Text vorsichtiger (Indiz, zwei H-Punkte). Daten nicht monoton: Y 0.0186 → 5.0 gegen Y 0.016 → 7.0; Y 0.0224 → 8.5 gegen Y 0.0221 → 10.05. Unterer Anker der Einsatzschwelle (Y 0.0141 → +0.006) ist ein Tabellensuch-Punkt bei 35 °C, an dem die Linien auf dem Bracket liegen. | gering–mittel |
| F3.3 | Befund 77 h): „not an artefact of the selection that found them“ | W-Punkte (30/32.5 °C, φ 0.60–0.82) liegen innerhalb des explorativen Bereichs desselben deterministischen Modells. Schließt Gitter- und Suchselektion aus, ist aber keine unabhängige Replikation. „Confirmation in the simulator“ ist benannt. | gering |
| F3.4 | Befund 73 a): „The control is the main result (project lead)“ | Festgelegt in `61ce8cf` (06.10., 09:57), nach den Ergebnissen von Stage 0/0b (09:25/09:35); in Befund 71 war die Kontrolle noch der Build-Check. Vor den gemeinsamen Läufen und bei negativem Verdikt, also keine Verzerrung zugunsten des Gewinns; der Zeitpunkt ist aber nicht offengelegt. | gering |
| F3.5 | Befund 77 g): u „1.3–1.4 m/s in the rounded printout, not checked to the third digit“ | In 8 von 10 Szenarien exakt 1.26; abweichend nur Szenario 2 (Linie A 1.362) und 5 (Linie B 1.387). | trivial |

---

## Empfehlung (nichts davon umgesetzt)

1. **Löser-Grenze zuerst.** M1 bei 1.26 erneut rechnen, mit einem Bracket, das unterhalb von
   T_amb + 10 K gültige Trocknungszustände zulässt. Diese Grenze bindet in F2 und F3 an denselben
   warmen Punkten.
2. Danach entscheiden, ob Befund 69 (Regret bei 1.26) und Befund 76 e) einen datierten Nachtrag
   brauchen.
3. In docs/08 den Persistenz-Faktor auf der Feuchte (F1.1) in physikalischen Einheiten angeben und
   die Persistenz-Unterschreitung auf dem Konstantlauf (F1.2) nennen.
