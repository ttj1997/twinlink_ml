# Prüfung 2: Aussagen gegen Code und Artefakt

**Stand:** 2026-10-09, Repo-Stand `c7c9621` (master, sauberer Arbeitsbaum)
**Umfang:**
- `docs/08_synthese_m2.md`: alle Zahlen im Ergebnisteil (Abschnitte 1–7 und Anhang)
- Befund 69, 73, 76 und 77 in `docs/04_stand_ap1.md`: Kernaussagen und Anwendung der vorab festgelegten Regeln
- Umrechnungen bei allen Aussagen mit Einheit

**Vorgehen:**
- Nur gelesen, nichts im Repo geändert.
- Kein Skript aus `scripts/` ausgeführt.
- Nachrechnungen nur mit eigenen Skripten außerhalb des Repos bzw. im Speicher; der Simulator lief ohne Schreibzugriff.
- Befund 73, 76 und 77 wurden direkt geprüft.
- Synthese und Befund 69 wurden von Teilprüfungen bearbeitet; deren schwerste Punkte wurden direkt nachgerechnet. Punkte, die nur die Teilprüfung nachgerechnet hat, sind mit **(TP)** markiert.

**Schweregrade:**

| Grad | Bedeutung |
|---|---|
| kritisch | ändert eine Kernaussage oder deren Deutung |
| mittel | Zahl stimmt, misst aber nicht das Behauptete, oder ein Vorbehalt fehlt |
| gering | Tippfehler, falscher Verweis, Rundung, Herkunft |

---

## Kurzfassung

- Fast alle Zahlen lassen sich aus den committeten Artefakten reproduzieren.
- Die vorab festgelegten Regeln von Befund 69, 73, 76 und 77 wurden unverändert angewendet.
- Bei den Umrechnungen gibt es keinen Fehler. Der Wirkungsgrad wird nur einmal angewendet.
- Die Probleme liegen woanders:
  1. Eine nicht berichtete Grenze des Stationärlösers setzt an warmen Punkten das Optimum (K1).
  2. Die Synthese (Stand 25.09.) fällt hinter spätere Laborbuch-Nachträge zurück (K2, M6, M7).
  3. Tabelle 2 mischt Messungen von zwei Rechnern (K3).
  4. Mehrere Quotienten haben auf beiden Seiten eine ungleiche Basis: andere Kanäle, Testmengen, Metriken oder Schätzer (M1–M4).

---

## Kritisch

### K1: Die Lösergrenze T_bed ≥ T_amb + 10 K setzt an warmen Punkten das Optimum (Befund 69, 73, 76, 77)

**Aussagen:**
- Befund 76 e: „No solution approaches the validity boundary … Where the gain lies at a bound it is reported here.“
- Befund 77 g: „the other line stays near its own optimum“
- Befund 69, Nachtrag: „the moisture limit binds at only 7 of 25 points“

**Fundstellen:**
- `docs/04_stand_ap1.md`, Z. 9651, 9718 und 9434
- Code: `src/twinlink_ml/simulation/steady_state.py`, Z. 300–302. Die untere Grenze der äußeren Temperatur-Bisektion ist fest `ambient_temperature + 10.0`.
- `src/twinlink_ml/simulation/oracle_search.py`, Z. 50ff.: `evaluate_point` übergibt bewusst kein eigenes Bracket.

**Beleg:**

Befund 76, Lauf mit Zuluft-Box 60–120 °C: die Betttemperatur liegt genau auf T_amb + 10 K.

| Szenario | wo |
|---|---|
| 15, 20, 21 | S2 und M3', beide Linien |
| 22–24 | Linie A, S2 und M3' |

Befund 76, Lauf mit Box 60–100 °C:

| Szenario | wo |
|---|---|
| 22, 23 | S2, beide Linien |
| 24 | S2, nur Linie A |

Befund 77 (im Speicher nachgerechnet, Set-Points aus den Artefakten):
- Linie A der S2-Referenz liegt in 5 von 6 W-Szenarien in beiden Boxen genau auf 40,000 °C bzw. 42,500 °C. Ausnahme ist (30,0; 0,82). Die Linie ist dort übertrocknet: X = 0,117–0,141 gegen die verschärfte Grenze 0,144.
- In der Negativkontrolle N liegen beide Linien dort.

Befund 69 (TP):
- Bei 1,26 m/s wird der Festbetriebspunkt F durch diese Grenze festgelegt, nicht durch die Qualitätsgrenze; ebenso im Stressfall 0,20.
- Von den 18 nicht bindenden Punkten bei 1,26 m/s liegen 10 an der Box-Untergrenze 60 °C (siehe M10) und 8 an dieser Lösergrenze.

Gegenprobe mit breiterem Bracket (T_amb − 5, T_in + 5), Linie A, 1,26 m/s:

| Ort | Zuluft | Standard-Bracket | breites Bracket |
|---|---|---|---|
| 30 °C / 0,60 | 68,06 °C | Fehler „kein Vorzeichenwechsel“ | T_bed 40,00 °C, X 0,129, zulässig |
| 30 °C / 0,60 | 67,06 °C | Fehler | T_bed 39,23 °C, X 0,135, zulässig |
| 35 °C / 0,55 | 72,96 °C | Fehler | T_bed 44,22 °C, X 0,108, zulässig |
| 35 °C / 0,55 | 68,96 °C | Fehler | T_bed 41,12 °C, X 0,131, zulässig |

- Unterhalb der Grenze gibt es also konvergierte, zulässige Trocknungszustände mit Residuum ~1e-10.
- (TP) Y_out ≈ 0,026 liegt weit unter Y_s(T_bed). Das ist Trocknung, nicht der Kondensationszweig, den Code-Kommentar und Laborbuch (Z. 7055ff.) als Begründung der Grenze nennen.

**Folge:**
- S2, F und M0 sind an diesen Punkten keine echten Optima, sondern durch die Lösergrenze beschnitten.
- Ob und in welche Richtung sich G (Befund 73, 76, 77) sowie Regret und Restnutzen (Befund 69) ändern, ist nicht berechnet.
- Die berichteten Zahlen selbst reproduzieren korrekt.
- Zu entscheiden ist:
  - Entweder die Grenze bleibt als bewusste Modellgrenze bestehen; dann ist sie überall als aktive Grenze zu berichten.
  - Oder es wird mit erweitertem Bracket neu gerechnet; dann sind die Zahlen in 69, 73, 76 und 77 neu zu bestimmen.

**Schwere:** kritisch für die Deutung.

---

### K2: Die Synthese nutzt noch die z-Schwellen; das Laborbuch ist seit 30.09. auf t umgestellt

**Aussagen:**
- Tabelle 1 und die Spannen im Text
- „große [Rückstände] bleiben es, wie die weiterhin auflösbaren Rückstände von Transformer, xLSTM und Mamba … zeigen“
- Transformer/LSTM, Feuchte: „auflösbar (Konstantlauf knapp)“

**Fundstellen:**
- Synthese Z. 85–110, 144f., 164f.; Anhang Z. 379
- Laborbuch Z. 8846–8884 (Nachtrag zu Befund 63, Teil von Befund 67/SX1): „The statement for the comparison of this finding follows the t thresholds from here on.“

**Beleg:**
- Mit t (Welch-Freiheitsgrade, ν_eff = 3,76) ist der Transformer-Rückstand auf `product_moisture_kg_per_kg` in beiden Testmengen nicht mehr auflösbar:

  | Testmenge | Differenz | Schwelle z | Schwelle t |
  |---|---|---|---|
  | angeregt | −98,6 % | 91,0 % | 124,6 % |
  | Konstantlauf | −45,9 % | 45,5 % | 55,9 % |

- xLSTM war auf der Feuchte schon mit z nicht auflösbar. Auf der Feuchte bleibt also nur Mamba auflösbar.
- (TP) Die t-Spannen lauten:
  - `energy_target`: 4,9–12,6 %
  - `bed_temperature_celsius`: 6,5–14,9 %
  - Feuchte: 41,3–516,8 %
- (TP) Transformer/LSTM, Feuchte, Konstantlauf bei 600 Epochen: Differenz −45,9 %, Schwelle schon mit z 53,5 %, also nicht auflösbar. `analyze_m2_resolution_verdicts.py` (Z. 246) rechnet bei 600 Epochen nur `bed_temperature_celsius` und `energy_target`; diese Zelle wurde nie gerechnet.
- Damit stimmt auch der Satz „Jede Rangaussage in Abschnitt 3 ist gegen die tatsächlich erreichte Schwelle geprüft“ (Z. 97f.) nicht mehr.

**Schwere:** kritisch.

---

### K3: Tabelle 2 und die Transformer-Kostenfaktoren vergleichen zwei Rechner

**Aussagen:**
- „nur die untereinander thread-geprüfte, protokollgleiche Datenfamilie“
- „9,5- bis 16,9-fach teurer bei Inferenz … 9,6- bis 11,4-fach teurer beim Training“

**Fundstellen:** Synthese Z. 124–127, 166–169, 214f.; Anhang Z. 385 und 387.

**Beleg (Feld `params.platform`):**

| Datei | Architekturen | Plattform |
|---|---|---|
| `arch_learning_curve_projected.jsonl` | Persistenz, Statisch, ARX, LSTM, GRU | Linux-6.8, alle 200 Records |
| `arch_learning_curve_projected_extended.jsonl` | Transformer, xLSTM, Mamba | Windows 11, alle 120 Records |

- Die Thread-Felder sind überall 1. Die Plattform hat aber niemand geprüft, auch die Konsistenzprüfung SY4 nicht.
- (TP) Auf demselben Rechner (`arch_epoch_budget_probe600.jsonl`, Windows, LSTM und Transformer):
  - Inferenz Transformer/LSTM ≈ 50× (2758 gegen 55,4 µs/Fenster)
  - Training ≈ 15× pro Epoche

**Folge:**
- Die Faktoren unterschätzen den Mehraufwand des Transformers. Die Richtung der Aussage bleibt richtig.
- Quotienten zwischen {Baselines, LSTM, GRU} und {Mamba, xLSTM, Transformer} sind nicht zulässig.
- Der Vorbehalt in Z. 239–242 (Abweichung des GRU-Werts „erklärt“ durch fehlendes Thread-Logging) ist vermutlich ein Plattformeffekt.
- Was hält: „LSTM schneller als GRU“ liegt ganz in der Linux-Familie und zeigt sich auch auf Windows.

**Schwere:** kritisch.

---

## Mittel

### M1: Faktor 6,3 gehört zur falschen Größe
- **Aussage:** „während Persistenz auf `product_moisture_kg_per_kg` um Faktor 6,3 schwankt (Befund 58 OA5a)“ (Synthese Z. 67f.)
- **Beleg:**
  - Laborbuch Z. 5352 (OA5a) nennt den Faktor 6,3 für `energy_target` bei h=1 (0,1014 gegen 0,0162).
  - (TP) Für die Feuchte ergibt sich 4,13 (h=1) bzw. 9,53 (h=36).
  - Der Gegenwert „statische Regression ~1,4“ gilt für `energy_target` bei h=36. Damit stehen verschiedene Horizonte auf beiden Seiten des Vergleichs, und der Kanal ist nicht genannt.
- **Schwere:** mittel.

### M2: „40 bis 41 Seeds“: nur ein Kanal, und die Formel widerspricht dem eigenen Absatz (TP)
- **Aussage:** „Um eine kleine Differenz von 12,5 % … wären überschlägig 40 bis 41 Seeds nötig“ (Synthese Z. 95–97)
- **Beleg:**
  - Nachgerechnet: 41 / 40. Das gilt aber nur für die Feuchte.
  - Die Rechnung nutzt nur die LSTM-Streuung und die einseitige Form, die Befund 63g als falsch korrigiert hat.
  - Mit der Zwei-Stichproben-Form auf der Feuchte:

    | Vergleich | angeregt | Konstantlauf |
    |---|---|---|
    | GRU gegen LSTM | 81 Seeds | 39 Seeds |
    | Transformer gegen LSTM | 266 Seeds | 67 Seeds |

- **Zusätzlich:** Verglichen werden Mediane, die Schwelle nutzt aber den Standardfehler von Mittelwerten (σ/√n). Bei n=5 ist der Standardfehler des Medians etwa 1,2-mal so groß.
- **Schwere:** mittel.

### M3: K5-Testmengenfalle mischt Metriken und Budgets (TP)
- **Aussage:** Faktor gegen Persistenz „1,31 (GRU)/1,70 (LSTM) auf dem Konstantlauf-Holdout, 4,4 bis 24 auf der angeregten Testmenge“ (Synthese Z. 248)
- **Beleg:**
  - 4,4 stammt aus der physikalischen RMSE, 24 aus der NRMSE. Konsistent wäre „NRMSE 4,5–24,3“; Befund 66 (Z. 8592, 8662) hat das schon festgestellt.
  - 1,31/1,70 gilt nur für die NRMSE bei n_train=64. Über n=8..64 liegt der Konstantlauf-Faktor bei 0,33–1,70: Bei n ≤ 32 sind GRU und LSTM auf dem Holdout schlechter als die Persistenz.
- **Schwere:** mittel.

### M4: Abschnitt 3.1: R² hergeleitet statt gemessen; „Faktor 20“ mischt Testmengen (TP)
- **Aussage 1:** „R² = 0,972 … R² ≥ 0,999 / ≥ 0,9989 … Faktor 5,2 bis 7,8 im RMSE“
  - R²_stat = 0,972 stammt aus `docs/02_prozess_setup.md`, anderer Datensatz und Fingerprint.
  - Die Modell-R² sind nur aus 1 − 0,028/k² gerechnet, nie gemessen.
  - Auf denselben Daten gemessen liegt der gepoolte RMSE-Faktor bei ≈ 8,5–10,8.
- **Aussage 2:** „FIR, k=43: 3,62 K gegenüber GRU (0,18 K) … Faktor 20“
  - 3,62 K gilt für den FIR auf dem Konstantlauf-Holdout ohne Horizont.
  - 0,18 K gilt für die GRU auf der angeregten Testmenge bei h=36.
  - Auf derselben Testmenge wäre der Faktor 27,6.
  - Alle 18 Records zu Befund 50j sind `git/dirty=True`.
- **Folge:** Die qualitative Aussage hält.
- **Schwere:** mittel.

### M5: Feststoffanteil 59–73 % wurde mit den Dataclass-Defaults statt der Projektkonfiguration gerechnet
- **Aussage:** Feststoffanteil „zwischen rund 59 und 73 %“ (Synthese Z. 279; Befund 56 NB2)
- **Beleg (direkt geprüft):**
  - `scripts/study_plant_solid_fraction_prediction.py` (Z. 84) und `scripts/study_plant_validation.py` (Z. 288) verwenden `ProcessConfig()` mit kritischer Feuchte 0,020 und Gleichgewichtsfeuchte 0,003.
  - `conf/process/greybox_gab_placeholder.yaml` (Z. 8f., 83f.) hat 0,200 und 0,030 und warnt: „NICHT auf 0.020 senken“.
- **(TP) Mit der YAML-Konfiguration:**
  - P1 71,5 % statt 72,6 %, P2 64,9 % statt 71,2 %.
  - Die Übereinstimmung beider Produkte bei 71–73 % fällt damit weg.
  - Die F1-Annahme „nahezu vollständige Verdampfung“ gilt für P2 nicht mehr; etwa 20 % des Sprühwassers verlassen die Anlage mit dem Produkt.
  - `cp_suspension` = 3500 J/(kg·K) steht fest (TODO in der YAML). Mit einer Mischungsregel ergeben sich grob 57–71 %.
- **Umrechnungen korrekt (TP):**
  - Nm³/h × ρN (1,2922 kg/m³)
  - relative → absolute Feuchte einmal umgerechnet
  - cp je kg Trockenluft
  - Wandverlust einmal
- **Schwere:** mittel. „Deutlich über 30 %“ bleibt.

### M6: Synthese ohne den Nachtrag zu Befund 52 vom 05.10. (Untergrenze der Gasgeschwindigkeit)
- **Aussagen:**
  - „bindet an 17 von 25 geprüften Störgrößenkombinationen“ (Z. 292)
  - Regret „Median 16 %, bis 26 %“ (Z. 346)
  - „an einer Mehrheit der geprüften Betriebspunkte“ (Z. 337f.)
- **Beleg:**
  - Reproduziert sind 17/25, 16,03 % und 25,80 % (Bruttoverbrauch, Labormaßstab).
  - Alle drei Zahlen gelten nur bei der Untergrenze 0,80 m/s, einem Platzhalter unterhalb des Laborbereichs (Laborbuch Z. 9400).
  - Bei 1,26 m/s bindet die Feuchte nur an 7 von 25 Punkten (Befund 69, Nachtrag). „Mehrheit“ ist damit nicht robust. K1 kommt hinzu.
- **Schwere:** mittel.

### M7: Sicherheitsmarge 85,7× / 31,3× mit einem später verworfenen Maßstab (TP)
- **Aussage:** Marge „85,7-fach … bzw. 31,3-fach so groß wie der dreifache Vorhersagefehler“ (Z. 353–355)
- **Was stimmt:**
  - 0,139545 / (3 · 0,000543) = 85,70 und 0,139545 / (3 · 0,001488) = 31,25
  - Beide Seiten in kg/kg absolut, gleiche Testmenge, gleicher Schätzer
- **Später eingeschränkt:**
  - Der 3. Nachtrag zu Befund 65 (30.09., Z. 8430ff.) nennt die globale Holdout-Zahl „the wrong yardstick for the operating point“.
  - An den 17 bindenden Optima ergeben sich ≈ 24,6× bzw. ≈ 13,3×.
- **Schwere:** mittel.

### M8: „Early Stopping bei ≈24 statt ≈170“ gilt nur bei kleinem Budget (TP)
- **Aussage:** „nach rund einem Achtel der Epochen (Early Stopping bei ≈24 statt ≈170)“ (Z. 257–259)
- **Beleg:**
  - Bei n=8/16 liegen die Werte bei ≈ 23–25 gegen 166–176 Epochen, das Verhältnis eher ein Siebtel.
  - Bei n=64 sind es 79 / 100 gegen 150 / 144 Epochen.
  - Befund 55 hält ausdrücklich fest, dass sich Plateau und Überanpassung nicht unterscheiden lassen. „Messung des Informationsgehalts“ geht darüber hinaus.
- **Schwere:** mittel.

### M9: Befund 77 f: „der Gewinn beginnt zwischen 0,014 und 0,016 kg/kg“ wird von den eigenen Daten widersprochen
- **Fundstelle:** Laborbuch Z. 9716
- **Beleg (direkt):**
  - Bei 0,0120 kg/kg (20 °C, φ 0,82, Gruppe H) liegt G bei 0,94–0,99 kg/h.
  - Befund 73 hat ähnliche Gewinne schon bei geringerer absoluter Feuchte:
    - Szenario 10 (20 °C/0,25, ≈ 0,0036 kg/kg): G = 1,24 kg/h
    - Szenario 14 (20 °C/0,85, ≈ 0,0125 kg/kg): G = 1,50 kg/h
  - Der Vergleich „gleich groß bei gleicher absoluter Feuchte“ (5,6 / 5,8 / 7,0 kg/h) nutzt Median- bzw. Bestwerte über die Seeds. Das Kriterium K_lab nutzt das Minimum; bei 30 °C/0,60 sind das 2,88 statt 7,0. Der Schätzer ist nicht genannt.
- **Folge:** Haltbar ist nur: Die großen Gewinne (> ~5 kg/h) treten erst ab etwa 0,016 kg/kg auf.
- **Schwere:** mittel (Deutung, als „evidence, not proof“ markiert).

### M10: Befund 69: Untergrenze der Zuluft-Box 60 °C bindet bei 1,26 m/s, nicht genannt (TP)
- M0 liegt an 10 von 25 Punkten auf T_in = 60,000 °C, M an 6 von 25.
- 60 °C ist ein Platzhalter (`docs/02_prozess_setup.md`, Z. 64); das Labor fährt 50–100 °C.
- **Schwere:** mittel.

---

## Gering

### Befund 76
1. **76 e** (Z. 9651):
   - „at S2 of 22–24 both lines are 0.000–0.017 below it“ stimmt nicht. Linie A liegt 0,041 / 0,029 / 0,013 darunter; sie sitzt auf der Lösergrenze, siehe K1.
   - Der Abstand zur Kinetik-Grenze bei S2 beträgt 0,056–0,097, nicht 0,056–0,073.
   - „distance at least 0.11“ gilt nur für die heiße Linie; die andere Linie liegt bei 18 und 19 bei 0,056.
2. **76 c** (Z. 9627): „over the 25 pairs of seeds its value varies by at most 0.07 kg/h at 22–24“ stimmt nicht. Laut Artefakt sind es 0,012 / 0,009 / 0,013. Der Wert 0,071 gehört zu Szenario 16.
3. **76 d** (Z. 9639): Der Quotient vergleicht das Minimum über die Seeds bei 100 °C mit dem Maximum bei 120 °C, also verschiedene Schätzer. Max/Max ergibt gerundet dieselben 69 / 81 / 65 %, die Wirkung ist null.
4. **Nachtrag zu Befund 76** (Z. 9682): Der T_kalt-Sweep 15/20/25/29,1/35 °C steht in Befund 57 h (NB1, Z. 5171ff.), nicht in Befund 52.
5. **Regel-Anwendung:**
   - Pro Szenario wurde das Kriterium unverändert angewendet. Die Skripte wurden vor den Läufen committet (c5c3ccc vor 66b195a; 149b4e8 vor 88b92dc, d4999fd und a3033ae).
   - Das vorab festgelegte Gesamtkriterium (mindestens 3 von 5) bezog sich auf die versehentlich gewählte Liste 16, 17, 22, 23, 24. Das Ergebnis dort ist 4 von 5, CONFIRMED (`m3x_confirm_evaluation_umin126`: `overall_confirmed: true`). Dieses Urteil steht nicht im Laborbuch.
   - Für 18, 19, 22, 23, 24 gibt es nur die Aussage pro Szenario (5 von 5).
6. **76 g:** „`coupling_gain_verdict` UNCHANGED“ stimmt für die Funktion. Die Eingaben wurden aber geändert:
   - G mit dem besten von 5 Seeds statt Seed 0
   - Streuung über 5 Seeds × 25 Szenarien statt 3 × 3
   - Festgelegt in 149b4e8, nachdem die Rohdaten von Lauf 1 vorlagen
   - Der relative Median nutzt die Linien-CO2 aus Befund 73 als Nenner.
   - Das Urteil bleibt in jedem Fall negativ.

### Befund 77
7. **77 g** (Z. 9718): „The gas velocity is 1.3–1.4 m/s in the rounded printout“ stimmt nicht. 10 von 12 Linien-Sollwerten der besten M3'-Lösungen liegen genau bei 1,26 m/s, zwei bei 1,36 bzw. 1,39 m/s (die nicht heiße Linie bei (30; 0,82) bzw. (32,5; 0,82)).

### Befund 69 (TP)
8. Die Quote an Punkt 24 ist entartet (3908), weil F − M0 ≈ −7,8e-7 kg/h ist. Sie geht in den Median ein; die Wirkung beträgt 0,2 Prozentpunkte.
9. Abweichungen vom Plan in docs/09:
   - Der Anker „Befund 64“ und der RL-Vergleich auf demselben Testset fehlen.
   - `eta_gas` = 0,80 ist fest gesetzt, der Default in HeatNetworkConfig ist 0,90. Das ist als Setzung dokumentiert.
10. Rundung: Der absolute Regret-Median in kWh beträgt 0,128 und nicht 0,129.

### Synthese (TP)
11. **Herkunft:**
    - Die zitierten Commits (0b3eaaf, 256d644, 47c5292, 7e3fb1d, 0ed48e5, 9652942, bc6b097, f3d9a7a, b0ae64a, a0bbc19) sind seit der neu angelegten Historie (7384455) nicht auflösbar. Die `git/commit`-Felder der Records stimmen aber.
    - `analyze_m2_resolution_verdicts.json` hat keinen Provenienzblock.
    - Die Artefakte zu Befund 39, ARX/NARX und 50j stammen aus `dirty`-Läufen.
12. Die Kostenfaktoren 9,5–16,9 und 9,6–11,4 stehen in keinem Befund. Das verstößt gegen die Regel aus CLAUDE.md, nur Zahlen aus Befund und Artefakt zu verwenden.
13. Der laut Befund 60 (Z. 5752) zwingende Vorbehalt „untere Schranke, keine Bauvorschrift“ fehlt bei der Pinch-Aussage (Z. 363–366).
14. „Fingerprint der vollständigen Konfiguration“ ist überzogen: `process/fingerprint` deckt nur die Prozesskonfiguration ab.
15. Die Vorzeichenkonvention der Prozentangaben (Referenz LSTM, negativ = Kandidat schlechter) wird nicht erklärt.
16. Die Platzhalterliste in Abschnitt 6 ist unvollständig: Es fehlen die Untergrenze 0,80 m/s und die Doppelrolle der kritischen Feuchte als Kinetikparameter und Qualitätsgrenze.
17. Befund 55 selbst (Z. 4622): Für GRU gilt 2,2–2,5, nicht 2,2–3,1. Außerdem unterscheiden sich die `split/digest` der beiden Dateien, der Holdout ist aber derselbe.
18. Die Zahl „10 bis 15 %“ der M2-Definition steht nicht in der formalen Definition, sondern in einer Notiz im Steckbrief und im Arbeitsplan (3 Seeds, NRMSE, Persistenz-Referenz). Sie ist mit den Schwellen aus Tabelle 1 nicht direkt vergleichbar.

---

## Bestätigt

### Befund 73 (direkt)
- Tabelle d: alle 25 Werte G, G_set, G_alloc und relatives G stimmen mit `m3_joint_evaluation_umin126` überein.
- Median 0,384; 13 von 25 über 0,2; 9 über 1,0; 5 über 5,0.
- Seed-Streuung 2,794 (Szenario 17: 1155,50 / 1152,71 / 1153,60); 3 × Streuung = 8,38.
- Relativer Median 0,23 %, Maximum 10,69 %.
- Wirkung der Grenze c = 0,5 / 1,0 / 1,5 (Median, Spanne, Anzahl positiv) und Mediane 0,004 / 0,001 / 0,145.
- Genauigkeitsregel 12 von 12; Prüfung mit sechs Variablen −0,082.
- G ist aus den CO2-Werten gerechnet; Strafwert und CO2 sind identisch, alle Lösungen zulässig.
- Die Regel `coupling_gain` wurde unverändert angewendet.

### Befund 76 (direkt)
- Tabellen c und d: alle S2- und M3'-Werte je Seed sowie Minima und Maxima von G.
- Kriterium bei 100 °C: 5 von 5.
- Robustheitsurteil: Median 0,560 (ebenso mit dem Median-Seed), Streuung 2,223 (Szenario 22), 14 von 25, relativ 0,35 %.
- Abschnitt f, Szenario 19: Stufenleistungen und CO2-Anteile reproduziert. Netto −21,56 + 12,52 − 0,39 = −9,43 kg/h.

### Befund 77 (direkt)
- K_lab 6 von 6 und K_box 6 von 6. Alle Minima und Maxima, Sollwerte und Feuchten der heißen und der anderen Linie, Negativkontrollen und Gruppe H stimmen.
- `scripts/run_m4_confirm_test.py` ist seit dem Vorab-Commit d09f8ce unverändert, ebenso `src/`.
- Provenienz der Artefakte: d09f8ce bzw. 29db2cc, `dirty=False`.
- Absolute Feuchten (0,0160–0,0256, 0,0091/0,0093, 0,0120/0,0164) mit Magnus auf ±0,0001 nachgerechnet.

### Einheiten (direkt)
- Brennstoffleistung E_gas = Q / η_H, mit m_gas = Q/(H_u·η_H) und E = m_gas·H_u (`heatnetwork/balance.py`, Z. 648; `site_network.py`, Z. 318).
- CO2 = 0,366 kg/kWh · E_el + 0,201 kg/kWh · E_gas (`evaluation/net_energy.py`).
- Der Wirkungsgrad wird nur einmal angewendet.
- Verschärfte Grenze: 0,15 − 0,005962 = 0,144.

### Befund 69 (TP)
- Alle Zahlen, darunter: Kosten der Marge 1,295 / 1,064 %, Restnutzen 85,42 / 93,79 %, Optimum auf der Gasgrenze 21–23 von 25.
- Nachtrag: 7 von 25, Regret 19,8 / 17,6 %.
- M0-Kosten.
- Vorab festgelegte Stabilitätsregel vor dem Lauf committet (2e42876); alle Läufe `dirty=False`; Anker s05 bitgenau reproduziert.
- Wirkungsgrad: Prozessmodell P_H = Q / 0,98, Gasbuchung Q·0,98 / 0,80, also genau einmal.

### Synthese (TP)
- Tabelle 1 (alle 24 Zellen, z-Variante)
- GRU/LSTM 6 von 6; Transformer 1,026 / 1,000 / 1,082 / 1,158 mit Schwellen
- xLSTM/Mamba 10 von 12; ARX −7,8 %, NARX 69 → 12 %
- Tabelle 2 (Werte); 33 % mehr Parameter, Faktor 1,77
- 29–68 %; 1,31 / 1,70 (NRMSE, n=64); Faktor 2,2–3,1 (NRMSE)
- Variationskoeffizient 0,12–2,01 %; 600 s
- 17/25; 16 % / 26 %; 85,7 / 31,3; Pinch 2,25–4,16 %
- Platzhalter 0,20 kg/kg / 80 °C / 750 µm

---

## Empfohlene Reihenfolge (nach CLAUDE.md: erst berichten, ändern nur nach Bestätigung)

1. **K1** entscheiden: Bleibt die Lösergrenze als Modellgrenze und wird berichtet, oder wird mit erweitertem Bracket neu gerechnet? Das betrifft Befund 69, 73, 76 und 77.
2. **K2 und K3** in der Synthese: auf t-Schwellen umstellen und die Plattformen trennen.
3. Die mittleren Punkte als datierte Nachträge (Laborbuch, nur anhängen) bzw. in der nächsten Fassung der Synthese.
