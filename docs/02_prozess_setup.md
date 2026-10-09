# Prozess-Setup: Wirbelschichtsprühgranulation

> **Diese Datei ist die wichtigste im Projektwissen.** Ohne konkrete Wertebereiche und
> Zeitkonstanten bleibt jede Modellierung generisch.
>
> Stand: 13.08.2026 · Fassung 3. **Der bisherige Nachtrag ist in diese Datei eingearbeitet
> und damit hinfällig** — eine Datei mit drei Anhängen wird nicht gelesen, sondern
> überflogen. Der Umsetzungsstand des ML-Teils steht in `04_stand_ap1.md`.
>
> Neu in dieser Fassung: die Werte der SPE-Laboranlage aus Kräuter, Zhou & Heinrich
> (ABS23, TwinGuide) — die erste Quelle im Projekt, die keine Platzhalter enthält.

## Status der Zahlenwerte — Legende

| Marke | Bedeutung |
|---|---|
| `MESS` | gemessen oder von einem Partner bestätigt |
| `LAB` | **neu**: gemessen an der SPE-Laboranlage im Vorprojekt TwinGuide, aber am **Trockner** und mit MCC-Pellets, nicht an der Granulation und nicht bei Anwendungspartner |
| `PLH` | Platzhalter, für den Architekturvergleich brauchbar, für Absolutaussagen **nicht** |
| `MODELL` | aus dem Grey-Box-Modell berechnet — gültig unter den `PLH`-Annahmen |
| `TODO` | fehlt vollständig |

`LAB` ist zwischen `MESS` und `PLH` einzuordnen und ausdrücklich **nicht** dasselbe wie
`MESS`: die Werte stammen aus einem Trocknungsprozess ohne Feststoffaufbau, mit
Cellets® 200 als inertem Bettmaterial, im Chargenbetrieb und im Labormaßstab. Für
Größenordnungen, Wertebereiche und Zeitraster sind sie belastbar; für die Granulation bei
Anwendungspartner sind sie eine Analogie.

Die maßgebliche, versionierte Fassung aller Werte ist
`conf/process/greybox_placeholder.yaml`. Diese Datei ist die fachliche Begründung dazu.

---

## Prioritätenliste der offenen Zahlen

Reihenfolge nach Wirkung auf die Versuchsplanung. Gegenüber Fassung 2 sind drei Punkte
erledigt und zwei neu.

| # | Fehlende Zahl | Wer | Warum sie zuerst kommt |
|---|---|---|---|
| 1 | **Maximale Änderungsraten der Stellgrößen** | Anwendungspartner, SPE | Sie legen die realisierbare Anregungsbandbreite fest. Mit den `PLH`-Werten ist die Ratenbegrenzung dauerhaft aktiv: das Signaldesign wird durch die Aktordynamik ersetzt. |
| 2 | **Inline-Messbarkeit der Produktfeuchte bei Anwendungspartner** | Anwendungspartner | **Neu auf Platz 2.** Der Architekturvergleich zentriert sich auf die Feuchte; ob sie gemessen wird, entscheidet, ob es eine Vorhersage- oder eine Soft-Sensor-Aufgabe ist. Beide sind implementiert (siehe unten), aber sie haben verschiedene Baselines, verschiedene Primärmetriken und verschiedene Ergebnisse. Im SPE-Labor ist die Messung vorhanden (`LAB`, Mikrowellenresonanz). |
| 3 | **Sorptionsisotherme des tatsächlichen Stoffsystems** | SPE | **Aufgewertet.** Im Vorprojekt war die falsche Isotherme (Cellets® 500 statt 200) die *Hauptursache* dafür, dass die Dyssol-Vorhersage der Feuchte unbrauchbar war. Das ist keine Feinjustierung, sondern der Unterschied zwischen R² = 0.16 und brauchbar. |
| 4 | **Zeitkonstante des Lufterhitzers** | Anwendungspartner, SPE | Damit läge die Aktordynamik im Prozessmodell statt in der Anregung. Ohne sie ist `dE` zu 97 % durch eine statische Abbildung erklärt. |
| 5 | **Definition von `dE`** | Verbund | Momentanleistung [kW] oder spezifischer Verbrauch [kWh/kg]? Die Monotonie-Constraints gelten universell nur für die Momentanleistung. |
| 6 | Rauschniveau der Anlagensensoren | Anwendungspartner | **Abgestuft.** Die Rangfolge der Modelle ist über zwei Größenordnungen des Rauschniveaus stabil (Rauschstudie aus Schritt 5, siehe `04_stand_ap1.md`, Abschnitt „Das Sensormodell ist zu gutmütig"). Entscheidend bleibt sie nur für `d_50`. |
| 7 | Betriebsart und Austragsintervalle | Anwendungspartner, SPE | Kontinuierlich mit Füllstandsregelung oder getakteter Austrag? Im SPE-Labor: **Chargenbetrieb** (`LAB`). Ändert eine Zeitkonstante des Systems. |
| 8 | Zyklon- und Rückführungsverhalten | Anwendungspartner | Im Standard-Anregungsdesign liegt `u_gas` in 13 % der Abtastschritte über `u_t` des mittleren Korns. |
| 9 | Lower limit of the product moisture | Anwendungspartner | New (Befund 77, `09_entwurf_modellbasierte_optimierung.md`). The coupling gain of the two-line optimum requires one line to dry its product to 0.043-0.080 kg/kg against the upper limit 0.144 kg/kg. If a lower limit exists, the mechanism may fall away. |
| 10 | Condensing heat recovery from the humid exhaust | Anwendungspartner | New (Befund 77). The gain requires a condensing recovery; the apparatus (condensate drain, exchanger material, fouling, pressure loss) is not modelled. Is it feasible in the plant, and at what cost? |

**Erledigt gegenüber Fassung 2:** Zeitschrittweite der Anlagendaten (2 s, `LAB`),
Rechenzeit pro Dyssol-Lauf (siehe unten, `LAB`), Frage nach der geregelten Größe bei
`u_gas` (Volumenstrom, `LAB`).

---

## Variablenstruktur

### Stellgrößen (Actions, vom Regler manipulierbar)

| Symbol | Größe | Einheit | Min | Max | Betriebspunkt | Max. Änderungsrate | Status |
|---|---|---|---|---|---|---|---|
| `T_in` | Zulufttemperatur | °C | 60 | 120 | 90 | 0.5 K/s | `PLH`; Labor 50…100 °C (`LAB`) / Rate `TODO` |
| `u_gas` | Gasgeschwindigkeit | m/s | 0.8 | 2.5 | 1.6 | 0.05 m/s² | `PLH`; Labor 1.26…1.88 m/s (`LAB`) / Rate `TODO` |
| `m_spray` | Sprührate | kg/h | 0.4 | 3.0 | 1.5 | 0.02 (kg/h)/s | `PLH`; Labor 0.6…3.9 kg/h (`LAB`) / Rate `TODO` |
| `p_nozzle` | Düsendruck | bar | 1.0 | 4.0 | 2.5 | 0.05 bar/s | `PLH`; Labor 1.0…1.8 bar (`LAB`) / Rate `TODO` |

**Die Platzhalterbereiche sind durch die Laborwerte bestätigt** — mit einer Ausnahme: der
Düsendruck wird im Labor nur zwischen 1.0 und 1.8 bar gefahren, nicht bis 4.0. Das ist
mehr als eine Bereichskorrektur: da `p_nozzle` im Grey-Box-Modell reine Kostenwirkung ohne
Prozessnutzen hat, vergrößert ein zu weiter Bereich künstlich den Spielraum, den ein
RL-Agent dort zu holen glaubt.

**Zu `u_gas`: die Anlage regelt den Volumenstrom, nicht die Geschwindigkeit** (`LAB`).
Damit ist der offene Punkt aus Fassung 2 entschieden, und die Konsequenz ist unangenehm:
bei konstantem Volumenstrom ändert sich mit `T_in` die Gasdichte und damit der
Luft**massen**strom — also die Kopplung zu `dE` und zur Trocknungskapazität. Im Modell ist
`u_gas` die Leerrohrgeschwindigkeit bei Bettbedingungen; für den Abgleich mit Anlagendaten
ist die Umrechnung über die Zustandsgleichung nötig, und sie ist **nicht** eine Konstante.
Umrechnung im Labormaßstab (0.0177 m² angenommen): 80 m³/h → 1.26 m/s, 120 m³/h →
1.88 m/s. Der Nominalwert 1.6 m/s des Platzhalters liegt mitten darin.

**Ratenbegrenzung, unverändert:** aus `t_switch = Amplitude / max. Änderungsrate` folgt die
minimal realisierbare Umschaltzeit je Kanal. Bezogen auf die schnellste Prozessmode
(`tau_fast`, siehe Abschnitt Zeitkonstanten) kann nur `u_gas` sie überhaupt anregen. **Die
effektiv schnellste beobachtbare Zeitskala wird von den Aktoren gesetzt, nicht vom
Prozess.**

**Harte Betriebsgrenzen (Safety-Constraints für RL)**

Fluidisierungsfenster `u_mf < u_gas < u_t`, Wen-Yu bzw. Haider-Levenspiel,
`rho_p = 1500 kg/m³` (`PLH`), Sphärizität 0.9, 60 °C — Status `MODELL`:

| `d_50` | `u_mf` | `u_t` |
|---|---|---|
| 150 µm (Keime) | 0.010 m/s | 0.62 m/s |
| 400 µm (Startbett) | 0.070 m/s | 2.10 m/s |
| 750 µm (Zielkorn) | 0.222 m/s | 3.57 m/s |
| 1000 µm | 0.350 m/s | 4.36 m/s |

**Das Fenster ist zustandsabhängig.** Zwischen 400 und 750 µm wächst `u_mf` um Faktor 3;
ein fester unterer Grenzwert führt am Chargenende zum Zusammenbruch der Fluidisierung.
Für RL: die Action-Bounds sind eine Funktion des Zustands.

Weitere Grenzen: Abluftfeuchte ≤ Sättigungsfeuchte (strukturell erfüllt),
Partikelfeuchte ≥ Gleichgewichtsfeuchte, max. Bettfeuchte vor Agglomerationsrisiko
`TODO`. Im Labor wurde die Überfeuchtungsgrenze **absichtlich überschritten** (Charge 4:
niedrige Betttemperatur, niedriger Luftstrom, hohe Sprührate → 22 % Feuchte, `LAB`) — es
gibt also einen gemessenen Referenzpunkt für den Zusammenbruch, wenn auch nur einen.

### Störgrößen (exogen, messbar aber nicht steuerbar)

| Symbol | Größe | Einheit | Bereich | Nominal | Status |
|---|---|---|---|---|---|
| `T_amb` | Umgebungstemperatur | °C | 5 … 35 | 20 | `PLH`; Labor 20.5 … 24.5 (`LAB`) |
| `phi_amb` | Umgebungsfeuchte (rel.) | – | 0.25 … 0.85 | 0.50 | `PLH`; Labor 0.08 … 0.70 (`LAB`) |
| `w_solid` | Suspensionsfeststoffanteil | – | 0.20 … 0.40 | 0.30 | `PLH` (Labor: reines Wasser, also 0) |

**Der Labor-Bereich der Umgebungsfeuchte ist weiter als der Platzhalter** (bis 0.08 herab).
Das ist relevant, weil `phi_amb` über die Zulufttrockenheit direkt in die
Trocknungskapazität eingeht — der Platzhalter unterschätzt die Streuung der wichtigsten
Störgröße.

**Der Tag/Nacht-Indikator bleibt draußen.** Er ist keine physikalische Eingangsgröße; er
wirkt ausschließlich über `T_amb` und `phi_amb`. Als Modellinput wäre er ein Kanal ohne
kausale Wirkung, den ein Netz als Abkürzung missbraucht.

**Zustandsgrößen, nicht Störgrößen:** `d_50`, `T_part`, `X_part`. Eine Störgröße wird dem
Modell als Eingang gegeben, eine Zustandsgröße muss es propagieren. Wer Zustandsgrößen
einspeist, baut ein Modell, das an der Anlage vollständige Zustandsmessung braucht.

| Zustandsgröße | Bereich Platzhalter | Bereich Labor (`LAB`) | Bemerkung |
|---|---|---|---|
| `d_50` | 250 … 900 µm (Startstreuung) | Cellets® 200, konstant | im Labor ohne Wachstum |
| `T_part` | 35 … 80 °C | 33.6 … 92.4 °C | Platzhalter zu eng |
| `X_part` | 0.003 … 0.020 kg/kg | **0.033 … 0.29 kg/kg** | **Platzhalter um Faktor 10 zu trocken** |

### Der Feuchtebereich ist der wichtigste Korrekturbedarf

Im Labor liegt die Partikelfeuchte zwischen 3.2 und 22.5 % bezogen auf die **feuchte**
Masse, also 0.033 bis 0.29 kg/kg trocken (`LAB`). Der Platzhalter im Modell reicht von
0.003 bis 0.020 kg/kg — **eine Größenordnung zu trocken.** Cellets® sind mikrokristalline
Cellulose und stark sorptiv; ein Granulat aus dem Anwendungspartner-Sortiment kann trockener sein.
Die Zahl ist deshalb nicht ungeprüft zu übernehmen, aber die Richtung ist eindeutig.

**Das ist keine kosmetische Korrektur, sondern verschiebt die Zeitkonstanten.** Nachgerechnet
am selben Betriebspunkt, mit `X_cr` und `X_eq` proportional angehoben (Status `MODELL`):

| Fall | `X_cr` / `X_eq` / `X_0` | τ schnell | τ zweite Mode | Steifigkeit |
|---|---|---|---|---|
| Platzhalter | 0.020 / 0.003 / 0.005 | **13.7 s** | **85 s** | 1171 |
| Laborniveau | 0.20 / 0.03 / 0.05 | 36.8 s | 371 s | 435 |
| Laborniveau, hoch | 0.29 / 0.033 / 0.08 | 33.3 s | 600 s | 480 |

Die schnellste Mode wird um Faktor 2.5 **langsamer** und die zweite um Faktor 4 bis 7. Der
Grund ist physikalisch und nicht numerisch: bei 0.5 % Feuchte arbeitet das Bett tief im
zweiten Trocknungsabschnitt (`nu` ≈ 0.06, die Trocknungskapazität der Luft übersteigt den
Wassereintrag um mehr als eine Größenordnung), bei 5 % nicht mehr. **Die in Fassung 2
berichtete Zeitkonstante von 9.5 s ist damit ein Artefakt eines zu trockenen
Platzhalters**, und mit ihr die Aussage „Δt = 60 s scheitert an der Feuchte-Mode".

Konsequenzen, die vor der nächsten Δt-Entscheidung zu ziehen sind:

- Δt = 60 s ist bei realistischem Feuchteniveau **nicht** mehr disqualifiziert: 10 Schritte
  auf τ ≈ 35 s sind vertretbar. Das halbiert die Sequenzlänge gegenüber Δt = 25 s.
- Die Steifigkeit fällt von 1171 auf ~450. Der Abstand zwischen schnellster und langsamster
  Mode bleibt drei Dekaden, aber die Spreizung innerhalb der schnellen Moden schrumpft.
- **Die Δt-Vorstudie ist mit korrigierter Sorptionsisotherme zu wiederholen.** Sie ist
  billig (drei Läufe je Δt), und ihr Ergebnis bestimmt die Sequenzlänge aller
  Architekturen.

### Zielgröße(n)

| Symbol | Größe | Einheit | Status | Anmerkung |
|---|---|---|---|---|
| `dE` | Energieverbrauch | kW **oder** kWh/kg | Definition `TODO` | primäre Zielgröße des Antrags |
| `X_prod` | Produktfeuchte | kg/kg | Zielgröße des Architekturvergleichs | im Modell = Bettfeuchte |
| `PSD` | Partikelgrößenverteilung | – | ab AP 2; Ziel 10 % | **mit dem Grey-Box-Modell nicht adressierbar** |
| `d_50` | mittlere Korngröße | µm | **aus dem Vergleich genommen** | frühe Messung ergab ~84 % Rauschanteil an der Streuung (**nicht reproduzierbar**: die Originalzahl stand in einer inzwischen umnummerierten Fassung von `04_stand_ap1.md`, Nummerierung 1–15 dort nicht mehr erhalten, siehe Befund 26 Teil 4, und ist nicht mehr eindeutig einer Stelle zuordenbar). Mit an dieser Sitzung reproduzierten Zahlen unabhängig bestätigt: FIR-R²_holdout durchgehend negativ (−0.042 bei k=1 bis −0.123 bei k=43, `artifacts/runs/probe_e2_e3_e4_lag_depth.json`), LSTM-Seed-Spannweite bei h=1 ≈0 (Kollaps auf Mittelwertsprädiktor, `artifacts/runs/probe_f3_f4_arch_holdout_inference.json`) -- siehe `experiments.py::DataSpec.target_names`-Docstring |

**Vorhersageaufgabe:** Zielgröße bei `t+1` aus Stell-, Stör- und Zielgrößen der Schritte
`0..t` — **falls** die Zielgröße gemessen wird. Andernfalls aus Stell- und Störgrößen
allein; das ist die Soft-Sensor-Aufgabe, siehe unten.

**Statischer Varianzanteil** — Bestimmtheitsmaß einer quadratischen Regression auf die
*aktuellen* Stell- und Störgrößen ohne Historie. Die Kennzahl ist **keine
Prozesseigenschaft**, sondern eine Eigenschaft von Prozess *und* Anregungsdesign, und
immer mit dem Design zu berichten (Status `MODELL`):

| Größe | R² mit Streuung (Standarddesign, Platzhalter-Config) | R² GAB, reproduziert (fp `82f65f576a2e5253`, Seeds 20260812/13/14, n_runs=32, Median) |
|---|---|---|
| `dE` (Momentanleistung) | 0.974 | 0.972 |
| Betttemperatur | 0.843 | 0.768 |
| Produktfeuchte | 0.274 | 0.508 |
| `d_50` | **0.100** | 0.144 |

Vormalige Spalte "R² ohne Streuung der Anfangszustände" entfernt -- mit `report_baselines.py`
in der bisherigen Fassung nicht erzeugbar (kein Schalter für
`randomise_initial_conditions`, siehe Skript-Docstring dort) und die Provenienz der
Originalzahlen nicht rekonstruierbar. Historische Werte, nicht als falsch markiert,
sondern als nicht reproduzierbar: `dE` 0.987, Betttemperatur 0.87, Produktfeuchte 0.43,
`d_50` 0.72.

**Die dynamischen Größen sind die Qualitätsgrößen, nicht der Energieverbrauch.** Die
Heizleistung ist algebraisch in `(T_in, u_gas, T_amb)`; der Prozesszustand wirkt nur über
Gasdichte und Bettdruckverlust. Folgen: der Architekturvergleich braucht mindestens eine
Qualitätsgröße, und die Pflicht-Baselines brauchen eine **statische Regression** — für
`dE` ist Persistenz die falsche Referenz.

---

## Beobachtbarkeit: zwei Betriebsarten, nicht eine

Neu in dieser Fassung, weil es die Aufgabe ändert und nicht nur ihre Umsetzung.

| | Inline gemessen | Soft Sensor |
|---|---|---|
| Historie enthält vergangene Zielwerte | ja | nein |
| Niveau des Laufs | steht in der Historie | Teil der Aufgabe |
| parameterfreie Baseline | Persistenz | Mittelwertvorhersage |
| Primärmetrik | NRMSE laufweiser Median | **NRMSE gepoolt** |

**Warum die Primärmetrik wechselt.** Der Befund aus Schritt 3 lautete: der gepoolte NRMSE
ist zu klein, weil die gepoolte Streuung die Niveauunterschiede zwischen den Läufen
enthält, „und die muss kein Modell vorhersagen, sie stehen in der Fensterhistorie". Der
zweite Halbsatz ist eine **Voraussetzung**: er gilt nur bei Inline-Messung. Ohne sie ist
das Niveau Teil der Aufgabe, und die laufweise Normierung wird zusätzlich instabil — bei
einem Lauf mit kleiner Eigenstreuung geht der Nenner gegen Null. Gemessen: eine
parameterfreie Mittelwertvorhersage erreicht laufweise Mediane über 3 mit
Seed-Spannweiten über 19, gepoolt dagegen sauber 1.01.

**Stand der Messbarkeit.** Im SPE-Labor wird die Partikelfeuchte inline über
**Mikrowellenresonanz** gemessen (TEWS MW4200, Sensor SPW-043-020-C), kalibriert gegen
thermogravimetrische Referenzproben (`LAB`). Für die Anwendungspartner-Anlage ist es offen — und
das ist Priorität 2 der Gesamtliste.

| Größe | In Dyssol | Labor SPE | Anwendungspartner-Anlage |
|---|---|---|---|
| Abluftfeuchte | ja | **ja** (`LAB`) | `TODO` |
| Ablufttemperatur | ja | **ja** (`LAB`) | `TODO` |
| Zulufttemperatur nach Erhitzer | ja | **ja** (`LAB`) | `TODO` |
| Partikeltemperatur im Bett | ja | **ja** (`LAB`) | `TODO` |
| Partikelfeuchte (inline) | ja | **ja, Mikrowellenresonanz** (`LAB`) | `TODO` **dringend** |
| Druckverlust Filter / Expansionsraum | ja | **ja** (`LAB`) | `TODO` |
| Volumenströme Fluidisierung und Zerstäubung | ja | **ja** (`LAB`) | `TODO` |
| PSD (inline) | ja | nein (offline CAMSIZER) | `TODO` |
| Stromverbrauch je Aggregat | ja | `TODO` | `TODO` |

Elf Inline-Kanäle im Labor (`LAB`) — das ist mehr, als das Grey-Box-Modell derzeit als
Eingang anbietet, und deutlich mehr als die vier Stell- plus drei Störgrößen der aktuellen
Konfiguration. Für AP 2 heißt das: der Merkmalsraum an der realen Anlage ist reicher, nicht
ärmer.

---

## Zeitliche Auflösung

| Parameter | Wert | Status |
|---|---|---|
| Zeitschrittweite Anlagendaten (SPE-Labor) | **2 s** | `LAB` |
| Zeitschrittweite Anlagendaten (Anwendungspartner) | `TODO` | |
| Zeitschrittweite Dyssol | `TODO` | von TUHH-A erfragen |
| Typische Chargendauer (SPE-Labor) | **1.5 … 3.9 h**, im Mittel 1.8 h | `LAB` |
| Simulationshorizont im ML-Modell | 8 h pro Lauf | `PLH` — **zu lang, siehe unten** |
| Rechenzeit Dyssol (Trocknermodell, 10 Chargen ≈ 18 h Prozesszeit) | **17 500 s**, also ~0.27 s Rechenzeit pro Sekunde Prozesszeit | `LAB` |
| Abtastzeit ML-Pipeline | 25 s | `MODELL`, zu überprüfen |

**Die angesetzte Laufdauer ist zu lang.** 8 h je Lauf gegen 1.8 h im Labor ist Faktor 4.4.
Das ist nicht nur eine Kostenfrage: die langsame Mode (Verweilzeit des Feststoffs,
~16 000 s ≈ 4.4 h) **passt gar nicht in eine Laborcharge**. In AP 2 ist Kornwachstum über
eine Charge damit ein angeschnittener Übergangsvorgang und kein Beharrungszustand — und
die Streuung der Anfangsbedingungen zwischen den Läufen, mit der die langsame Mode im
Anregungsdesign abgedeckt wird, ist dort nicht Kür, sondern die einzige Quelle dieser
Information.

**Die Dyssol-Rechenzeit begrenzt den Endpunkt der Lernkurven.** Aus 0.27 s Rechenzeit pro
Sekunde Prozesszeit folgt für 128 Läufe à 2 h Prozesszeit rund **69 h auf einem Kern** —
und das für ein Trocknermodell ohne Feststoffaufbau und ohne PSD. Das Granulationsmodell
wird langsamer sein. Zum Vergleich: das Grey-Box-Modell erzeugt 128 Läufe in etwa 40 s,
also rund Faktor 6000 schneller. Damit ist die Reihenfolge des Arbeitsplans bestätigt — die
Methodik gehört auf das Grey-Box-Modell, und Dyssol-Läufe sind ein knappes Gut, dessen
Sampling-Design **vor** der Produktion festzulegen ist.

**Es gibt keine „dominante Zeitkonstante"** (Status `MODELL`, Platzhalter-Feuchteniveau —
siehe die Korrektur oben):

| Mode | τ | physikalische Ursache |
|---|---|---|
| Partikelfeuchte + Betttemperatur (gekoppeltes Paar, siehe Anmerkung) | 13.7 s (Platzhalter) / ~35 s (Laborniveau) / 30.23 s & 359.93 s (GAB, Eigenwerte) | Trocknungskapazität der Luft und Wärmekapazität des Betts -- unter GAB nicht mehr sauber trennbar |
| Massenregelung | 600 s | Füllstandsregelung (Konfigurationswert, kein Prozess) |
| Partikelanzahl / Kornwachstum | 16 000 s | Verweilzeit des Feststoffs |

**Anmerkung GAB.** Unter der GAB-Sorptionsisotherme (Befund 34) existiert keine
wohldefinierte einzelne thermische Zeitkonstante des Betts: die geschlossene Form
τ_T = 84.75 s (`τ_T = C/(ṁ_L·c_pL + UA)`) liegt zwischen den beiden Eigenwerten
τ₁ = 30.23 s und τ₂ = 359.93 s und trifft keinen von beiden -- Feuchte- und
Temperaturdynamik sind unter GAB gekoppelt. Die Aufteilung "Partikelfeuchte" /
"Betttemperatur" als getrennte Zeilen ist deshalb nur für die Platzhalter-/Laborwerte
(konstante Gleichgewichtsfeuchte) gültig, nicht für GAB
(`artifacts/runs/probe_a2_a3_static_variance_time_constants.json`).

**Die Sequenzlänge ist die Entscheidung, welche Mode modelliert wird, keine
Hyperparameterfrage.** Die langsame Mode ist innerhalb eines Laufs nicht durch die
Stellgrößen anregbar (monotones Wachstum ohne Rückführung); sie wird über die Streuung der
Anfangsbedingungen abgedeckt.

**Zur Wahl von Δt** (Status `MODELL`, konstanter physikalischer Horizont von 600 s, Fenster
800 s, linearer ARX, laufweiser Median):

| Δt | Schritte pro Fenster | Persistenz h=1 (`dE`) | Rollout 600 s `dE` | Rollout 600 s `X_prod` |
|---|---|---|---|---|
| 5 s | 160 | 0.069 | 0.177 | 0.727 |
| **25 s** | **32** | 0.259 | **0.192** | **0.688** |
| 60 s | 13 | 0.362 | 0.162 | 1.404 |

Der **Ein-Schritt**-Fehler steigt monoton mit Δt, der **Rollout**-Fehler über denselben
physikalischen Horizont ist fast invariant. **Δt ist nach Rechenkosten und Sequenzlänge zu
wählen, nicht nach Genauigkeit.** Der Ausschluss von Δt = 60 s stützte sich auf die
Feuchte-Mode von 9.5 s und ist mit dem korrigierten Feuchteniveau neu zu prüfen.

Methodisch für alle künftigen Δt-Vergleiche: Fenster- und Horizontlänge in **Sekunden**
konstant halten und die Schrittzahlen umrechnen.

---

## Anlagengeometrie

| Parameter | Wert | Status |
|---|---|---|
| Anlage im SPE-Labor | ProCell® 5 LabSystem, Prozesskammer GF3 (Glatt) | `LAB` |
| Querschnittsfläche Wirbelschichtkammer | 0.0177 m² (150 mm) angenommen | `PLH` |
| Bettmasse (Feststoff, trocken) | 2.0 kg | `PLH`; Labor 1 oder 2 kg (`LAB`) |
| Wärmeverlust `UA` | 2.0 W/K | `PLH` |
| Betriebsart | kontinuierlich mit Füllstandsregelung | `PLH`; Labor **Charge** (`LAB`) |
| Düsentyp | Zweistoffdüse, **Top-Spray** | `LAB` (Kennlinie weiter `TODO`) |
| Bettmaterial im Labor | Cellets® 200, MCC, inert und wasserunlöslich | `LAB` |
| Maßstab Anwendungspartner-Produktionsanlage | `TODO` | |

Die angenommene Bettmasse trifft den Laborwert. Die **Betriebsart trifft nicht**: das Modell
führt eine kontinuierliche Füllstandsregelung mit τ = 600 s, das Labor fährt Chargen. Für
AP 2 ist die Füllstandsregelung abzuschalten oder als Chargenbetrieb zu parametrisieren —
sonst enthält das Modell eine Zeitkonstante, die es im Validierungsversuch nicht gibt.

---

## Physikalische Zusammenhänge für das Grey-Box-Modell

**Zustandsvektor** — vier Zustände, ideal durchmischtes Bett, Gasphase quasistationär:

    x = [m_s, m_w, N, T_bed]

`d_50` ist **kein** Zustand, sondern folgt algebraisch aus
`d_50 = (6 m_s / (pi rho_p N))^(1/3)`. Damit bleiben Massen- und Zahlenbilanz linear und im
Integrator exakt erhalten.

**Gasphase quasistationär:** Verweilzeit ≈ 0.2 s, drei Größenordnungen unter der
thermischen Zeitkonstante. Die Abluftfeuchte wird algebraisch bestimmt — das entfernt die
steifste Mode und erlaubt explizites RK4.

**Enthalpiebilanz** für die Betttemperatur, alle abfließenden Ströme auf `T_bed` referenziert:

    (m_s c_ps + m_w c_pw) dT_bed/dt
      = ṁ_L (c_pL + Y_in c_pD)(T_in - T_bed)
        + ṁ_spray c_psusp (T_susp - T_bed)
        - ṁ_evap Δh_v(T_bed)
        - UA (T_bed - T_amb)

**Kein zusätzliches Verzögerungsglied für das Bett.** Die thermische Zeitkonstante folgt
bereits aus der Bilanz zu `tau_T = C / (ṁ_L c_pL + UA)`. Ein Filter darauf zählt sie
doppelt. Das PT1 gehört ins **Sensormodell** (`PLH` 10 s).

**Zum Sensormodell, neu aus dem Labor:** die Rohdaten der Mikrowellenresonanz-Messung
enthielten je Charge 17 bis 34 % Ausreißer und wurden mit einem Savitzky-Golay-Filter
dritter Ordnung über 100 Abtastschritte geglättet (`LAB`) — bei Δt = 2 s also über 200 s.
Zwei Folgerungen, beide unangenehm:

1. **Das Platzhalter-Sensormodell ist zu gutmütig.** 0.5 % relatives Rauschen und ein PT1
   mit 10 s beschreiben nicht ein Signal, aus dem ein Viertel der Punkte als Ausreißer
   entfernt werden muss. Ein realistisches Modell braucht einen **Ausreißerprozess**
   (schwerschwänzig, nicht gaußisch) neben dem additiven Rauschen. Das ist ein
   Modellierungsauftrag, kein Parameterwert.
2. **Ein zentriertes Glättungsfenster über 200 s ist nicht kausal.** Es benutzt Messwerte
   aus der Zukunft. Für eine Offline-Analyse ist das legitim, für einen Online-Soft-Sensor
   oder einen Regler nicht — und in einem Datensatz mit zufällig gemischten Beobachtungen
   erzeugt es zusätzlich Informationsfluss zwischen Trainings- und Testmenge. Wenn geglättet
   werden muss, dann **kausal** (einseitig), und die Filterlänge gehört als Versuchsfaktor
   ins Protokoll, nicht in die Vorverarbeitung.

**Trocknungskinetik** über die normierte Trocknungskurve:

    ṁ_evap = nu_max * eta * ṁ_L * (Y_s(T_bed) - Y_in),    eta = clip((X_p - X_eq)/(X_cr - X_eq), 0, 1)

`nu_max` ≈ 0.6 (`PLH`, Literatur 0.4 … 0.8). **Der Kondensationszweig ist nicht optional:**
ist die Zuluft feuchter als die Sättigung bei Betttemperatur, kehrt sich der Stofftransport
um. Ohne ihn liefert das Modell bei kaltem Bett stillschweigend übersättigte Abluft — genau
der Betriebspunkt, in den ein untrainierter RL-Agent zuerst läuft.

**Layering-Wachstum** — im gewählten Zustandsraum eine Folgerung, kein Postulat:

    dd_50/dt = ṁ_spray w_solid d_50 / (3 m_s) = 2 V̇_fest / A_Partikel

**Beharrungsdurchmesser** `d_50,ss = d_nuc * b_nuc^(-1/3)`, mit `d_nuc` = 150 µm und
`b_nuc` = 0.008 (`PLH`) → 750 µm. Der Nukleationsanteil ist damit **kein freier
Fitparameter**.

**Harte Grenzen (als Output-Projektion umsetzbar)**
- Abluftfeuchte ≤ Sättigungsfeuchte — **strukturell** erfüllt, nicht per Projektion.
- Partikelfeuchte ≥ Gleichgewichtsfeuchte — strukturell über `eta → 0`, **aber** bei heißer
  Zuluft vom Integrator verletzbar; Projektion als Sicherheitsnetz nötig.
- `dE` ≥ 0, physikalische Wertebereiche aller Größen.

**Monotonie (architektonisch erzwingbar)**: ∂`dE`/∂`T_in` > 0, ∂`dE`/∂`u_gas` > 0. Im
Grey-Box-Modell strukturell erfüllt. **Vorbehalt:** gilt universell nur für die
*Momentanleistung*; beim spezifischen Verbrauch hängt sie daran, dass der
Produktmassenstrom nicht von `T_in` abhängt — was entfällt, sobald eine Qualitätsgrenze die
Sprührate begrenzt.

**Bemerkenswert:** oberhalb `u_mf` ist der Bettdruckverlust **unabhängig** von der
Gasgeschwindigkeit — er hängt nur an der Bettmasse. Die Abhängigkeit der Ventilatorleistung
von `u_gas` kommt aus Volumenstrom und Anströmboden, nicht aus dem Bett.

**Bewusst nicht im Modell:** PSD (nur zwei Momente), Agglomeration, Abrieb, Bruch,
Elutriation, Tropfengrößenverteilung, räumliche Gradienten, **Ausreißerprozess der
Sensorik**. Folge für `p_nozzle`: reine Kostenwirkung ohne Prozessnutzen — im RL fixieren
oder mit Qualitätsstrafe versehen.

---

## Anregungsdesign für die Datengenerierung

Der Antrag sieht für TUHH-A-AP 3 vor, Parameter „im Rahmen ihrer zu erwartenden realen
Schwankungsbreite" zu variieren. Das ist für Validierung geeignet, regt die Dynamik für
Lernaufgaben aber **zu schwach** an. Umgesetzt:

- **Sprungfolgen** mit zufälligen Amplituden und Verweilzeiten
- **PRBS** als echte m-Sequenz über ein Schieberegister — nicht als Bernoulli-Münze
- **Rampen** unterschiedlicher Steigung — die Schleppabweichung ist direkt proportional
  zur Zeitkonstante
- **Konstantläufe** als Referenz für Beharrungszustände, **gleichmäßig über die
  Design-Reihenfolge verteilt** (als Block am Anfang enthielt `subset(8)` 37 % statt 10 %)
- **Quasi-Monte-Carlo über den Störgrößenraum**, mit Tagesgang und OU-Drift
- **Streuung der Anfangszustände** über den Betriebsbereich

**Sobol statt Latin-Hypercube**, und **Stützstellen nur als Zweierpotenzen**
(8/16/32/64/128): die Balance-Eigenschaft einer Sobol-Folge gilt nur für `n = 2^k`, und
nur verschachtelte Designs trennen Datenmenge von Abdeckungsqualität.

**Ratenbegrenzung ist Pflichtbestandteil.** Solange die Heizerdynamik im Prozessmodell
fehlt, ist sie der einzige Ort, an dem Aktordynamik repräsentiert ist. Preis: das Spektrum
ist nicht mehr flach. Realisierte Bandbreite gegen die Eckfrequenz der schnellen Mode:
`T_in` 0.16, `u_gas` 1.00, `m_spray` 0.23, `p_nozzle` 0.53.

**Anfangszustände werden mitgestreut.** Startet jeder Lauf bei `d_50` = 400 µm und wächst
monoton, ist `d_50` innerhalb eines Laufs praktisch eine Funktion des Zeitindex — Modelle
mit Positionsinformation leiten die langsame Mode dann aus der Schrittzahl ab.

**Kollinearität:** Signalart und Amplitudenstufe werden pro Lauf **und pro Kanal**
unabhängig gezogen. Fahren alle Kanäle dieselbe Sprungfolge, sind ∂`dE`/∂`T_in` und
∂`dE`/∂`u_gas` nicht trennbar — genau die Ableitungen, die die Monotonie-Constraints
erzwingen sollen. Realisierte maximale Korrelation: 0.23.

**Zum Vergleich das Labor-Versuchsdesign** (`LAB`): 10 Chargen, fünf Parameter auf zwei bis
fünf Stufen, überwiegend Konstantfahrten, eine Charge mit Sprührate in fünf Stufen. Das ist
ein klassisches DoE zur Grenzensuche und für Lernkurven **zu schwach angeregt** — die
Stellgrößen nehmen über alle Chargen nur wenige verschiedene Werte an, was sie zu schwachen
Prädiktoren macht. Für AP 2 heißt das: das Versuchsdesign der Laborvalidierung ist
gegenüber dem Vorprojekt zu erweitern, nicht zu übernehmen. **Das ist die konkrete,
billige Maßnahme, die sich aus dem Vergleich der beiden Designs ergibt.**

---

## Offene Fragen an TUHH-A

1. **Variablenliste als Interface-Vertrag**: exakte Namen, Einheiten, Klassifikation
   (Stell-/Stör-/Zustands-/Ausgangsgröße). Die Trennung Stör- vs. Zustandsgröße ist
   inhaltlich wichtig.
2. **Zeitschrittweite und typischer Simulationshorizont.** Referenzpunkt aus dem
   Vorprojekt: 2 s Anlagendaten, 1.8 h Chargendauer.
3. **Realistische Stellgrößenbereiche inkl. Fluidisierungsgrenzen** und maximale
   Änderungsraten.
4. **Rechenzeit pro Simulationslauf.** Referenzpunkt aus dem Vorprojekt: 0.27 s Rechenzeit
   pro Sekunde Prozesszeit für das Trocknermodell. Gilt das für die Granulation auch nur
   näherungsweise, ist ein 128-Lauf-Datensatz eine Angelegenheit von Tagen, nicht Stunden.
5. Welche Prozesseinheiten statisch, welche transient.
6. **Sorptionsisotherme im Materialsystem von Dyssol.** Im Vorprojekt war die falsche
   Isotherme die Hauptursache der Abweichung. Welche ist im neuen Modell hinterlegt, und
   für welches Material?
7. Erstes (ruhig unvalidiertes) Fließschema `.dflw` des Einzelgranulators.
8. **Werden Zeitkonstanten in Dyssol so aufgelöst, dass ein Vergleich mit den
   Grey-Box-Werten möglich ist?** Das ist das quantitative Kriterium für die Kalibrierung —
   stimmen die Zeitkonstanten, stimmt die für Sequenzmodelle relevante Dynamik. Zu
   vergleichen sind die Werte beim **realistischen** Feuchteniveau, nicht die der
   Platzhalter.
