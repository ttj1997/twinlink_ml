# Prüfung 3: Code der tragenden Module

**Stand:** 2026-10-09, Repo-Stand `c7c9621` (master, sauberer Arbeitsbaum)
**Umfang:**
- Physik: `src/twinlink_ml/simulation/` (Bilanzen, Einheiten, Feuchtebezug relativ/absolut, Normkubikmeter, Bezugszustände der Enthalpien)
- Datentrennung: `data/` (Splits, Fenster), Normierung nur aus Trainingsdaten, Fenster über Laufgrenzen
- Metriken und Auflösungsschwelle: `evaluation/` (Schätzer, t gegen z, Freiheitsgrade)
- Zielfunktion und CO2-Verbuchung: `evaluation/net_energy.py`, Standort- und Wärmenetz-Module, Wirkungsgrad und Heizwert
- Kopplungsgewinn: `evaluation/coupling_gain.py` und die Suche (`evaluation/joint_search.py`)

**Vorgehen:**
- Nur gelesen, nichts im Repo geändert, nichts committet.
- Volle Testsuite gelaufen (`.venv`, `pytest -p no:cacheprovider`): **673 bestanden, 2 übersprungen, grün.** Mehrere Befunde unten werden von keinem Test erfasst.
- Kein Skript aus `scripts/` mit Schreibzugriff ausgeführt; Nachrechnungen mit eigenen Skripten im Session-Scratchpad (außerhalb des Repos, nicht dauerhaft).
- Fünf Teilprüfungen parallel (Physik ×2, Datentrennung, Metriken, Zielfunktion/CO2); Kopplungsgewinn und Suche direkt geprüft.
- **[v]** = direkt am Code nachgeprüft oder mit Repo-Code nachgerechnet. Ohne Markierung: Rechnung der Teilprüfung (TP), nicht direkt nachgeprüft.
- Befunde, die bestehende Ergebnisse berühren (A1, A2, B1, C1/C2, E1), sind nach CLAUDE.md **nur gemeldet, nicht korrigiert.**

---

## A. Zielfunktion, Wärmenetz, CO2

**A1 (schwer) [v]: Die Kaskade prüft die Mindestgrädigkeit nur am warmen Ende einer Stufe.**
- **Wo:** `heatnetwork/balance.py:545-547`. Begrenzt wird nur `T_Senke,aus ≤ T_Quelle,ein − ΔT_min`. Ob am kalten Ende `T_Quelle,aus ≥ T_Senke,ein + ΔT_min` gilt, wird nie geprüft.
- **Warum falsch:** Im Gegenstrom liegt die engste Stelle am kalten Ende, sobald die Senke den größeren Wärmekapazitätsstrom hat. Der Docstring (`:92-101`) nennt die Grädigkeit eine harte Nebenbedingung jeder Stufe. Der Test `tests/test_heatnetwork_balance.py:108` prüft nur das warme Ende.
- **Rechnung mit Repo-Code** (M4-C, Szenario 5, M3'-Optimum aus `m4c_confirm_lab_umin126.json`): Stufe LA→S1, Q = 420 kW, Quelle 64,6 → 29,1 °C, Senke 23,9 → 44,6 °C. Am warmen Ende 20,0 K, **am kalten Ende 5,2 K** (gefordert 20 K). In S2 liegen dieselben Stufen bei 14–17 K.
- **Bedeutung:** Genau über diese Stufe läuft der Gewinnmechanismus aus Befund 76/77 g (Abluft heizt S1 vor, dadurch wird Q1 für SA/SB frei).
- **Nur Rechnung der TP, nicht direkt nachgeprüft:** Mit am kalten Ende korrigierten Stufen und festgehaltenen M3'-Sollwerten wird G negativ (Sz. 0: +7,0 → −16,8; Sz. 5: +11,4 → −21,2 kg CO2e/h). Das ist **kein** korrigierter Kopplungsgewinn, denn dafür müsste neu optimiert werden. Es zeigt aber, dass G aus Befund 73/76/77 an dieser fehlenden Prüfung hängen kann.
- **Frage:** Ist die Grädigkeit als Minimum über den ganzen Tauscher gemeint, wie in der Wärmeübertragung üblich?

**A2 (schwer) [v]: Die Netto-Gutschrift in `evaluation/net_energy.py:72-122` hat weder eine Obergrenze noch eine Temperaturprüfung.**
- **Wo:** Die Funktion nimmt `Q_rec` in jedem Modus an. Das Skript `study_operating_point_regret_net.py` (Befund 57c, MA4) rechnet `Q_rec` ohne ΔT_min und ohne Begrenzung durch die Senke.
- **Rechnung mit Repo-Code** (Nennsollwerte, 35 °C / φ 0,85, kondensierend):
  - Q_rec = 3,60 kW bei einer Lufterhitzerleistung von 1,65 kW, also **218 %**. `net_kw` wird −2,17 kW, `net_co2` wird −0,134 kg/h.
  - Ginge die Wärme in die eigene Zuluft, käme diese auf 154,7 °C, gespeist von 63,6 °C warmer Abluft.
  - Bei 35 °C / 0,55 sind es 171 %.
- **Frage:** Welche Senke soll die Gutschrift im Einzellinienfall ersetzen? Ist es die eigene Zuluft, ist das physikalisch unmöglich. Ist es eine Senke am Standort, fehlt die Temperaturprüfung.
- **Folge:** Der „Nenner nahe null“, den Befund 57c als Grund für die instabile Regret-Kennzahl in V5 nennt, wäre dann diese Gutschrift und keine bloße Numerik.
- `carrier_energy.py:84` begrenzt korrekt mit `max(duty − Q_rec, 0)`.

**A3 (mittel, dokumentiert): `net_energy.py:100/122` mischt die Energieträger.** Der Bruttowert ist elektrisch (Elektroerhitzer, η 0,98), die Gutschrift wird aber als Gas gebucht (`Q_rec/0,90·0,201`). `carrier_energy.py` (D5) löst das ab, Befund 57 beruht aber noch darauf.

**A4 (gering):**
- `recoverable_power.py:95-96` begrenzt nicht bei 0. Mit T_out 38 °C und Y = 0,04 ergibt sich −13,3 kW. Das stört nur, wenn der Wert als Gutschrift dient (MA4).
- `site_network.py:274` baut die Konfiguration mit dem Default `GasConfig()` statt mit `config.gas`. Die Werte sind heute identisch, das ist ein latentes Abdrift-Risiko.
- `dryer_block_comparison.py:306` zieht `outlet_duct_heat_loss_k` nicht ab (heute 0,0), inkonsistent zu `site_network.py:189`.

**A5 (Frage): Jahrgang des Strom-Emissionsfaktors.** Der Strom-Faktor 0,366 kg/kWh entspricht dem UBA-Strommix 2020 und erfasst nur direktes CO2, ist aber mit „CO2e“ beschriftet. Gas 0,201 ist auf den Heizwert H_u bezogen; beide sind untereinander konsistent. Das Verhältnis 1,82 entscheidet über das Vorzeichen jedes Tauschs „Strom gegen Gas“, daher zählt der Jahrgang.

**Geprüft und korrekt:**
- Wirkungsgrade in der richtigen Richtung (`duty/η`, η genau einmal).
- H_u kürzt sich heraus (45 vs. 50 MJ/kg ändert nur den Brennstoffmassenstrom).
- kW·kg/kWh = kg/h; keine Doppelzählung im Standortmodell.
- Taupunkt aus der absoluten Feuchte.
- Pinch-Problemtabelle: ±ΔT_min/2 genau einmal verschoben.
- Die Sentinels 1e6 und das Strafgewicht erreichen keine berichteten Werte.

---

## B. Kopplungsgewinn und Suche (direkt geprüft)

**B1 (mittel) [v]: `G_set` kann negativ werden, dadurch ist `G_alloc` überhöht.**
- **Wo:** `joint_search.py:319-320, 357` und `scripts/run_m3_joint.py:283-291`.
- **Was passiert:** Die S2-Allokation wird von ihrem *Tabellenoptimum* aus poliert, nicht von den S2-Sollwerten aus.
- **Warum falsch:** Die S2-Sollwerte sind selbst ein Kandidat für „beste Sollwerte für die S2-Allokation“. Also muss `G_set ≥ 0` gelten.
- **Artefakt `m3_joint_control_umin126.json`:** `G_set < 0` in 10 von 25 Szenarien. Szenario 17: G = −0,166, G_set = −1,530, daraus **G_alloc = +1,364**.
- Mit `G_set = max(G_set, 0)` wäre G_alloc in Szenario 17 gleich −0,166 und damit nicht „material“. Befund 73 d führt Szenario 17 aber mit 1,4 kg/h als materiellen Allokationsgewinn.
- **c0.5:** Szenario 17 (1,285) und Szenario 23 (0,389) kippen genauso.
- Der Nachtrag zu Befund 73 („G sind untere Schranken“) deckt G ab, aber nicht die Aufteilung.
- Am Urteil (nicht nachgewiesen) ändert das nichts.

**B2 (gering) [v]: Die Tabellenstrafe weicht von der echten Strafe ab.**
- **Wo:** `line_tables.py:99-103`.
- **Was passiert:** Der Gültigkeitsterm berücksichtigt nur die Feuchte über der kritischen Feuchte. `solution.feasible` (`steady_state.py:371-397`) wird zusätzlich bei u ≤ u_mf, u ≥ u_t, Sättigungsgrad > 1 und Feuchte unter dem Gleichgewicht falsch.
- **Folge:** Außerhalb des Wirbelfensters ist die echte Strafe um 1 höher als in der Tabelle. Das betrifft nur den Tabellenfehler E und die Fenstererweiterung, nicht die polierten Werte (die Strafen sind in allen Artefaktzeilen 0).

**Fragen:**
- **Duplikatregel:** Gleicher Tabellenwert und gleiches Tabellenoptimum heißt nicht zwingend gleiche echte Zielfunktion, zum Beispiel wenn eine Verbindung nur am Tabellenoptimum keine Wärme führt.
- **Fehlschläge am Tabellenoptimum:** Eine Allokation, deren echte Lösung dort scheitert, fällt ohne Ersatzstart heraus. In den Artefakten war das 0-mal der Fall.
- **Extrapolation:** `LineTable.interpolate` extrapoliert linear außerhalb des Gitters (Gewichte nicht begrenzt) (TP).

**Korrekt:**
- Die Kriterien i–iii in `coupling_gain_verdict`, NaN-Behandlung, Betrag der gemischten Differenz im Probe-Skript.
- Die Abbruchlogik der Fenstererweiterung.
- Die S2-Allokation wird als Tupel gefunden (`s2_allocation_index` ist nie `None`).
- Die Strafe hängt nicht von der Allokation ab, daher ist `min` über CO2 zulässig.
- Die Auswertung bildet G aus CO2 und nur über zulässige Szenarien.

---

## C. Metriken und Auflösungsschwelle

**C1 (schwer) [v am Code]: Die Schwelle in `resolution.py:373-410, 556-594` hat die falsche Varianz, den falschen Maßstab und eine ungepaarte Formel.**
- **Formel:** Die Schwelle ist z·σ̂·√(2/n), mit σ̂ allein aus der Streuung der Persistenz (bzw. Klimatologie).
- **Falsche Varianz:** Die Streuung von Kandidat und Referenz geht nie ein.
- **Falscher Maßstab:** Die Schwelle ist relativ zum Median der Persistenz, die Verbesserung relativ zum Median der Referenz. Bei `energy_target` ist NRMSE ≈ 0,93 gegen ≈ 0,07.
- **Ungepaart:** Auf der Achse JOINT ist das Design gepaart.
- **Simulation der TP, n = 3, ohne echten Unterschied:** Falsch-positiv-Rate 0 % bei Modellstreuung 0, 22 % bei Modell-σ = Daten-σ, 45 % bei 2·σ.
- **Neu gerechnet auf `arch_composite.jsonl` (TP):**
  - SOFT_SENSOR `energy_target`: „LSTM vs. GRU nachweisbar +1,6 %“ hat gepaart das falsche Vorzeichen. LSTM ist in 4 von 5 Replikaten schlechter, p = 0,076.
  - Umgekehrt gibt es gepaart klare Fälle (5/5, p ≈ 0,01), die die Bibliothek „nicht auflösbar“ nennt.
- **Bisher nur in Skripten korrigiert:** Die Befunde 63g und 67 haben auf die t-Form umgestellt, aber nur in den Skripten. Ältere Urteile im Laborbuch (z. B. Befund 16, Tabellen um Zeile 1224) beruhen auf der Bibliotheksformel.
- Dieselbe Übertragung in `plots.py:150-154` und `scripts/study_validation_policy.py:107`.

**C2 (mittel-schwer) [v]: z statt t (`resolution.py:98`, `:194`).**
- Bei n = 3: 1,96 statt t(4) = 2,776, die Schwelle ist etwa 1,4-fach zu klein.
- Bei n = 5: t(8) = 2,306, etwa 1,2-fach zu klein.
- Mit t wäre Δmin bei n = 3 etwa 1,34 × Spannweite. Damit stimmt die Docstring-Aussage „kleiner als die Spannweite“ nicht mehr.

**C3 (mittel):**
- Verglichen werden Mediane, die Formel nutzt aber den Standardfehler des Mittelwerts. var(Median)/var(Mittel) ≈ 1,35 (n = 3).
- `minimum_detectable_difference` und `required_replicates` liefern nur 50 % Power (simuliert 0,52). Die „41 Seeds“ aus Befund 63h sind der Wert für 50 % Power, bei 80 % Power etwa doppelt so viele.

**C4 (mittel) [v]: Schwelle 0 bei n = 1 oder Spannweite 0** (`resolution.py:165-166`). Dann ist jeder Unterschied „nachweisbar“. Das kann über die Persistenz auf dem Konstantlauf-Holdout passieren, die bitgleich über die Seeds ist (Befund 63b).

**Gering:**
- Keine Korrektur für Mehrfachvergleiche.
- `summarise_policy_gap` (`protocol.py:630-669`) bildet die Differenz zweier Mediane statt gepaart.
- `summarise_curve` (`experiments.py:542-567`) gruppiert nur nach (label, n_train); bei gemischten Logs würde es Varianzquellen poolen (latent, heutige Aufrufer homogen).
- Läufe mit Referenzstreuung 0 werden still per `nanmedian` verworfen (`evaluator.py:105, 162-164`).

**Frage:** Die Klimatologie ist nicht parameterfrei (Trainingsmittel). Ihre Schwelle fällt mit n_train (5,4 → 0,3 %). Das widerspricht `resolution.py:385-388`.

**Korrekt:**
- Die Hartley-d2-Werte, `ddof=1`.
- Die NRMSE-Definition: Mittelwertprädiktor ergibt genau 1, affin invariant.
- Die Replikateinheit ist der Seed, nicht das Fenster.
- Median/Min/Max über Seeds.
- Welch-Satterthwaite in den SX1-Skripten (ν_eff 3,75).

---

## D. Datentrennung

Kernaussage: **Die Regeln halten.**
- Splits sind laufweise.
- Die Skalierer kommen nur aus Train, je Stützstelle neu.
- Kein Fenster mischt Läufe. Mit Probedaten, die Lauf und Zeit kodieren, geprüft: 85 Fenster, keine Vermischung, das erste Zukunftsziel ist t+1.
- Validation dient nur dem Early Stopping, berichtet wird auf Test.

**D1 (mittel, Frage) [v am Code]: Projizierte Zukunftsaktionen tragen Information über den wahren Zustand.**
- **Wo:** `base.py:369-394` speichert die Aktion nach `project_action`. Das Clipping der Gasgeschwindigkeit hängt an u_mf und u_t aus dem wahren d50. Über `windows.py:385` landet sie in `future_exogenous`.
- **Probe der TP:** 30,6 % der Testfenster enthalten eine projizierte Zukunftsaktion.
- **Wer betroffen ist:** Nur Läufe mit `enforce_gas_velocity_projection=True`, das sind die Arch-Lernkurven-Worker (`run_arch_learning_curve_worker.py:239`, `run_arch_constant_trained_learning_curve_worker.py:157`).
- **Nebenwirkung:** Der „Konstantlauf“ ist dann in u nicht mehr konstant.
- **Frage:** Kennt der Anlagenregler die künftigen Clip-Grenzen? (`Trajectory.requested_actions` existiert bereits als Alternative.)

**D2 (gering, latent) [v]:** `protocol.py:462-481`, `estimate_noise_floor(which=VAL)` vergleicht Test (verrauscht) mit Val (sauber). Die Fensterzahlprüfung greift bei 32/32 nicht. Ergebnis: Rauschgrenze 1,99 statt 0,021. Kein Aufrufer nutzt `VAL`.

**D3 (gering, latent) [v]:** `point_split.py:207-208`. Input- und Target-Skalierer werden auf verschiedenen Fenstermengen gefittet. Das bricht Persistenz, Residual-Kopf und NARX-Anker (Bias 30 Einheiten in der Probe). Heute nutzt nur ExtraTrees den Point-Split, und Bäume sind dagegen invariant.

**Weitere Punkte:**
- **Schwache Tests:** `test_pipeline.py:299-304` prüft nur `start+span ≤ n_steps` und ist damit konstruktionsbedingt immer wahr. `test_pipeline.py:307-313` vergleicht nie mit einem reinen Train-Refit.
- **Benennung des Testsets:** Das „angeregte“ Testset enthält 3 von 32 Konstantläufen (etwa 9 %). `experiments._record` loggt keinen Testset-Namen, das tun nur die Skripte.
- **Dokumentation:** `evaluator.py:328-332` behauptet, `physical_units` decke Fehler in der Skalierer-Weitergabe auf; NRMSE ist affin invariant, das kann es nicht.
- **Annahme, dokumentiert:** Zukünftige Störgrößen (T_amb, RH, Feststoffanteil) gelten über den Horizont als bekannt (`windows.py:14-21`).

---

## E. Physik

Kernaussage: **Kein Vorzeichen- oder Einheitenfehler in den vier ODE-Rechten-Seiten.**
- Wasser- und Feststoffbilanz schließen stationär auf etwa 1e-14.
- Relative und absolute Feuchte sind sauber getrennt (Y_in = 7,26 g/kg bei 20 °C / 50 %, von Hand nachgerechnet).
- Nm³ kommt im Simulator nicht vor.

**E1 (mittel, von zwei TP unabhängig gefunden) [v am Code]: Die Abluftfeuchte schließt die Gasseite nicht.**
- **Wo:** `thermo.py:333-359`. Zurückgegeben wird die gedämpfte Iterierte, die Verdampfung aber ungedämpft.
- **Rechnung:** Bei `ConstantIsotherm` ist ṁ_L(Y_out − Y_in)/E = 1 − 0,3⁵ = **0,99757** (gemessen). Bei GAB 0,9988–0,9998.
- **Widerspricht:** `thermo.py:301` („exakt“) und `exhaust.py:21-24`.
- **Betroffen ist alles, was `outlet_humidity` liest:**
  - Latente Abluftwärme (0,7125 statt 0,7142 kW).
  - Taupunkt (−0,02 K).
  - `heatnetwork`, `recoverable_power`, `site_network`.
- **Kleine Zahlen, aber systematisch:** Der Test (`tests/test_exhaust.py:59`) vergleicht zwei Pfade mit demselben Bias.

**E2 (mittel) [v]: `diagnostics.mass_balance_residual` (`:321-338`) rechnet falsch.**
- **Fehler:**
  - `states[:,0]` ist der Zustand *nach* Schritt 1.
  - Der Saatstrom fehlt.
  - Es wird mit Rechtecken integriert.
- **Messung der TP:** 2,5e-3 statt wahr 5,5e-13. Mit Saat 0,2 kg/h: 0,31.
- **Folge:** Die versprochene 1e-6-Schranke ist so nicht einlösbar.
- **Test ohne Aussage:** Der Test (`tests/test_greybox_balances.py:63-74`) startet am Sollwert, der Rest ist exakt 0, er prüft also nichts. Die Funktion wird nur dort benutzt.

**E3 (gering-mittel, Frage) [v am Code]: Störgröße w_solid im Tagesgang.**
- **Wo:** `excitation.py:561-567`. Jede Störgröße ohne „humidity“ im Namen bekommt den Tagesgang in Phase mit T_amb, auch `suspension_solid_fraction`.
- **Messung:** Korrelation T_amb ↔ w_solid innerhalb eines Laufs im Median 0,71, maximal 0,94 (`broadband_derived`, 64 Läufe).
- **Folge:** Das baut eine Verwechslung von Heizleistung und Wasserlast in die Daten ein. Der Docstring erwähnt nur T_amb und RH.
- **Frage:** Ist das gewollt?

**E4 (gering):**
- `cp_suspension = 3500` ist fest, während w zwischen 0,2 und 0,4 variiert. Gemischt wären es 3584 bis 2988. Das gibt etwa +5 W Bilanzrest bei 831 W, abhängig vom °C-Bezug.
- Die Steigung der Verdampfungsenthalpie, 2358,6 statt c_w − c_v = 2320, gibt −0,7 W.
- `test_enthalpy_balance_closes_at_steady_state` bildet die eigene Modellformel nach und kann das nicht finden.
- Saatgut ohne Enthalpieterm (implizit bei T_Bett); Saatstrom ist heute 0.
- `exhaust_sensible_kw` (`steady_state.py:348-353`) nutzt T_Bett statt T_Bett − Kanalverlust (heute 0,0).

**Fragen:**
1. **Düsendruck absolut oder Überdruck?** (`energy.py:116`) Bei 1,0 bar wird das Druckverhältnis 0,987, es wird abgeschnitten, und die Zerstäuberleistung ist 0 W. Als Überdruck wären es 17 W (2,5 bar: 78,5 statt 57 W). Die Optimierer landen genau auf diesem Knick bei 1,0 bar (Laborbuch etwa Zeile 9511).
2. **Zerstäuberluft** (etwa 1,2–1,3 % der Wirbelluft, etwa 11–16 W Kühlung) fehlt in allen Bilanzen.
3. **Bezugszustand von u_gas:** Der Massenstrom ist ρ(T_Bett)·u·A. Ist der Anlagensollwert volumetrisch bei Zulufttemperatur oder in Nm³/h, sind es etwa 8 % Unterschied.
4. **„d50“** ist d30, kein Massenmedian.
5. **Relatives Sensorrauschen auf einem °C-Wert** hängt von der Temperaturskala ab.
6. **Gültigkeit des nassen Betts:** Bei 60 °C / 0,8 m/s / 3,0 kg/h / 35 °C / 0,85 endet ein Lauf bei X = 1,69 kg/kg, ohne Grenze für Bettnässe oder Entfluidisierung.
7. `Trajectory.actions` wird vor `_clip_action` gespeichert. Ohne Wirkung, solange die Anregung im Kasten bleibt.
8. Sättigungsgrad-Diagnose: Die Prüfung „Sättigungsgrad ≤ 1“ (`steady_state.py:379`) kann wegen `max_saturation_degree ≤ 1` nie anschlagen.
9. `0.622` ist in `thermo` fest kodiert, `r_specific_dry_air` dagegen konfigurierbar.
10. `compute_regret` beschriftet die Ausgabe `absolute_kw` (`oracle_search.py:193`), auch bei `target = specific_kwh_per_kg`.

**Korrekt (Auswahl):** Buck-Sättigungsdruck (≤ 0,04 % gegen IAPWS bis 90 °C), Y = 0,622·p_v/(p − p_v) und Umkehrung, Taupunkt, Dichte feuchter Luft und Aufteilung trocken/feucht, Sutherland, Wen–Yu, Haider–Levenspiel, Kondensationszweig, Produktfeuchte auf Trockenbasis, Anzahlbilanz, kg/h → kg/s, kWh/kg = kW/(3600·kg/s), RK4-Stabilität (λ·dt ≈ −0,36), Boden-Clipping greift in den Standardläufen nie.

---

## Wie weiter

Vorschlag zur Reihenfolge:
1. **A1 (Grädigkeit am kalten Ende):** Kann den Kopplungsgewinn aus Befund 73/76/77 tragen oder kippen. Vor jeder Korrektur ist dafür eine Neuoptimierung nötig.
2. **C1/C2 (Auflösungsschwelle):** Hier hängen ältere Rangaussagen im Laborbuch dran.
3. **A2 (Netto-Gutschrift, Befund 57c).**

Bei A1, A5, D1, E3 und den Physik-Fragen 1 und 3 ist zuerst eine Festlegung nötig, wie es gemeint ist. Danach Korrekturen getrennt auf Branches.
