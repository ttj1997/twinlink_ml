# Arbeitsplan ML-Start (TUHH-B-AP 1)

> Fassung 5. Reihenfolge bewusst gewählt: erst der durchgehende Strang von Daten bis
> Metrik, dann die interessanten Architekturen. Nicht umgekehrt.
>
> Stand: 13.08.2026 · Schritte 1–5 erledigt, Schritt 6 zur Hälfte. Umsetzungsstand und
> Messwerte: `04_stand_ap1.md`. Prozessannahmen: `02_prozess_setup.md` (Fassung 3, der
> frühere Nachtrag ist eingearbeitet).

## Leitidee

Der Wert von AP 1 liegt nicht darin, xLSTM zum Laufen zu bringen, sondern in einem
**belastbaren Vergleichsrahmen**. Jede Kennzahl, die entscheidet, ob ein Vergleich
aussagekräftig ist, wird berechnet und mitprotokolliert, nicht angenommen. Bisher haben die
Schritte widerlegt, was vorher plausibel aussah:

| Schritt | widerlegte Annahme |
|---|---|
| 1 | „es gibt eine dominante Zeitkonstante" |
| 1 | „`dE` ist eine dynamische Zielgröße" (R² statisch 0.97) |
| 2 | „ein PRBS regt alle Kanäle breitbandig an" (Ratenbegrenzung dominiert) |
| 3 | „gepoolter NRMSE ist eine neutrale Metrik" (Faktor 31 bei der Feuchte) |
| 4 | „mehr Trainingsläufe helfen" (linearer ARX sättigt bei 16) |
| 4 | „Δt fein ist besser" (Rollout-Fehler bei konstantem Horizont fast invariant) |
| 5 | „drei Seeds reichen für einen Architekturvergleich" (Auflösung 10–15 %) |
| 5 | „`d50` ist eine Zielgröße" (zu 84 % Rauschgrenze) |
| 5 | „die Feuchte scheitert am Rauschen" (auch rauschfrei schlägt Persistenz den ARX) |
| 6 | „die Feuchte braucht Nichtlinearität" (das NARX ist schlechter als der lineare ARX) |
| 6 | „der laufweise NRMSE ist *die* Primärmetrik" (gilt nur bei Inline-Messung) |
| 6 | „mehr Historie hilft" (Sprung zwischen 1 und 2 Schritten, danach flach) |
| 6 | „der ARX sättigt, also sättigt die Aufgabe" (die NARX-Kurve läuft bei n=64 weiter) |

Die letzten vier verschieben den Fokus erneut: **die offene Aufgabe ist die
Fehlerakkumulation im Rollout, nicht die Funktionsklasse und nicht die Fensterlänge.**

---

## Schritte 1 bis 5 — **erledigt**

**1 Grey-Box-Prozessmodell.** Vier Zustände, ideal durchmischtes Bett, Gasphase
quasistationär.

**2 Anregungsdesign.** Sprungfolgen, PRBS als echte m-Sequenz, Rampen, Konstantläufe
(gleichmäßig verteilt). Sobol statt Latin-Hypercube, Stützstellen als Zweierpotenzen,
Ratenbegrenzung als Pflichtbestandteil, gestreute Anfangszustände. Regel: Maximaldesign
genau einmal bauen, alles Weitere über `subset()`.

**3 Vertikaler Durchstich.** Datengenerierung → Fensterung → Splits → Modell → Evaluation →
Ergebnis-Log. Dreiblock-Fenster, `RunSplitter`-ABC, rollout-zentriertes `SequenceModel`,
`ExperimentLogger` mit Fingerprints. Δt = 25 s als Standard, nach Rechenkosten gewählt.

**4 Baselines.** Persistenz, statische quadratische Regression, linearer ARX(4,2) mit
`j = 0`-Term. Ergebnis: `energy_target` ist bei 0.172 zu schlagen, `product_moisture` bei
0.794 — und letzteres schafft der lineare ARX nicht.

**5 Evaluationsprotokoll.** Varianzzerlegung nach Quelle (`SeedPlan`), Auflösungsvermögen
(13.2 % / 9.7 % / 14.5 % mit drei Seeds), modellfreie Rauschgrenze (`NoiseFloor`), beide
Validierungs-Betriebsarten, Lernkurven-Notebook. Nebenertrag: das Sensormodell ist vom
Integrator entkoppelt (`remeasure`), Rauschvarianten kosten 0.005 s statt 0.78 s.

---

## Schritt 6 – Architektur-Slots (Teil 1 erledigt, Teil 2 offen)

### Teil 1 — der Rahmen, gemessen

- **Beobachtbarkeit als Versuchsfaktor.** `ObservationMode.INLINE` und `SOFT_SENSOR`;
  Maskierung in der Fensterung, nicht im Modell. Parameterfreie Referenz ist Persistenz
  bzw. Mittelwertvorhersage, Primärmetrik laufweiser Median bzw. gepoolt.
- **Architektur-Registrierung** (`models/registry.py`) mit Verfügbarkeitsprüfung; nicht
  implementierte Slots sind deklariert und melden, was fehlt.
- **Verdikte** (`architecture_verdicts`): eine Differenz erscheint nur zusammen mit der
  Auflösungsschwelle und als fertig formulierter Satz.
- **`RandomFeatureNARX`** — nichtlinear, ohne PyTorch, mit echtem Modell-Seed. Ändert
  gegenüber dem linearen ARX **genau eine** Sache: die Basisfunktion. Damit ist bei einem
  Gewinn des GRU trennbar, ob er aus Nichtlinearität, gelerntem Zustand oder dem
  Rollout-Training kommt.
- **`torch_sequence.py`** mit austauschbarem Kern (GRU / LSTM) hinter *einem* Trainer.

### Teil 2 — was noch fehlt, und in welcher Reihenfolge

**Vor dem ersten neuronalen Lauf, ohne Ausnahme:**

1. **Sechs übersprungene Torch-Tests grün bekommen.** Darunter die Zusicherung
   „nullinitialisierter Kopf = Persistenz, exakt" und „GRU und LSTM teilen den Trainer".
2. **Lernkurven bis 128 und 256** für ARX und NARX. Die NARX-Kurve läuft bei n = 64 noch;
   ohne diese Punkte wird eine Sättigung berichtet, die nur die lineare Baseline betrifft.
3. **Δt-Vorstudie wiederholen** mit korrigiertem Feuchteniveau (τ_schnell ≈ 35 s statt
   13.7 s). Δt = 60 s ist damit nicht mehr ausgeschlossen, und die Sequenzlänge aller
   Architekturen hängt davon ab.
4. **Seed-Budget und Relevanzgrenze festlegen.** Drei Seeds lösen 10–15 % auf; für 5 % sind
   12–26 nötig. Die Relevanzgrenze ist getrennt zu setzen — im Soft-Sensor-Betrieb sind
   5 % über der Mittelwertvorhersage statistisch nachweisbar und fachlich bedeutungslos.

**Dann die Architekturen**, in dieser Reihenfolge und mit diesem Zweck:

| Slot | Was sie beantworten soll |
|---|---|
| GRU | Bringt Rollout-Training gegenüber Ein-Schritt-Anpassung etwas? Das ist die Hauptfrage. |
| LSTM | Hilft ein zweiter, additiv fortgeschriebener Speicher bei wenigen Läufen, oder ist die zusätzliche Kapazität ein Risiko? |
| xLSTM (mLSTM-lastig) | Hilft die matrixwertige Speicherzelle bei der langsamen Mode? Erst mLSTM, sLSTM nur bei Bedarf (im Paper xLSTM[7:1]). |
| Mamba | Ist ein selektiver Zustandsraum bei drei Dekaden Steifigkeit besser als ein Tor-Mechanismus? Als reine PyTorch-Referenz schreiben, nicht über `mamba-ssm` (CUDA-Kernel). **Umgesetzt (Befund 38):** O(1)-Kerneigenschaft bestätigt (Schrittzeit flach über 68 Schritte), aber 5.2-9.7x teurer als GRU/LSTM je nach Vergleichsmethode (Composite-Inferenz 365 µs/Fenster) und erst nach drei Stabilitäts-Fix-Zyklen lauffähig (RMSNorm behebt einen NaN-Kollaps durch unbeschränkt wachsenden Rollout-Eingang; Weight-Decay-Ausschluss für `A_log`/`D` ohne messbaren Effekt bei `n_train=16`). Im reduzierten Einzeldatenpunkt (1 Seed, `n_train=16`) bei `product_moisture_kg_per_kg` (h=36) schlechter als reine Persistenz. |
| Transformer | Kontrollgruppe ohne rekurrenten Zustand. **Umgesetzt (Befund 37):** bei H=36 (Befund 35, nicht mehr die 24 Schritte von Befund 23) ist die quadratische Attention **nicht** unkritisch -- kein KV-Caching im Decoder-Loop, 14-18x teurer als GRU/LSTM bei Training und Inferenz gleichermaßen, in einem reduzierten Einzeldatenpunkt (1 Seed, `n_train=16`) zugleich schlechter im NRMSE als GRU/LSTM. |

**Zielgröße: `product_moisture_kg_per_kg` im Mehrschritt-Rollout.** `energy_target` läuft
als Kontrollgröße mit, `d50_um` ist gestrichen (zu 84 % Rauschgrenze). Begründung nach
Schritt 6, Teil 1 geschärft: bei der Feuchte scheitern Persistenz-Unterbietung **und**
Nichtlinearität, der ARX-Rollout akkumuliert Fehler statt sie zu dämpfen. Das ist die
Lücke, für die Rollout-Training existiert.

**Vorgaben für alle Architekturen:**

- **Residual-Parametrisierung an einem Anker.** Der Kopf sagt Δy vorher; Nullgewichte
  entsprechen exakt der parameterfreien Baseline. **Präzisierung aus Befund 21:** wo es
  keinen Anker gibt (Soft-Sensor-Betrieb), ist der Kopf **direkt** zu parametrisieren —
  sonst ist die Vorhersage ein Integrator ohne Rückführung und driftet über den Horizont.
- **Kein Teacher Forcing als Standard.** Primärmetrik ist der Rollout; Parameter für ein
  Curriculum vorhanden, Voreinstellung 0.0.
- **Sequenzlänge ist eine Modellentscheidung, keine Hyperparameterfrage.** Bei Δt = 25 s
  und 600 s Horizont sind es 24 Schritte; die langsame Mode (16 000 s) liegt strukturell
  außerhalb und wird über die Streuung der Anfangsbedingungen abgedeckt. Befund 23 zeigt,
  dass die Baselines von mehr als zwei Verzögerungen nicht profitieren — ob ein Modell, das
  32 Schritte *lesen kann*, davon profitiert, ist eine der offenen Fragen von Teil 2.
- Physik-Constraints: Nichtnegativität und Wertebereiche als Output-Projektion, Monotonie
  (∂dE/∂T_in > 0, ∂dE/∂u_gas > 0) über monotone Netze / ICNN.

---

## Was bewusst noch NICHT gemacht wird

- **Kein RL.** Erst müssen stabile Mehrschritt-Rollouts stehen. Bei der Feuchte ist der
  beste Rollout auch rauschfrei und auch nichtlinear schlechter als Persistenz — ein
  RL-Agent auf diesem Surrogat würde Modellartefakte ausnutzen. Frühestens ab Monat 5–6.
- **Keine PSD-Aussagen aus dem Grey-Box-Modell.**
- **`p_nozzle` im RL nicht freigeben**, solange die Kopplung Tropfengröße → Benetzung →
  Wachstum fehlt. Der Laborbereich ist außerdem enger als der Platzhalter (1.0–1.8 bar
  gegen 1.0–4.0), was den scheinbaren Spielraum weiter verkleinert.
- **Kein blind geschriebener Architekturcode.** xLSTM und Mamba wurden erst geschrieben,
  nachdem geprüft war, dass sie auf dieser Maschine ausgeführt werden können (beide
  inzwischen umgesetzt, Befund 38 für Mamba).
- **Keine weitere Literatur.** Die offenen Fragen sind empirisch.

---

## Welche Lücken welchen Schritt blockieren

| Lücke | Blockiert | Umgehung bis dahin |
|---|---|---|
| Max. Änderungsraten der Stellgrößen | Sequenzlänge | Platzhalter; Δt-Vorstudie zeigt geringe Sensitivität |
| **Inline-Messbarkeit der Feuchte bei Anwendungspartner** | **welches Ergebnis gilt** | beide Betriebsarten implementiert und berichtet |
| **Sorptionsisotherme** | Zeitkonstanten, damit Δt und Sequenzlänge | Sensitivitätsrechnung liegt vor (`02_prozess_setup.md`) |
| Zeitkonstante des Lufterhitzers | dynamischer Anteil von `dE` | Feuchte als Hauptziel |
| Definition von `dE` | Gültigkeit der Monotonie-Constraints | beide Varianten implementiert |
| Rechenzeit pro Dyssol-Lauf | Endpunkt der Lernkurven | Referenzwert aus dem Vorprojekt: ~69 h für 128 Läufe |
| Rauschniveau der Anlagensensoren | ~~Sättigungshöhe~~ | Rangfolge über zwei Größenordnungen stabil; **offen ist der Ausreißerprozess** |
| **PyTorch in der Entwicklungsumgebung** | **Schritt 6, Teil 2 vollständig** | NARX trägt den Vergleich bis dahin |

**Fazit:** Ein Schritt ist jetzt echt blockiert — Teil 2 von Schritt 6 braucht eine Maschine
mit PyTorch. Alles andere läuft. Platzhalter sind zulässig, solange sie explizit und
versioniert sind, aber zwei von ihnen (Feuchteniveau, Sorptionsisotherme) bestimmen die
Versuchsplanung mit und sind jetzt als falsch erkannt, nicht nur als unbekannt.
