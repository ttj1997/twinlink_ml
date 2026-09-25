# Wärmenetz-Modul: Spezifikation für das künftige Dyssol-Bilanzmodul

> Stand: 18.09.2026 · Fassung 5 (Pinch-Zielwert/Problem-Table-Algorithmus als Massstab fuer
> die sechs Verschaltungsfaelle ergaenzt; Pinch als betriebspunktabhaengige Groesse
> ergaenzt, `lp-benchmark`-Zweig, Befund 60 und 61).
>
> Diese Datei beschreibt Dauerhaftes (Modulzweck, die drei Korrekturen gegenüber der
> Vorlage, der Skalenvorbehalt), kein Messergebnis mit Zeitpunkt -- deshalb ein eigenes
> Dokument analog zu `02_prozess_setup.md`, nicht ein weiterer Eintrag in
> `04_stand_ap1.md`. Die Vorbehalte, die Befund 52/57 (dort: Bruttoverbrauch ohne bzw. mit
> Rückgewinnungsgutschrift) betreffen, bleiben im Befund-Log -- siehe Verweis unten und
> `04_stand_ap1.md`, Befund 52d/57.

## Zweck

Der Grey-Box-Simulator (`simulation/greybox.py`) liefert über `simulation/exhaust.py`
den Abluftzustand des Prozesses, aber bislang kein Modell der Wärmerückgewinnung. Die
bisherige Optimierung (Befund 52, `04_stand_ap1.md`) minimiert deshalb nur den
Bruttoverbrauch (`power_total`) -- nicht den Netto-Verbrauch nach Rückgewinnung, der das
eigentliche Projektziel ist (Befund 52d nennt das bereits als einen von vier
Vorbehalten: "Ein Prozess, kein Verbund").

`heatnetwork/` schließt diese Lücke: aus Abluftzuständen (einer oder mehrerer Quellen)
und den Prozessluft-Senken, die aufgeheizt werden müssen, berechnet es die
rückgewinnbare Leistung und den resultierenden Brennstoffbedarf. Gebraucht wird das für

- die künftige Netto-Zielfunktion (Nachfolger von Befund 52, siehe unten),
- einen späteren LP-Benchmark,
- eine RL-Umgebung (`routing.MatrixRouting` ist bereits als deren Aktionsraum
  vorgesehen),
- die Schnittstelle zum Verbundpartner.

**Dieses Modul ist die Spezifikation für das künftige Dyssol-Bilanzmodul
(AP3 des Software-Integrationspartners).** Es portiert eine bestehende Excel-Bilanz des Verbundpartners (zwei
Trocknerlinien, zwei Wärmeübertrager, zwei Lufterhitzer) 1:1 in den Rechenvorschriften,
korrigiert dabei aber drei Defekte der Vorlage (siehe unten) -- ein künftiges
Dyssol-Modul, das dieselbe Vorlage liest, darf diese Defekte nicht erben, nur weil es
dieselbe Quelle hat.

## Die drei Korrekturen gegenüber der Excel-Vorlage

Ausführlich (mit den exakten Formeln) im Moduldocstring von
`src/twinlink_ml/heatnetwork/balance.py`. Zusammengefasst:

1. **Taupunktbezug je Quelle.** Die Vorlage lässt den zweiten Wärmeübertrager gegen den
   Taupunkt der ERSTEN Linie rechnen -- ein Kopierfehler zwischen zwei
   Tabellenblättern, kein Modellierungsentscheid. Jede Quelle im Modul hat ihre eigene,
   aus ihrer eigenen Abluftfeuchte berechnete Taupunktgrenze.

2. **Restenthalpie kaskadieren statt verwerfen.** Die Vorlage kappt die
   Senkentemperatur auf ihren Sollwert und verwirft die nicht genutzte
   Quellenenthalpie -- dokumentiert an einem Fall von 891,8 kW bei 87,4 °C, weit
   oberhalb jeder Taupunktgrenze und damit weiter nutzbar. Im Modul ist jede Quelle ein
   Strom mit über die Kaskade abnehmender Temperatur: nicht genutzte Enthalpie steht der
   nächsten Stufe (ggf. einer anderen Senke) zur Verfügung, bis die Taupunktgrenze
   erreicht ist. `tests/test_heatnetwork_balance.py::
   test_cascading_beats_the_uncorrected_excel_template` rechnet die unkorrigierte
   Vorlage nach und zeigt, dass die Kaskadierung strikt weniger Brennstoff braucht --
   die Zahl, die an den Verbundpartner geht.

3. **Mindestgrädigkeit je Stufe.** Die Vorlage prüft nirgends
   `T_Luft,aus <= T_Quelle - dT_min`; eine dortige 120-°C-Kappung verhindert eine
   Verletzung nur zufällig. Im Modul ist die Grädigkeit eine harte Nebenbedingung jeder
   Stufe, mit der Kaskadenreihenfolge Niedertemperaturquellen zuerst (sonst ist die
   Senkenluft bereits zu warm, um eine kalte Quelle noch aufzunehmen).

## Kondensationsmodus (MA1, Befund 57)

Anlass: der Verbundpartner erweitert sein Wärmemodell um einen KONDENSIERENDEN
Wärmeübertrager (Model v3) statt des bisher angenommenen rein sensiblen Betriebs. Damit
kehrt sich die Bedeutung der Abluftbeladung `Y_out` um -- ohne Kondensation hebt Feuchte
nur den Taupunkt (Störfaktor, engt die sensible Kühlgrenze ein), mit Kondensation liefert
sie zusätzliche latente Wärme (Ressource). Am Nominalpunkt des Simulators (Befund 57,
MA2) verifiziert: `+10 %` Störung der Produktfeuchte senkt `Q_rec` im sensiblen Modus um
1,60 % (-4,95 kW, Verbund-Massenstrom), hebt es im kondensierenden Modus um 2,73 %
(+14,20 kW) -- derselbe Vorzeichenwechsel, den ein Überschlag am Betriebspunkt des
Verbundpartners (8 kg/s, Y=0,01998, T_out=81,4 °C) bereits vor der Implementierung
nahelegte. **Einschränkung (Nachtrag NB1, Befund 57h): gilt nur, solange die Linie
tatsächlich kondensiert.** Bei der 15-°C-Kaltseite tut sie das; bei der realistischeren
Kaltseite von ~29 °C (siehe unten) nicht mehr -- dann gilt für genau diesen Betriebspunkt
wieder die sensible Betrachtung (Feuchte SENKT `Q_rec`).

`HeatNetworkConfig.recovery_mode` (`"sensible"` Default/bitgleich zum Stand vor MA1,
`"condensing"` neu) wählt den Modus -- Config-Feld, kein Funktionsparameter, damit der
gewählte Modus im `RunRecord` dokumentiert ist. Im kondensierenden Modus kühlt die Quelle
bis `cold_temperature_celsius` (`T_kalt`, nicht bis `Taupunkt + dT_approach`) und liefert
zusätzlich die latente Wärme des dabei anfallenden Kondensats
(`Kondensat = max(0, Y_out - Y_s(T_kalt)) · ṁ_L`, `Q_latent = Kondensat · Δh_v(T_kalt)`).
Formeln, Kaskadenintegration (latenter Vorrat je Quelle, erst nach der sensiblen Reserve
angezapft) und die drei MA1-Tests (sensibel bitgleich, Q_rec-Monotonie, stetiger Rückfall
oberhalb des Taupunkts, Massenbilanz des Kondensats) im Moduldocstring von
`heatnetwork/balance.py`.

**`T_kalt` -- Platzhalter, OPTIMISTISCH, mit offener Frage wie `dT_min`.** Default:
**15 °C** (`heatnetwork.config.DEFAULT_COLD_TEMPERATURE_CELSIUS`, unverändert -- dieser
Nachtrag ändert den Code-Default nicht, nur seine Einordnung in der Dokumentation).
Begründung des ursprünglichen Werts: plausibler Wert für einen Kaltwasserkreis
(Rückkühlwerk oder leicht gekühltes Kühlwasser), deckt sich mit dem eigenen Überschlag am
Betriebspunkt des Verbundpartners, der diesen Wert als Ankerpunkt verwendet -- weder
Messung noch Verbundpartner-Angabe.

**Nachtrag NB1 (Befund 57h): 15 °C ist zu niedrig, um ohne Kältemaschine erreichbar zu
sein.** 15 °C entspricht der Modell-Umgebungstemperatur -- die Kaltseite hätte damit eine
Treibtemperaturdifferenz von null. Das Leitsystem des Anwendungspartners nennt für die
Kompressorstation **29,1 °C Kühlwassertemperatur**, ein realer Anhaltspunkt für die
tatsächliche Kaltseite. `Q_rec` fällt zwischen 15 °C und 29,1 °C von 5893,2 kW auf
4502,2 kW (Summe beider Trocknerlinien, Sprühturm + Wirbelschicht, MA5-Werte); ab
`T_kalt≈25 °C` (oberhalb des Taupunkts der Wirbelschichtlinie, 24,92 °C) kondensiert die
Wirbelschichtlinie gar nicht mehr -- für sie gilt dann wieder die rein sensible
Betrachtung. Details und die volle Sweep-Tabelle in Befund 57h. Der Code-Default bleibt
15 °C (keine unbegründete Änderung ohne expliziten Auftrag), aber jede Aussage, die den
Default als realistischen Betriebspunkt liest, ist zu korrigieren -- 15 °C ist der
GÜNSTIGE Fall, nicht der erwartbare.

**Offene Frage (unverändert):** welches Temperaturniveau der tatsächliche Kaltwasserkreis
im Partnermodell fährt, insbesondere ob er sich mit dem in Befund 56 bestätigten
Warmwasserspeicher-Zwischenkreis (`B-3236`, für `dT_min`) deckt oder ein separater Kreis
ist -- die 29,1-°C-Kühlwassertemperatur der Kompressorstation ist ein Anhaltspunkt, keine
bestätigte Antwort für DIESEN Kreis. Aus der Rekonstruktion des Sechs-Fälle-Modells (MA5,
Befund 57, `Model_v3.xlsx`) ist zudem bekannt, dass die zwei Trocknerlinien dort
UNTERSCHIEDLICHE `T_kalt` fahren (Sprühturm-Linie 10 °C, Wirbelschichtlinie 15 °C) --
`HeatNetworkConfig.cold_temperature_celsius` ist netzweit, nicht je Quelle; eine
Mehrfachquellen-Rechnung mit unterschiedlichem `T_kalt` je Quelle (wie in MA5 gebraucht)
ist damit AUSSERHALB von `evaluate_cascade` zu bilden (siehe
`scripts/probe_six_case_reconciliation.py`, dort bewusst nicht in `heatnetwork/`
verallgemeinert, um die Netto-weite Konfiguration nicht zu verkomplizieren, solange nur
eine Studie sie braucht). Geht in den Fingerprint ein, löst dieselbe
`PlaceholderConfigWarning` aus wie `dT_min`.

## `dT_min` -- Platzhalter, mit offener Frage an Swantje

Default: **20 K** (`heatnetwork.config.DEFAULT_MINIMUM_PINCH_K`). Begründung: der
Verbundpartner modelliert den Zwischenkreis zwischen Abluft und Prozessluft als Black
Box -- vermutlich liegen darin zwei Übertragerstufen in Reihe (Abluft → Wasser,
Wasser → Prozessluft). Die effektive Grädigkeit über die Kette ist näherungsweise die
Summe der Einzelstufen-Grädigkeiten; 10 K ist ein für Gas/Gas-Übertrager üblicher
Richtwert für EINE Stufe, 20 K setzt zwei Stufen an. Zusätzlich gilt: eine zu klein
angesetzte Grädigkeit überschätzt die rückgewinnbare Leistung -- der konservative Wert
ist in dieser Richtung der ehrlichere Fehler.

**Offene Frage:** Besteht der Zwischenkreis tatsächlich aus zwei Stufen, oder wird
direkt Abluft gegen Prozessluft übertragen? Bei Direktübertragung wären 10-15 K
angemessener. Wie bei `critical_moisture_kg_per_kg` (Befund 54, `04_stand_ap1.md`) ist
der nächste Schritt eine Sensitivitätsrechnung über 5/10/15/20/25 K -- **nicht Teil
dieses Auftrags**, aber vorgemerkt: wenn das Ergebnis über den plausiblen Bereich
robust ist, erübrigt sich die Rückfrage an den Verbundpartner, wie schon bei
`critical_moisture_kg_per_kg`.

`dT_min` ist Pflichtparameter mit Default (`HeatNetworkConfig.minimum_pinch_k`), geht in
den Fingerprint ein, und löst wie jeder andere Platzhalter im Projekt eine
`PlaceholderConfigWarning` aus (`HeatNetworkConfig.warn_if_placeholder`).

**Teilantwort auf die offene Frage (Befund 56, `04_stand_ap1.md`).** Zwei
Leitsystem-Screenshots des Anwendungspartners (Wärmerückgewinnung, Anlage F-5) zeigen den
Zwischenkreis konkret: Kompressoren (Abwärmequelle) → Wärmeübertrager `W 3237` →
Warmwasserspeicher `B-3236` (drei Temperaturfühler, dort 57-69 °C je nach Zeitpunkt) →
Wärmeübertrager `W 3223` (Prozessluftvorwärmung). Das bestätigt zwei Dinge aus der obigen
Annahme direkt: der Kreis ist tatsächlich **zweistufig** (zwei Übertrager in Reihe, nicht
Direktübertragung), und ein **Pufferspeicher ist real vorhanden** (`B-3236`) -- stützt die
Wahl `dT_min = 20 K` als Summe zweier Einzelstufen-Grädigkeiten. Keine vollständige
Beantwortung: die Screenshots zeigen als Quelle die Kompressoren-Abwärme, nicht explizit die
Granulator-Abluft selbst -- ob und wie beide Quellen denselben Pufferkreis speisen, ist damit
nicht geklärt und bleibt die eigentliche offene Rückfrage an den Verbundpartner.

## Skalenvorbehalt

Das Verbundmodell rechnet mit rund 8 kg/s Prozessluft
(`evaluation.recoverable_power.DISTRICT_MASS_FLOW_KG_PER_S`), der Laborsimulator mit rund
0,03 kg/s (`dry_air_mass_flow_kg_per_s` in `data.trajectory.diagnostics`) -- zwei
Größenordnungen. **Jede kW-Zahl aus der Kopplung von Labor-Quellen mit Verbund-Senken
(oder umgekehrt) ist eine Skalierungsrechnung, keine Anlagenaussage** -- dieselbe
Einordnung wie in `evaluation/recoverable_power.py` (Befund 50) und dort bereits an den
NRMSE-/RMSE-Zahlen der X3-Auswertung durchgehalten.

## Anbindung an den Simulator

`simulation/exhaust.reconstruct_exhaust_state` liefert `outlet_temperature_celsius`
(den AUSLASS-Kanal) -- nicht `bed_temperature_celsius`. Bei
`ExhaustConfig.outlet_duct_heat_loss_k = 0.0` (Default) sind beide numerisch gleich, der
Vertrag an der Verbundschnittstelle ist aber der Auslass-Kanal (dieselbe Unterscheidung,
die `exhaust.py` bereits als Vertragskorrektur führt). Der Trockenluftmassenstrom kommt
aus `dry_air_mass_flow_kg_per_s` in den Diagnostics/`SteadyStateSolution`, nicht aus der
Gasgeschwindigkeit neu berechnet. Details (wörtliche Signatur, was geliefert wird und was
fehlt) im Moduldocstring von `balance.py`.

## Nicht Teil dieses Auftrags

Kein LP, kein Optimierer, keine RL-Umgebung, kein Einlesen der Excel-Datei selbst (MA5,
Befund 57, liest `Model_v3.xlsx`-Werte nur als Konstanten in einem eigenen Skript ein,
nicht als Datei-Parser). `routing.py` stellt die Verschaltungs-Abstraktion
(`HeatNetworkRouting`) bereit, mit einer festen und einer matrixbasierten Implementierung
(letztere der spätere RL-Aktionsraum) -- sie entscheidet nichts selbst.

## Pinch-Zielwert (Problem-Table-Algorithmus)

`heatnetwork/pinch.py` (Zweig `lp-benchmark`) beantwortet eine andere Frage als
`balance.evaluate_cascade`: nicht "wie viel Brennstoff braucht DIESE Verschaltung", sondern
"wie viel Brennstoff braucht JEDE DENKBARE Verschaltung mindestens" -- der Massstab, gegen
den die sechs Verschaltungsfaelle des Partnermodells (Befund 57) und jeder spaetere Regler
gemessen werden.

**Methode.** Fuer das reine Zielwertproblem (minimaler Heiz-/Kuehlbedarf, keine Frage nach
Anzahl/Anordnung der Waermeuebertrager) ist der klassische Problem-Table-Algorithmus
(Pinch-Analyse) exakt und braucht keinen Solver -- ein LP waere erst fuer die
Netzwerksynthese noetig (welcher Uebertrager wohin), nicht Teil dieses Moduls. Verschobene
Temperaturen (heiss `-dT_min/2`, kalt `+dT_min/2`), Kaskade der
Intervall-Waermebilanzen von der hoechsten Temperatur nach unten, groesste negative
Kaskadenlast = minimaler Heizbedarf, Nulldurchgang der (um den Heizbedarf verschobenen)
Kaskade = Pinch-Temperatur.

**Kondensation ohne zweite Formel.** Die Abluftquellen haben unterhalb ihres Taupunkts
keine konstante Waermekapazitaet mehr. Statt einer eigenen Kondensationsformel tastet
`hot_segments_for_source` die Enthalpie-Temperatur-Kurve am bereits vorhandenen
`balance.evaluate_cascade` (`recovery_mode="condensing"`) ab -- derselbe, bereits getestete
Code, nur wiederholt bei verschiedenen `T_kalt`-Werten aufgerufen. Oberhalb des Taupunkts
ist die Kurve exakt linear (ein Segment, kein Diskretisierungsfehler moeglich); unterhalb
wird sie an `n_segments_below_dew_point` Stuetzstellen abgetastet, mit dem Taupunkt selbst
als exakter Intervallgrenze.

**Parameter (`scripts/run_pinch_benchmark.py`, PA2-Datenfall).** `T_kalt` ueber
`{15; 25; 29,1; 35} °C`, `dT_min` ueber `{10; 15; 20} K` -- 12 Kombinationen, je verglichen
gegen den besten der sechs Faelle (`heatnetwork.balance.evaluate_cascade`, dasselbe Modul,
keine zweite Kaskadenimplementierung -- `T_kalt` ist hier netzweit EIN Parameter, das passt
direkt in `HeatNetworkConfig.cold_temperature_celsius`, anders als im aelteren
`probe_six_case_reconciliation.py` mit quellenindividuellem `T_kalt`).

**Untere Schranke, keine Bauvorschrift (Vorbehalt, MUSS bei jeder Verwendung genannt
werden).** Der Pinch-Zielwert setzt beliebig viele, ideal gegeneinander geschaltete
Waermeuebertrager voraus. Eine kleine Luecke zum besten konkreten Fall (im getesteten
Datenfall: 2,3-4,2 %) kann vollstaendig durch dessen Beschraenkung auf zwei Uebertrager
erklaert sein -- sie beweist nicht, dass keine bessere realisierbare Verschaltung existiert,
nur dass sie hoechstens so viel einsparen kann.
`tests/test_heatnetwork_pinch.py::test_pinch_target_never_exceeds_the_best_of_six_cases`
macht die Ungleichung (Pinch-Ziel <= bester Fall) explizit pruefbar -- schlaegt dieser Test
fehl, ist die Implementierung falsch, nicht die Faelle.

Details, Zahlen (inkl. Korrektur einer frueher genannten Taupunkt-Verwechslung, 47,9 °C
statt 44,94 °C fuer Q1) und der Nebenbefund, dass der Pinch fuer diesen Datenfall am
heissen Ende sitzt und deshalb unabhaengig von `T_kalt` ist, in Befund 60
(`04_stand_ap1.md`).

**Der Pinch ist selbst eine betriebspunktabhaengige Groesse, keine feste Eigenschaft des
Standorts:** weil er hier am Eintritt der Wirbelschichtlinie (Quelle 2) sitzt und deren
Ablufttemperatur ueber die Prozesssollwerte verstellbar ist, verschiebt eine Aenderung der
Prozessfahrweise den Standort-Pinch -- bis hin zu einem Sprung auf eine strukturell andere
Engpassstelle (Taupunkt von Quelle 1), sobald die Wirbelschicht-Ablufttemperatur dessen
Taupunkt unterschreitet (Befund 61, `scripts/run_process_pinch_sensitivity.py`).

**Nachtrag RA2 -- die T_kalt-Unabhaengigkeit des minimalen Heizbedarfs gilt AUCH im unteren
Regime, an zwei zulaessigen QA3-Punkten bestaetigt.** Bislang nur fuer das obere Regime
belegt (PA2-Datenfall, Pinch durchgehend am Eintritt von Quelle 2). `T_kalt` ueber
{15; 25; 29,1; 35} °C (`dT_min=20 K`) an beiden zulaessigen unteren-Regime-Punkten aus QA3
(`T_out`(Q2)=40,94 °C und 43,74 °C, Pinch = Q1-Kondensationskurve bei 44,94 °C): der
minimale Heizbedarf bleibt an BEIDEN Punkten ueber den gesamten Sweep exakt bitgleich
(4548,378 kW), nur der minimale Kuehlbedarf faellt mit steigendem `T_kalt` (z. B. Punkt 1:
3532,0 kW bei 15 °C auf 1372,8 kW bei 35 °C). Mechanistisch erwartbar, keine Ueberraschung
im Nachhinein: der minimale Heizbedarf ist im Problem-Table-Algorithmus durch die Kaskade
OBERHALB des Pinch bestimmt, `T_kalt` veraendert nur die UNTERE Grenze der
Quellensegmente (wie weit sie abgekuehlt werden) -- solange sich die Pinch-Lage selbst
nicht verschiebt (hier durchgehend 44,94 °C, unveraendert ueber den ganzen Sweep), bleibt
die Kaskade oberhalb des Pinch unberuehrt. Die Aussage "Kondensation/`T_kalt` senkt den
minimalen Heizbedarf nicht" ist damit NICHT auf das obere Regime zu beschraenken --
Befund 57h und der Pinch-Befundentwurf bleiben unveraendert gueltig, keine Korrektur
noetig. Vorbehalt: an nur zwei Punkten geprueft (mehr zulaessige untere-Regime-Punkte
liefert das grobe QA2-Gitter nicht, siehe RA3), Zahlen und Reproduzierbarkeit in
`04_stand_ap1.md`, Befund 61.

## Bezug zu Befund 52 und 57

Befund 52 (`04_stand_ap1.md`) bewertet einen festen Betriebspunkt gegen den
Störgrößenbereich, ausschließlich über den Bruttoverbrauch. Vorbehalt d dort nannte
bereits "Ein Prozess, kein Verbund" als Einschränkung. **Befund 57 quantifiziert diesen
Vorbehalt** (MA3/MA4): Netto-Zielfunktion (`evaluation/net_energy.py`,
Brennstoffäquivalent-Gutschrift `Q_rec/eta_H`, zwei Gewichtungsvarianten kW-Summe/
CO2-gewichtet) über dieselbe Optimierungsmechanik wie Befund 52
(`scripts/study_operating_point_regret_net.py`). Ergebnis: die Gasgeschwindigkeit
verlässt ihre untere Schranke NUR unter kW-summierter Gewichtung, nicht unter
CO2-Gewichtung -- die vorab formulierte Erwartung (Kondensation treibt die Verschiebung)
trifft nicht zu, die MA3-Gewichtung dominiert den Effekt. **Nachtrag NB2 (Befund 57i):**
unter den beiden inhaltlich begründeten Bewertungen (Kosten, CO2 -- Industriestrom grob
3-4x teurer je kWh als Gas, 1,8x mehr CO2e) lohnt sich der Tausch mehr Strom gegen mehr
Rückgewinnung NICHT; nur die kW-Summe, die beide Energieformen ohne Begründung
gleichsetzt, zeigt die Verschiebung. Die Zielgrößenwahl ist damit eine offene
**Verbundentscheidung mit unmittelbarer Konsequenz für den späteren RL-Aktionsraum**,
keine Implementierungsdetailfrage. Details, Zahlen und Vorbehalte in Befund 57.
