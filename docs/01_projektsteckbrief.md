# Projektsteckbrief: TwinLink – ML-Teil (TUHH-B)

> Stand: 13.08.2026 · Pflegehinweis: nach jedem Verbundtreffen und nach jeder Lieferung
> aus TUHH-A aktualisieren.

## Meine Rolle

Wissenschaftlicher Mitarbeiter an der TU Hamburg, Institut für Feststoffverfahrenstechnik
und Partikeltechnologie (SPE). Verantwortlich für das Teilvorhaben **TUHH-B**: Training und
Validierung von ML-Modellen mittels Simulationen und Messungen (36 PM TUHH).

## Projekt

**TwinLink** – Digitaler Zwilling für Verbundstandorte zur Steigerung der Ressourcen-Effizienz
in der Verfahrenstechnik. Gefördert im 8. Energieforschungsprogramm (BMWK), Laufzeit 36 Monate.

**Verbundpartner**
| Partner | Rolle |
|---|---|
| Verbundkoordinator | Verbundkoordination, DZ-Framework, Verwaltungsschalen, Monitoring |
| TUHH / SPE | Dyssol-Modellierung (TUHH-A) + ML (TUHH-B) |
| Anwendungspartner | Anlagen- und Betriebsdaten, Anlagenbau |
| Software-Integrationspartner | Python-Schnittstelle für Dyssol, Regelungskonzepte, Energie-/CO₂-Bilanzen |
| assoziierter Partner | assoziiert, Testdaten und Szenarien |

**Exemplarischer Prozess:** Wirbelschichtsprühgranulation.

## Meine Arbeitspakete (TUHH-B)

| AP | Inhalt | Laufzeit |
|---|---|---|
| AP 1 | ML-Methodik für dynamische Regelung: Vergleich RL + RNN/xLSTM/Transformer; Auswahl nach Genauigkeit, Generalisierbarkeit, Echtzeitfähigkeit; Schwerpunkt **zuverlässige Vorhersage bei wenigen Trainingsdaten** | M1–M10 |
| AP 2 | Experimentelle Validierung: Laborversuche Wirbelschichtsprühgranulation mit variierenden Stellgrößen und Stoffsystemen; Kalibrierung; datengetriebene Übertragungsmodelle Simulation → Realität | M11–M21 |
| AP 3 | Reduzierung des Ressourcenbedarfs: ML-Modelle, die Dyssol-Daten in experimentelle Kontexte übertragen; Vorschläge für transiente Verfahrweisen mit minimalem Energiebedarf bei gleicher Produktqualität | M22–M29 |
| AP 4 | MLOps: robuste Update-/Trainingszyklen, automatisierte Überwachung (Concept Drift), Integration ins Framework des Verbundkoordinators | M30–M36 |

## Meilensteine

| MS | Ergebnis | Monat |
|---|---|---|
| M1 | Theoretische Vorarbeiten und Anforderungsanalysen abgeschlossen | 6 |
| M2 | **Methodik der Simulations- und ML-Modellierung dokumentiert** (Vorgehensweisen, Werkzeuge, Vorlagen; Reduzierung des Trainingsdatenbedarfs) | 21 |
| M3 | Alpha-Version des Digitalen Zwillings im Betrieb, mit mind. einer Anlage verknüpft | 27 |

M2 ist mein Zielmilestone. Das primäre Ergebnisformat dafür sind **Lernkurven**
(Fehler über Anzahl Trainingsläufe), nicht Einzelwerte.

Zwei Ergebnisse aus AP 1, die eigenständig in M2 gehören und dort nicht als Nebensache
stehen sollten:

- **Das Auflösungsvermögen eines Architekturvergleichs.** Mit drei Seeds sind Unterschiede
  unter 10 bis 15 % nicht nachweisbar; für eine Aussage auf 5-Prozent-Ebene braucht es 12
  bis 26 Seeds. Wer Lernkurven ohne diese Schwelle berichtet, berichtet Rauschen mit.
- **Der Trainingsdatenbedarf ist eine Eigenschaft der Modellklasse, nicht des Prozesses.**
  Der lineare ARX sättigt bei 16 Läufen, ein nichtlineares Modell mit identischer
  Verzögerungsstruktur läuft bei 64 noch weiter. Die Zahl „so viele Läufe braucht man" ist
  ohne Angabe der Modellklasse nicht interpretierbar.

## Abhängigkeiten

**Von TUHH-A (Kollege) – Dyssol-Modellierung**
| AP | Inhalt | Laufzeit |
|---|---|---|
| A-AP 1 | Grundoperationen erfassen und charakterisieren, Anlagenparameter, statisch/transient je Einheit, Referenz-Fließschemata | M1–M3 |
| A-AP 2 | Grundoperationen als Prozesseinheiten in Dyssol modellieren und validieren (Zielgenauigkeit 10 % für Partikelgröße und -feuchte); Untermodelle Strom, Gas, Abwärme | M4–M21 |
| A-AP 3 | Standortweites Fließschema, Toleranzen/Konvergenz, Szenarien, Überführung in den DZ | M22–M36 |

**Vom Software-Integrationspartner**
- AP 1 (M1–M7): **Python-Bibliothek um den Dyssol-Kern**, inkl. "Minimal-Modus" (nur Zielwerte,
  Zwischenergebnisse verworfen) für schnelle Läufe
- AP 2 (M8–M15): Regelungskonzepte in Dyssol, über Python integrierbar
- AP 3 (M16–M21): Energie- und CO₂-Bilanzen aus Fließbildern

**Vom Anwendungspartner**
- Historische Zeitreihendaten und Echtzeit-Monitoring am Standort
- **Neu und dringend: welche Größen inline gemessen werden**, insbesondere die
  Produktfeuchte. Der Architekturvergleich in AP 1 zentriert sich auf sie; ohne
  Inline-Messung ist es eine Soft-Sensor-Aufgabe mit anderer Baseline, anderer Primärmetrik
  und — nach den bisherigen Messungen — deutlich schlechterem erreichbaren Ergebnis.
- Wichtige Einschränkung: der DZ darf in dieser Projektphase **keine Betriebsparameter verändern**
  (Sicherheitsgründe) → Online-RL an der Realanlage ist ausgeschlossen

## Kritische Terminlage

Mein AP 1 startet M1, die neuen Dyssol-Granulationsmodelle sind erst ~M21 fertig. Überbrückung:

1. Bestehende Modelle aus dem Vorprojekt **TwinGuide** (2022–Anfang 2025). Achtung: dort wurde
   nur ein **Wirbelschichttrockner** implementiert und validiert; ein physikalisch konsistentes
   **Granulationsmodell** (Tropfenabscheidung, Partikelwachstum, Abrieb) war explizit **nicht**
   vollständig realisiert und wird in TwinLink erst neu gebaut.
2. Eigenes reduziertes Grey-Box-Prozessmodell in Python (siehe `02_prozess_setup.md`).

## Verwandte Vorarbeiten am Institut

- **TwinGuide** (DFG, trilateral mit dem Verbundkoordinator und dem Anwendungspartner): Knowledge-Module, Dyssol-Simulationen +
  Messdaten, bilanzbasierte Heuristiken. Keine lernende Regelung.
- **Soft Sensor für die Partikelfeuchte** (Kräuter, Zhou & Heinrich, ABS23, aus TwinGuide) —
  **die wichtigste interne Vorarbeit für TUHH-B.** Sie liefert vier Dinge, die AP 1 und AP 2
  direkt betreffen:
  1. **Die einzigen belastbaren Zahlenwerte im Projektwissen.** Anlage (ProCell® 5
     LabSystem, Kammer GF3), Stellgrößenbereiche, Abtastzeit 2 s, Chargendauer ~1.8 h,
     elf Inline-Kanäle, Rechenzeit des Dyssol-Trocknermodells. Eingearbeitet in
     `02_prozess_setup.md` als Status `LAB`.
  2. **Die Feuchte ist im SPE-Labor inline messbar** (Mikrowellenresonanz, kalibriert
     gegen Thermogravimetrie). Beim Anwendungspartner ist es offen — deshalb führt TUHH-B beide
     Betriebsarten (Vorhersage mit Messung / Soft Sensor ohne).
  3. **Ein Vergleichsmaßstab, der nicht übertragbar ist.** Das berichtete R² = 0.994
     entstand mit einer *zufälligen Aufteilung über Beobachtungen* bei Δt = 2 s und einer
     nicht kausal geglätteten Zielgröße; TUHH-B teilt strikt auf Lauf-Ebene und bewertet
     Mehrschritt-Rollouts. Die beiden Zahlen sind nicht vergleichbar. Das ist im Verbund
     aktiv zu erklären, nicht abzuwarten — Vorschlag: den Abstand beider Aufteilungen auf
     demselben Datensatz messen und als Methodenergebnis für M2 veröffentlichen.
  4. **Das Fließschema scheiterte an der Sorptionsisotherme** (die eines gröberen Materials
     war hinterlegt). Damit ist die Isotherme des tatsächlichen Stoffsystems eine der drei
     dringendsten offenen Zahlen — und der Grund, warum der Feuchte-Platzhalter im
     Grey-Box-Modell um Faktor 10 zu trocken war.
- **IGF-Projekt Prozessanomalien Wirbelschichtsprühagglomeration**: umfangreiche Laborversuche,
  überwachtes Lernen zur Anomaliedetektion (Überfeuchtung, Düsenverstopfung). Datenbank am SPE
  vorhanden und in TUHH-B-AP 2 zu erweitern.
- **DFG-Projekt mit UFSC**: Dyssol-basierte Simulation/Optimierung keramische Fertigung.

## Publikationsziel

Mind. 4 Beiträge in Fachzeitschriften, z. B. Powder Technology, Advanced Powder Technology,
Particuology. Konferenzen: DECHEMA/VDI Fachgruppe Agglomerations- und Schüttguttechnik,
DECHEMA/VDI Jahrestagung, Partec, Granulation Workshop.
