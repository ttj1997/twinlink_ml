# Arbeitsablauf

## Vor jedem Commit

```bash
ruff check --fix .        # behebt das Mechanische
ruff format .
ruff check .              # muss leer sein
mypy src
pytest -q                 # 193 passed (187 ohne PyTorch, 6 Torch-Tests)
nbstripout notebooks/*.ipynb
```

**Die Reihenfolge ist nicht beliebig.** `ruff check --fix` vor `ruff format`: der Formatter
bricht Zeilen um, und ein anschließendes `--fix` würde die Umbrüche teilweise wieder
zusammenziehen.

## Warum die Regelauswahl Ausnahmen hat

`pyproject.toml` schaltet vier Regelgruppen ab, jede mit Begründung an der Stelle. Die
wichtigste: `RUF001`/`RUF002` („ambiguous unicode") ist für ein deutschsprachiges
verfahrenstechnisches Projekt falsch — σ, τ, Δ, ϱ in Tabellenköpfen und Docstrings sind die
richtige Schreibweise, nicht ein Homoglyphen-Angriff.

**Regeln werden abgeschaltet, nicht mit `noqa` einzeln unterdrückt.** Ein `# noqa` ohne
Begründung ist ein verstecktes Ignore; im Projekt stehen `noqa`-Zeilen nur dort, wo die
Ausnahme lokal ist und der Kommentar den Grund nennt (`B027`, `N802`, `NPY002`, `ARG001`).

## Was der Linter nicht prüft, aber gilt

- Zeitspannen in Sekunden berichten, nicht in Abtastschritten.
- Jede berichtete Differenz zwischen zwei Modellen erscheint mit ihrer
  Auflösungsschwelle — dafür gibt es `architecture_verdicts`.
- Ein Modul, in dem kein Tensor-Vorgang ausgeführt wurde, liefert keine berichtbare Zahl
  und wird im Statusdokument als nicht ausgeführt gekennzeichnet.
