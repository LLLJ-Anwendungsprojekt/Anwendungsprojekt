# Datenbasis aus GED + Indexdaten

Diese Anleitung erklaert, wie das Skript `src/build_index_conflict_dataset.py` aus
`GEDEvent_v25_1.csv` und `indexData.csv` eine indexzentrierte Analysebasis erstellt.

## Warum diese neue Datenbasis?

Das fruehere Join-Muster (ein Konflikt-Ereignis x alle Indizes des Tages) fuehrte zu
massiven Duplikaten und verzerrten Clustern. Die Cluster trennten hauptsaechlich
Preisniveaus von Indizes statt Konflikt- und Reaktionsmuster.

Die aktuelle Logik setzt auf:
- Eine Zeile pro Index-Handelstag
- Tagesgenaues Matching von Konfliktaggregaten auf den Handelstag
- Strikten Filter auf `date_prec = 1`

## Input-Dateien

- `GEDEvent_v25_1.csv`
- `indexData.csv`

Standardpfade sind relativ zu `AWP/`:
- GED: `AWP/GEDEvent_v25_1.csv`
- Indizes: `AWP/indexData.csv`
- Output: `AWP/Anwendungsprojekt/data/processed/index_conflict_features.csv`

## Verarbeitungsschritte im Skript `src/build_index_conflict_dataset.py`

### 1) GED laden und filtern

Es werden nur notwendige GED-Felder geladen und ausschließlich Ereignisse mit `date_prec = 1` behalten.

### 2) Indexdaten laden und Features berechnen

Pro Index und Datum werden u. a. erzeugt:
- `daily_return = pct_change(Close)`
- `intraday_range = (High - Low) / Low`
- `volume_change = pct_change(Volume)`

Wichtig: Das Modell nutzt Renditen statt absolute Preise, damit z. B. NYA und N225
nicht allein wegen unterschiedlicher Preisniveaus getrennt werden.

### 3) Konflikte pro Tag aggregieren

Aus GED werden pro Kalendertag aggregierte Konfliktfeatures erzeugt, z. B.:
- `conflict_count`
- `fatalities_best_sum`, `civilian_deaths_sum`, `total_deaths_sum`
- `countries_affected`, `regions_affected`
- `type_of_violence_1_count`, `type_of_violence_2_count`, `type_of_violence_3_count`

### 4) Tagesgenaues Matching auf Indexdaten

Die Konfliktaggregationen werden per `date` auf jede Zeile der Indexdaten gemerged.
Tage ohne Konflikt erhalten fuer Konfliktspalten den Wert `0`.

## Output

Datei:
- `Anwendungsprojekt/data/processed/index_conflict_features.csv`

Typischer Inhalt:
- Eine Zeile pro Index-Handelstag
- Indexfeatures + tagesgenaue Konfliktaggregate

## Ausfuehrung

Aus `AWP/Anwendungsprojekt`:

```bash
c:/playground/AWP/.venv/Scripts/python.exe src/build_index_conflict_dataset.py
```

Optionen:

```bash
c:/playground/AWP/.venv/Scripts/python.exe src/build_index_conflict_dataset.py \
  --ged-path c:/playground/AWP/GEDEvent_v25_1.csv \
  --idx-path c:/playground/AWP/indexData.csv \
  --out-path c:/playground/AWP/Anwendungsprojekt/data/processed/index_conflict_features.csv
```

## Hinweise zur Nutzung

Die Analyse ist indexzentriert aufgebaut.

Eigenschaften:
- Basis ist jede Zeile aus `indexData.csv`
- Konflikte werden exakt ueber das Kalenderdatum auf den Handelstag gematcht
- Es werden nur GED-Ereignisse mit `date_prec = 1` einbezogen
- Konflikte werden pro Tag aggregiert, z. B. `conflict_count`,
  `fatalities_best_sum`, `countries_affected` und Counts je
  `type_of_violence`

Ausfuehrung:

```bash
c:/playground/AWP/.venv/Scripts/python.exe src/build_index_conflict_dataset.py \
  --ged-path c:/playground/AWP/GEDEvent_v25_1.csv \
  --idx-path c:/playground/AWP/indexData.csv \
  --out-path c:/playground/AWP/Anwendungsprojekt/data/processed/index_conflict_features.csv
```
