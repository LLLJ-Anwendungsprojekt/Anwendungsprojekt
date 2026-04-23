# Datenbasis aus GED + Indexdaten

Diese Anleitung erklaert, wie das Skript `src/build_analysis_dataset.py` aus
`GEDEvent_v25_1.csv` und `indexData.csv` eine robuste Analysebasis fuer
K-Means und KNN erstellt.

## Warum diese neue Datenbasis?

Das fruehere Join-Muster (ein Konflikt-Ereignis x alle Indizes des Tages) fuehrte zu
massiven Duplikaten und verzerrten Clustern. Die Cluster trennten hauptsaechlich
Preisniveaus von Indizes statt Konflikt- und Reaktionsmuster.

Die neue Logik setzt auf:
- Eine Zeile pro Konflikt-Ereignis
- Marktreaktionen als Rendite-/Volatilitaets-Features statt Rohpreise
- Klare Zielspalten fuer Klassifikation

## Input-Dateien

- `GEDEvent_v25_1.csv`
- `indexData.csv`

Standardpfade sind relativ zu `AWP/`:
- GED: `AWP/GEDEvent_v25_1.csv`
- Indizes: `AWP/indexData.csv`
- Output: `AWP/Anwendungsprojekt/data/processed/conflict_market_features.csv`

## Verarbeitungsschritte im Skript

### 1) GED laden und reduzieren

Es werden nur modellrelevante Felder geladen (keine Freitextspalten):
- Ereignis-/Kontextdaten: `type_of_violence`, `where_prec`, `event_clarity`, `date_prec`, `region`, `country_id`
- Geo: `latitude`, `longitude`
- Opferdaten: `deaths_a`, `deaths_b`, `deaths_civilians`, `deaths_unknown`, `best`, `high`, `low`
- Zeit: `date_start`, `date_end`, `active_year`

Zusatzfeature:
- `conflict_duration_days = date_end - date_start`

### 2) Indexdaten laden und normalisieren

Pro Index und Datum werden erzeugt:
- `daily_return = pct_change(Close)`
- `intraday_range = (High - Low) / Low`

Wichtig: Das Modell nutzt Renditen statt absolute Preise, damit z. B. NYA und N225
nicht allein wegen unterschiedlicher Preisniveaus getrennt werden.

### 3) Marktfenster je Ereignistag berechnen

Fuer jeden eindeutigen `date_start` werden ueber alle verfuegbaren Indizes berechnet:
- Pre-Fenster: letzte 10 Handelstage vor dem Ereignis
- Event-Tag: Rendite am Ereignistag
- Post-Fenster: erste 5 Handelstage nach dem Ereignis

Resultierende Marktfeatures:
- `pre_return`, `event_return`, `post_return`
- `pre_volatility`, `post_volatility`
- `market_reaction = post_return - pre_return`
- `volatility_change = post_volatility - pre_volatility`
- `n_indices_tracked`

### 4) Feature Engineering und Labels

Zusatzfeatures:
- `total_deaths = deaths_a + deaths_b + deaths_civilians + deaths_unknown`
- `lethality_ratio = best / number_of_sources`
- `region_enc` (Label-Encoding von `region`)

Zielspalten fuer Modelle:
- `market_direction_5d` (KNN):
  - `1` wenn `post_return >= 0`
  - `0` sonst
- `severity_class`:
  - Quantile-basierte Klassen aus `best` (0/1/2)

## Output

Datei:
- `Anwendungsprojekt/data/processed/conflict_market_features.csv`

Typischer Inhalt:
- Eine Zeile pro Ereignis
- Konfliktfeatures + Marktreaktionsfeatures + Zielspalten

## GitHub-Workflow fuer grosse Dateien

Die volle Datei `conflict_market_features.csv` ist gross und sollte nicht im Repo liegen.

Empfehlung fuer die Gruppe:
- Im Repo: nur kleine Sample-Datei committen, z. B. `data/processed/conflict_market_features_sample10k.csv`
- Lokal: volle Datei mit dem Skript neu erzeugen

Kleine Sample-Datei lokal erzeugen:

```bash
c:/playground/AWP/.venv/Scripts/python.exe -c "import pandas as pd; df=pd.read_csv('data/processed/conflict_market_features.csv'); s=df.sample(n=min(10000,len(df)), random_state=42); s.to_csv('data/processed/conflict_market_features_sample10k.csv', index=False)"
```

## Ausfuehrung

Aus `AWP/Anwendungsprojekt`:

```bash
c:/playground/AWP/.venv/Scripts/python.exe src/build_analysis_dataset.py
```

Optionen:

```bash
c:/playground/AWP/.venv/Scripts/python.exe src/build_analysis_dataset.py \
  --ged-path c:/playground/AWP/GEDEvent_v25_1.csv \
  --idx-path c:/playground/AWP/indexData.csv \
  --out-path c:/playground/AWP/Anwendungsprojekt/data/processed/conflict_market_features.csv \
  --min-indices 1
```

## Nutzung fuer K-Means und KNN

### K-Means (Clustering)

Empfehlung:
- Nutze numerische Konflikt- und Marktreaktionsfeatures
- Schließe Identifier/Freitext aus (`id`, `region` als Text etc.)
- Standardisiere Features vor dem Clustering

### KNN (Klassifikation)

Empfehlung:
- Zielvariable: `market_direction_5d`
- Features: Konflikt- + Marktreaktionsfeatures (ohne Leakage)
- Train/Test-Split zeitlich konsistent oder mindestens stratifiziert
- Feature-Scaling zwingend

## Bekannte Hinweise

- In sehr fruehen Randbereichen koennen einzelne Marktfenster fehlen.
- `event_return` kann NaNs enthalten, wenn am Ereignistag keine Boersenwerte vorliegen.
- `n_indices_tracked` sollte als Datenqualitaetsmerkmal mitbeachtet werden.
