# GPR & Aktienmärkte

Analyse des Zusammenhangs zwischen **geopolitischem Risiko (GPR-Index)** und der
Entwicklung internationaler **Aktienindizes**. Untersucht werden Daten von
2000–2021 mit vier Verfahren: lineare Regression (Event Study), K-Means,
Random Forest und KNN.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Verwendung

**Komplette Pipeline** – führt erst die Tests und danach drei der vier Verfahren
aus (Lineare Regression, K-Means, KNN):

```bash
python run_all.py
```

Bei fehlgeschlagenen Tests wird abgebrochen. In `results/` verbleiben danach
ausschließlich die Grafiken – von den Skripten erzeugte Zwischendateien
(CSV/TXT/…) werden zum Schluss automatisch entfernt.

**Random Forest wird zuletzt manuell ausgeführt.** Er liegt nur als Jupyter-
Notebooks in `src/random_forest/` vor und ist daher nicht Teil von `run_all.py`.
Nach der Pipeline die Notebooks in dieser Reihenfolge ausführen:

1. `random_forest_dataset_preparation.ipynb` – erzeugt den RF-Datensatz
2. `random_forest_full_period_results.ipynb` – trainiert und speichert die RF-Grafiken

**Einzelne Schritte** lassen sich auch direkt starten:

```bash
python src/build_dataset.py                          # Rohdaten -> data/processed/*
python src/lineare_regression/event_regression.py    # Lineare Regression (Event Study)
python src/kmeans/kmeans_tune.py                      # K-Means: Tuning + best_kmeans_pca.pdf
python src/kmeans/generate_tuning_reports.py          # K-Means: Boxplot-Reports
python src/knn/knn_gpr_analysis.py                    # KNN
python -m pytest tests/                               # nur die Tests
```

Die Ergebnis-Grafiken (PDF/PNG) werden nach `results/<verfahren>/` geschrieben.
Bei direktem Aufruf erzeugen die K-Means- und KNN-Skripte zusätzlich CSV/TXT-
Zwischendateien; `run_all.py` räumt diese am Ende weg.

## Ordnerstruktur

```text
.
├── README.md            # Diese Datei
├── requirements.txt     # Python-Abhängigkeiten (gepinnte Versionen)
├── run_all.py           # Tests + komplette Analyse-Pipeline ausführen
├── src/                 # Quellcode: Datenaufbereitung + Analyseverfahren
│   ├── build_dataset.py         # Roh- → aufbereitete Daten (Tages-/Monatspanel + Features)
│   ├── lineare_regression/      # Event Study: GPR-Schock → Aktienreaktion (OLS)
│   ├── kmeans/                  # K-Means-Clustering der Marktregime
│   ├── knn/                     # K-Nearest-Neighbors-Klassifikation
│   └── random_forest/           # Random-Forest-Notebooks (manuell, siehe Verwendung)
├── data/                # Datensätze
│   ├── raw/                     # Unveränderte Quelldaten (nicht bearbeiten)
│   └── processed/               # Von build_dataset.py erzeugte Panels
├── docs/                # Dokumentation: Architektur & Ablauf
│   ├── architecture_flow.png        # Ablauf/Reihenfolge der Skripte als PNG
│   └── architecture_components.png  # Schichten-/Komponentensicht als PNG
└── tests/               # Automatisierte Tests (pytest)
    ├── conftest.py              # Fixtures: lädt die aufbereiteten CSVs
    ├── test_schema.py           # Erwartete Spalten in den drei Dateien
    ├── test_integrity.py        # Keine NaN/Duplikate, Zeitraum & Indizes
    ├── test_labels.py           # Label-Spalten korrekt, kein Leak über Indizes
    └── test_event_study.py      # Konfigurations-Checks der Event Study
```

## Daten

### Quelldaten (`data/raw/`)

| Datei | Inhalt |
|-------|--------|
| `indexData.csv` | Tägliche Kurse internationaler Aktienindizes (Yahoo Finance): `Index`, `Date`, `Close`, `Adj Close`, … |
| `data_gpr_daily_recent.xls` | Geopolitical-Risk-Index nach Caldara & Iacoviello: `date`, `GPRD`, `GPRD_ACT`, `GPRD_THREAT`, … |

Betrachtet werden 12 Indizes (`KEEP_INDICES` in `build_dataset.py`), u. a.
GDAXI (DAX), IXIC (NASDAQ), N225 (Nikkei), HSI (Hang Seng), SSMI (SMI).

### Aufbereitete Daten (`data/processed/`)

Erzeugt durch `src/build_dataset.py` (Filter 2000-01-01 – 2021-05-31, Join über das Datum):

| Datei | Beschreibung |
|-------|--------------|
| `stocks_gpr_daily.csv` | **Tagespanel**: je Index/Tag der Schlusskurs + GPR-Werte. Basis der Event Study (lineare Regression). |
| `stocks_gpr_monthly.csv` | **Monatspanel**: letzter Monatsschlusskurs je Index + Monatsmittel des GPR, dazu prozentuale Veränderungen. |
| `stocks_gpr_features.csv` | Monatspanel **+ Features**: Lags (1–3), 6-Monats-Rollvolatilität, GPR-z-Score, Spike-Indikator, gewinsorisierte Renditen sowie die Zielvariablen `stock_down` und `gpr_up_next`. Basis für K-Means/KNN/Random Forest. |

### Wichtige Codes / Spalten

| Code | Bedeutung |
|------|-----------|
| `GPRD` | Geopolitical Risk Index (Tageswert, gesamt) |
| `GPRD_ACT` | Teilindex „Acts" – tatsächliche geopolitische Ereignisse |
| `GPRD_THREAT` | Teilindex „Threats" – geopolitische Bedrohungen/Drohungen |
| `*_daily_pct` / `*_monthly_pct` | Prozentuale Veränderung zum Vortag bzw. Vormonat |
| `gpr_spike` | 1, wenn \|GPR-Veränderung\| > 1 Standardabweichung |
| `stock_down` | Zielvariable: 1, wenn Monatsrendite < 0 |
| `gpr_up_next` | Zielvariable: 1, wenn GPR im Folgemonat steigt |
