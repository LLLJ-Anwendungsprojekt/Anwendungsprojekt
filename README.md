# Anwendungsprojekt: CRISP-DM Data-Science

## Fact Sheet
https://telekom-my.sharepoint.de/:x:/r/personal/lukas_niessen_telekom_de/Documents/Datenschema%20AWP.xlsx?d=wbea638ca5d42487cb926684175e76b68&csf=1&web=1&e=lyTyS4


## Struktur

```
Anwendungsprojekt/
├── src/           # Code
├── data/raw/      # Rohdaten
├── data/processed/    # Prozessiert
├── configs/       # Konfiguration
├── tests/         # Tests
└── requirements.txt
```

## CRISP-DM Phasen
1. Business Understanding  
2. Data Understanding  
3. Data Preparation  
4. Modeling  
5. Evaluation  
6. Deployment

## Datensätze
- Stock Exchange: https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data
- UCDP Conflict Data: https://ucdp.uu.se/downloads/#ged_global

## Setup
```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Siehe [README_2.md](./README_2.md) für Dependencies.

## Algorithmen
1. DB Scan (Lorenz)
2. K-Means Clustering (Leonard)
3. Lineare Regression (Lukas)
4. KNN (Johannes)

## K-Means ausfuehren

```bash
python src/kmeans_analyse.py --data-path data/processed/ConfilicsIndex2010.zip --k-min 2 --k-max 10
```

Fuer die neue indexzentrierte Datenbasis mit durchgaengigem `NEU_DB`-Naming:

```bash
c:/playground/AWP/.venv/Scripts/python.exe src/kmeans_analyse_NEU_DB.py
```

## Neue Analyse-Datenbasis (GED + Indexdaten)

Eine gruppentaugliche Dokumentation zur Erstellung der neuen Datenbasis steht hier:
- [docs/DATENBASIS_GED_INDEX.md](./docs/DATENBASIS_GED_INDEX.md)

Fuer eine indexzentrierte Datenbasis mit tagesgenau gematchten Konfliktaggregaten
und strikt `date_prec = 1` steht zusaetzlich dieses Skript bereit:

```bash
c:/playground/AWP/.venv/Scripts/python.exe src/build_index_conflict_dataset.py
```

Hinweis fuer GitHub:
- Die volle Datei `data/processed/conflict_market_features.csv` ist zu gross fuer den normalen Repo-Workflow.
- Fuer das Repo bitte die kleine Beispieldatei `data/processed/conflict_market_features_sample10k.csv` verwenden.

Skript zur Erstellung:

```bash
c:/playground/AWP/.venv/Scripts/python.exe src/build_analysis_dataset.py
```

Outputs werden in `results/` gespeichert:
- `kmeans_cluster_assignments.csv`
- `kmeans_clusters_pca.png`
- `kmeans_summary.txt`

## Ad-hoc Analysen

Neu erstellte Analyse-Skripte liegen gebuendelt unter:
- `src/ad_hoc_analysen/`

Enthaltene Skripte:
- `src/ad_hoc_analysen/check_codebook.py`
- `src/ad_hoc_analysen/analyze_outlier.py`
- `src/ad_hoc_analysen/analyze_rwanda_market_impact.py`

Codebook-Referenz:
- `docs/references/ged251.pdf`

Ausfuehren (aus `Anwendungsprojekt/`):

```bash
c:/playground/AWP/.venv/Scripts/python.exe src/ad_hoc_analysen/check_codebook.py
c:/playground/AWP/.venv/Scripts/python.exe src/ad_hoc_analysen/analyze_outlier.py
c:/playground/AWP/.venv/Scripts/python.exe src/ad_hoc_analysen/analyze_rwanda_market_impact.py
```
