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
2. Lineare Regression (Lukas)
3. KNN (Johannes)

## Neue Analyse-Datenbasis (GED + Indexdaten)

Eine gruppentaugliche Dokumentation zur Erstellung der neuen Datenbasis steht hier:
- [docs/DATENBASIS_GED_INDEX.md](./docs/DATENBASIS_GED_INDEX.md)

Fuer eine indexzentrierte Datenbasis mit tagesgenau gematchten Konfliktaggregaten
und strikt `date_prec = 1` steht zusaetzlich dieses Skript bereit:

```bash
c:/playground/AWP/.venv/Scripts/python.exe src/build_index_conflict_dataset.py
```

Hinweis fuer GitHub:
- Die Ausgabedatei `data/processed/index_conflict_features.csv` kann lokal neu erzeugt werden und muss nicht in voller Groesse versioniert werden.

Skript zur Erstellung:

```bash
c:/playground/AWP/.venv/Scripts/python.exe src/build_index_conflict_dataset.py
```

## Hinweise

- Die aktuelle Datenpipeline ist auf die indexzentrierte Datenbasis ausgerichtet.
- Das Build-Skript liegt in `src/build_index_conflict_dataset.py`.
