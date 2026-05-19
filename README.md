3. Daten-Join & Analysefenster
3.1 Gemeinsamer Analysezeitraum
Analysezeitraum: Januar 2000 – Mai 2021  (257 Monate × 12 Indizes = 3.084 Panel-Beobachtungen)

✓ Begründung Startpunkt 2000: Alle 12 inkludierten Indizes liefern ab Januar 2000 vollständige Daten ohne Lücken. Alle fünf post-2000 Krisenperioden sind abgedeckt. Der Zeitraum 1985–1999 entfällt, da N100 erst ab 2000 und Shanghai/Shenzhen/KOSPI/TAIEX erst ab 1997 verfügbar sind — ein Einschluss würde ein stark unbalanciertes Panel erzeugen.

3.2 Tages-Join (stocks_gpr_daily.csv)
Methode: LEFT JOIN der Aktienhandelstage auf das GPR-Datum über Key Date. GPR liefert tägliche Werte, kein Forward-Fill erforderlich.

df_gpr   = gpr[['date','GPRD','GPRD_ACT','GPRD_THREAT']].rename(columns={'date':'Date'})
df_daily = stocks[stocks['Index'].isin(KEEP_12)].merge(df_gpr, on='Date', how='left')
df_daily = df_daily[(df_daily['Date'] >= '2000-01-01') & (df_daily['Date'] <= '2021-05-31')]
# Ergebnis: ~30.840 Zeilen × 12 Spalten (12 Indizes × ~257 Handelsmonate)

3.3 Monats-Aggregation mit Lead-Features
Für die Zeitreihenmodelle wird ein monatliches Panel mit expliziter Lead-Struktur erstellt:

# Aktienrendite: letzter Schlusskurs des Monats
m = df_daily.groupby(['Index','YearMonth'])['Close'].last()
m['stock_ret'] = m.groupby('Index').pct_change() * 100

# GPR: Monatsdurchschnitt der täglichen Werte
g = df_daily.groupby('YearMonth')[['GPRD','GPRD_ACT','GPRD_THREAT']].mean()
g['gprd_ret'] = g['GPRD'].pct_change() * 100

# Lead-Features: Stock(t) als Prädiktor für GPR(t+k)
g['gprd_ret_lead1'] = g['gprd_ret'].shift(-1)  # GPR in t+1
g['gprd_ret_lead2'] = g['gprd_ret'].shift(-2)  # GPR in t+2
g['gprd_ret_lead3'] = g['gprd_ret'].shift(-3)  # GPR in t+3

df_monthly = m.merge(g, on='YearMonth', how='left')
# Panel: 3.084 Beobachtungen (12 Indizes × 257 Monate)

3.4 Empirische Lead-Korrelation (Datenbefund)
Die folgende Tabelle zeigt die berechneten Korrelationen zwischen Stock(t) und GPR(t+k) über den Zeitraum 2000–2021. Negative Werte bestätigen: Aktien fallen heute → GPR steigt später.

Index (Region)	k=0 (gleichzeitig)	k+1 (1 Monat Lead)	k+2 (2 Monate Lead)	k+3 (3 Monate Lead)	
NYA (USA)	-0.126	-0.037	+0.004	-0.083	
IXIC (USA)	-0.197	-0.112	-0.020	-0.027	*
GSPTSE (N.Am.)	-0.124	-0.040	-0.004	-0.108	
GDAXI (Europa)	-0.185	-0.113	-0.023	-0.051	*
N100 (Europa)	-0.165	-0.069	-0.019	-0.083	
SSMI (Europa)	-0.130	-0.077	-0.062	-0.095	
N225 (Japan)	-0.102	-0.094	-0.093	-0.072	*
HSI (China/HK)	-0.127	-0.088	-0.050	-0.014	
000001.SS (C/HK)	-0.045	-0.028	-0.089	+0.003	
399001.SZ (C/HK)	-0.082	-0.029	-0.094	-0.020	
KS11 (As.-Paz.)	-0.131	-0.025	-0.069	-0.091	
TWII (As.-Paz.)	-0.219	+0.015	-0.063	-0.055	*
* Stärkster Lead-Effekt (k=0 oder k+1 < −0.09). Alle negativen Korrelationen konsistent mit Hypothese: Aktien antizipieren GPR-Bewegungen.
3.5 Feature Engineering
Zusätzliche abgeleitete Features für den ML-Ansatz:
•	Stock_lag1, Stock_lag2, Stock_lag3 — Verzögerte Aktienrenditen als Hauptprädiktoren für GPR(t+k)
•	GPR_zscore — Z-Score des GPRD (24-Monats-Fenster) für Krisenidentifikation
•	GPR_spike — Binär: 1 wenn GPRD_monthly_pct > 1 Standardabweichung (Extremereignis-Flag)
•	Stock_vol12 — Realisierte 12-Monats-Rollendsvolatiliät der Aktienrenditen
•	Crisis_dummy — Binär für definierte Krisenperioden post-2000: 9/11, Finanzkrise, Arab Spring, Ukraine, COVID
•	Region — Kategoriale Variable (5 Gruppen): USA, Nordamerika, Europa, Japan, China/HK, Asien-Pazifik
•	Stock_ret_sign — Vorzeichen der Rendite: +1 / −1 (für Klasssifikationsmodell)
 








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

## Neue Analyse-Datenbasis (GED + Indexdaten)

Eine gruppentaugliche Dokumentation zur Erstellung der neuen Datenbasis steht hier:
- [docs/DATENBASIS_GED_INDEX.md](./docs/DATENBASIS_GED_INDEX.md)

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
