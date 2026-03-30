# Python-Bibliotheken für CRISP-DM Projekte

## 1. Datenexploration & Analyse

Für das Laden, Bereinigung und erste Analyse der Rohdaten.

| Bibliothek | Beschreibung | Use-Case |
|-----------|-------------|----------|
| **pandas** | Data Frame Manipulation, Datenbereinigung | CSV/Excel laden, Merging, Aggregation, Fehlende Werte handhaben |
| **numpy** | Numerische Berechnungen, Vektoroperationen | Matrix-Operationen, statistische Berechnungen |
| **scipy** | Statistische Tests, Optimierungsalgorithmen | Korrelation, Hypothesentests, Interpolation |

## 2. Visualisierung

Ergebnisse verständlich darstellen und Muster in Daten erkennen.

| Bibliothek | Beschreibung | Wann nutzen |
|-----------|-------------|----------|
| **matplotlib** | Basis-Plots (Linien, Balken, Scatter) | Explorative Datenanalyse, Publikationen |
| **seaborn** | Statistisch erweiterte Visualisierungen | Korrelationen, Verteilungen, Heatmaps |
| **plotly** | Interaktive, browserbasierte Grafiken | Dashboards, Reports mit Zoom/Hover |
| **streamlit** | Schnelle Web-Apps ohne Frontend-Code | Prototypen, Live-Dashboards |
| **dash** | Produktive, skalierbare Web-Apps | Enterprise-Dashboards mit erweiterten Features |

## 3. Preprocessing & Modeling

Daten vorbereiten und Vorhersagemodelle trainieren.

| Bibliothek | Beschreibung | Was macht es |
|-----------|-------------|----------|
| **scikit-learn** | Standard ML Library mit allen Basis-Algorithmen | Feature Scaling, Encoding, Regression, Klassifikation, Clustering, Cross-Validation |
| **statsmodels** | Klassische statistische Modelle | OLS-Regression, ARIMA für Zeitreihen, GLM |
| **xgboost** | Hochperformantes Gradient Boosting | Komplexe Muster in Daten lernen, Kaggle-Standard |
| **lightgbm** | Schnelle Alternative zu XGBoost | Große Datasets effizient verarbeiten |

## 4. Evaluation & Deployment

Modelle bewerten und produktiv machen.

| Bibliothek | Beschreibung | Zweck |
|-----------|-------------|----------|
| **scikit-learn** (metrics) | Evalutationsmetriken und Cross-Validation | ROC/AUC, Confusion Matrix, Accuracy, Precision/Recall |
| **streamlit** | Interaktive Prototyp-Dashboards | Resultate visualisieren und reale Benutzer testen |
| **flask** | Leichte REST-API für Modell-Vorhersagen | Modell in Production bereitstellen, andere Apps integrieren |

## 5. Testing & Utilities

Code-Qualität & Konfigurationsmanagement.

| Bibliothek | Beschreibung | Wichtig für |
|-----------|-------------|----------|
| **pytest** | Unit- und Integrationstests | Sicherstellen, dass Code funktioniert, Regressionstests |
| **python-dotenv** | Umgebungsvariablen (.env-Datei) | API-Keys, Database URLs sicher speichern |
| **pyyaml** | YAML-Konfigurationsdateien | Settings zentral verwalten, reproducibility |
| **jupyter** | Interaktive Notebooks | EDA, Experimentieren, Dokumentation |

## Installation

```bash
pip install -r requirements.txt
```

Oder einzeln:
```bash
pip install pandas numpy scipy matplotlib seaborn plotly streamlit scikit-learn xgboost pytest
```
