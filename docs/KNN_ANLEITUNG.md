# KNN Klassifikation für Geopolitische Konflikte → Aktienmärkte
## Production-Ready Python Skript

**Autor:** Johannes  
**Modul:** Anwendungsprojekt - CRISP-DM  
**Phase:** Modeling & Evaluation  

---

## 📊 Überblick

Dieses Skript implementiert einen **K-Nearest Neighbors (KNN) Klassifikator**, um vorherzusagen, ob ein Aktienmarkt bei geopolitischen Konflikten **steigt** oder **fällt**.

### Features:
✅ **Production-Ready Code** mit Logging und Fehlerbehandlung  
✅ **Automatische Hyperparameter-Optimierung** (GridSearchCV)  
✅ **5-Fold Cross-Validation** für robuste Evaluation  
✅ **Umfangreiche Metriken** (Accuracy, Precision, Recall, F1, ROC-AUC)  
✅ **Visualisierungen** für wissenschaftliche Arbeiten (Heatmaps, ROC-Kurve)  
✅ **Modell-Persistierung** (speichere/lade trainierte Modelle)  
✅ **Data Preprocessing Pipeline** (NaN-Handling, Outlier-Entfernung)  

---

## 📁 Projektstruktur

```
├── src/
│   ├── knn_analyse.py          # Hauptskript (KNNAnalyzer Klasse)
│   ├── knn_example.py          # Beispiel-Skripte & Demos
│   └── utils.py                # Utility-Funktionen (Data Loading, etc.)
├── configs/
│   └── knn_config.yaml         # Konfiguration (hyperparameter, paths)
├── data/
│   ├── raw/                    # Rohdaten von UCDP und Kaggle
│   └── processed/              # Bereinigter Datensatz
├── models/                     # Trainierte KNN-Modelle (PKL)
├── results/                    # Ergebnisse und Visualisierungen
└── tests/
    └── test_knn.py             # Unit Tests
```

---

## 🚀 Quick Start

### 1. Vorbereitung

```bash
# Aktiviere venv
venv\Scripts\activate

# Installiere Dependencies
pip install -r requirements.txt
```

### 2. Datensatz vorbereiten

**Dein merged Dataset sollte diese Struktur haben:**

```
merged_data.csv:
┌─────┬──────────────┬─────────┬────────────┬──────────┐
│ date│ close        │ volume  │ conflict_count │ deaths │
├─────┼──────────────┼─────────┼────────────┼──────────┤
│ 2020-01-01 │ 100.50      │ 1000000 │ 2          │ 50     │
│ 2020-01-02 │ 101.25      │ 1100000 │ 0          │ 0      │
│ ...
└─────┴──────────────┴─────────┴────────────┴──────────┘

Features: `close`, `volume`, `conflict_count`, `deaths`, ... (alle numerisch!)
Target: Wird automatisch aus `close` berechnet (1=steigt, 0=fällt)
```

**Daten-Zusammenführung:**
```python
# Beispiel mit UCDP + Kaggle Daten:
from utils import merge_conflict_and_stock_data

merged_df = merge_conflict_and_stock_data(
    conflict_file='data/raw/ucdp_ged_events.csv',
    stock_file='data/raw/stock_prices.csv',
    merge_on='date'
)
merged_df.to_csv('data/processed/merged_data.csv', index=False)
```

### 3. KNN Analyse ausführen

**Option A: Komplette Pipeline (empfohlen)**

```python
from knn_analyse import KNNAnalyzer

analyzer = KNNAnalyzer(random_state=42)

results = analyzer.run_pipeline(
    data_path='data/processed/merged_data.csv',
    target_col='market_direction',
    k_range=range(1, 21)  # Teste k von 1 bis 20
)
```

**Option B: Schritt für Schritt (für Kontrolle)**

```python
analyzer = KNNAnalyzer()

# Daten laden
df = analyzer.load_data('data/processed/merged_data.csv')

# Daten vorbereiten (Train/Test Split)
analyzer.prepare_data(df, target_col='market_direction', test_size=0.2)

# Bestes k finden (Hyperparameter Tuning)
best_k = analyzer.find_optimal_k(k_range=range(1, 21))

# Modell trainieren
analyzer.train(k=best_k)

# Evaluation
metrics = analyzer.evaluate()
analyzer.print_results()

# Visualisierungen erstellen
analyzer.visualize_results(output_dir='results')

# Modell speichern
analyzer.save_model(output_dir='models')
```

---

## 📖 Detailed Documentation

### KNNAnalyzer Hauptmethoden

#### `__init__(data_path=None, random_state=42)`
Initialisiert den Analyzer

#### `load_data(filepath)`
Lädt CSV-Datei mit Features und Target

#### `prepare_data(df, target_col='market_direction', test_size=0.2)`
- Entfernt NaN-Werte
- Teilt in Train/Test (stratifiziert, für balancierte Klassen)
- **Standardisiert Features** (Z-Score Normalisierung) ⚠️ *Wichtig für KNN!*

#### `find_optimal_k(k_range=range(1, 21))`
- GridSearchCV über k-Werte
- 5-Fold Cross-Validation
- Scoring: F1-Weighted (gut für unbalancierte Daten)
- **Rückgabe:** Beste k mit höchstem CV-Score

#### `train(k=5)`
Trainiert KNN mit `n_neighbors=k`

#### `evaluate()`
Berechnet auf **Test-Daten**:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (für binäre Klassifikation)
- Confusion Matrix
- Cross-Validation Scores

#### `visualize_results(output_dir='results')`
Erstellt 4 Plots in PNG (300 DPI):
1. **Confusion Matrix Heatmap**
2. **Performance Metriken Bar Chart**
3. **ROC-Kurve** (inklusive AUC)
4. **Predictions vs Actual Distribution**

#### `save_model(output_dir='models')`
Speichert mit Timestamp:
- `knn_model_YYYYMMDD_HHMMSS.pkl` (Modell)
- `scaler_YYYYMMDD_HHMMSS.pkl` (StandardScaler)
- `results_YYYYMMDD_HHMMSS.pkl` (Metriken & Vorhersagen)

---

## 🛠️ Utils Funktionen

### Data Preprocessing

```python
from utils import *

# Lade Konfiguration
config = load_config('configs/knn_config.yaml')

# Merge Konflikt + Stock Daten
merged = merge_conflict_and_stock_data(
    'data/raw/conflicts.csv',
    'data/raw/stocks.csv',
    merge_on='date'
)

# Erstelle Target-Variable
df = create_target_variable(df, price_col='close')

# Handle Missing Values
df = handle_missing_values(df, strategy='drop')

# Entferne Ausreißer (IQR-Methode)
df = remove_outliers_iqr(df, threshold=3.0)

# Wähle Features
X, feature_names = select_features(df, exclude_cols=['date', 'target'])

# Speichere verarbeitete Daten
save_dataset(df, 'data/processed/clean_data.csv')
```

### Inference (Vorhersagen mit geladenem Modell)

```python
# Lade trainiertes Modell und Scaler
model, scaler, results = load_model_artifacts('models/knn_model_20240101_120000.pkl')

# Neue Daten vorbereiten
X_new = pd.read_csv('data/new_conflicts.csv')

# Mache Vorhersagen
predictions = make_predictions(model, scaler, X_new)
# predictions: [1, 0, 1, ...] (1=Markt steigt, 0=Markt fällt)
```

---

## 📊 Output & Ergebnisse

Nach `analyzer.run_pipeline()` findest du:

### `results/knn_results.png` 
4-Panel Visualisierung (publication-ready)

### `models/knn_model_*.pkl`
Trainiertes Modell zum Laden in Production

### `knn_analysis.log`
Detailliertes Log aller Schritte

### Console Output
```
Accuracy:  0.7234
Precision: 0.6891
Recall:    0.7452
F1-Score:  0.7157
ROC-AUC:   0.8123

CV-Score:  0.6952 ± 0.0445
```

---

## ⚙️ Konfiguration (knn_config.yaml)

```yaml
data:
  processed_path: '../data/processed/'
  target_column: 'market_direction'
  test_size: 0.2

model:
  k_range: [1, 21]        # Range für GridSearch
  metric: 'euclidean'     # oder 'manhattan', 'chebyshev'
  weights: 'distance'     # oder 'uniform'

cv:
  n_splits: 5
  strategy: 'stratified'

features:
  scaling: true           # Z-Score Normalisierung
  scaler_type: 'standard'
  remove_outliers: false
```

Ändere diese, um Hyperparameter anzupassen!

---

## 🧪 Testing

```bash
cd tests
python test_knn.py
```

Testet:
- ✅ KNNAnalyzer Initialisierung
- ✅ Data Preparation
- ✅ Train/Predict
- ✅ Evaluation Metriken
- ✅ Utility-Funktionen
- ✅ Full Pipeline Integration

---

## 💡 Best Practices & Tipps

### 1. **Standardisierung ist KRITISCH für KNN!**
```python
# ✅ RICHTIG: StandardScaler nutzen
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ❌ FALSCH: Ohne Scaling funktioniert KNN schlecht
```

### 2. **Stratifiziertes Train/Test Split**
```python
# Balanciert Klassen in Train & Test
train_test_split(..., stratify=y)
```

### 3. **K-Wert richtig wählen**
- `k=1-3`: Zu flexibel, Overfitting-Risiko
- `k=5-11`: Gut für Klassifikation (default k=5)
- `k>15`: Zu rigid, kann Patterns übersehen

**Verwende GridSearchCV um optimales k zu finden!**

### 4. **Cross-Validation für robusten Estimate**
```python
# 5-Fold CV gibt besseren Überblick als nur Train/Test Split
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

### 5. **Feature Scaling bewusst wählen**
```python
# KNN ist distanz-basiert → alle Features sollten ähnliche Skala haben
# Normalisierung: [0, 1]  bzw.  Standardisierung: mean=0, std=1
```

---

## 🔧 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'sklearn'"
```bash
pip install scikit-learn==1.4.1
```

### Problem: "FileNotFoundError: data/processed/merged_data.csv"
- Stelle sicher, dass dein Datensatz unter `data/processed/merged_data.csv` existiert
- Oder passe den Pfad in `main()` an:
```python
analyzer.run_pipeline(data_path='data/processed/dein_file.csv')
```

### Problem: "KeyError: 'market_direction' not found"
- Erstelle die Target-Variable zuerst:
```python
df = create_target_variable(df, price_col='close')
```

### Problem: Zu schlechte Accuracy
- **Mehr Daten sammeln** (KNN braucht genug Training-Samples)
- **Features verbessern** (Feature Engineering)
- **k-Wert anpassen** via Tuning
- **Ausreißer entfernen** (können KNN stark beeinflussen)
- **Class Imbalance**? Nutze `class_weight='balanced'`

---

## 📚 Wissenschaftliche Referenzen

**KNN Klassifikation:**
- Cover, T., & Hart, P. (1967). "Nearest neighbor pattern classification"
- Altman, N. S. (1992). "An introduction to kernel and nearest-neighbor nonparametric regression"

**CRISP-DM Prozess:**
- Wirth, R., & Hipp, J. (2000). "CRISP-DM: Towards a standard process model for data mining"

---

## 📝 Lizenz & Kontakt

**Autor:** Johannes  
**Projekttyp:** Universitäts-Anwendungsprojekt  
**Datensätze:**
- UCDP Conflict Data: https://ucdp.uu.se/downloads/
- Kaggle Stock Data: https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data

---

## 🎓 Für deine wissenschaftliche Arbeit

**Verwende diese Section in deiner Arbeit:**

> *K-Nearest Neighbors (KNN) ist ein nicht-parametrischer überwachter Lernalgorithmus, der zur Klassifikation basierend auf Ähnlichkeit zu bekannten Beispielen verwendet wird. Für diese Analyse wird ein 5-facher Cross-Validation mit GridSearchCV durchgeführt, um die optimale Anzahl von Nachbarn (k) zu bestimmen. Die Merkmale werden standardisiert, um sicherzustellen, dass größere Merkmalswerte den Abstandsberechnung nicht dominieren.*

**Metriken erklären:**
- **Accuracy:** Anteil korrekter Vorhersagen
- **Precision:** Von den als "Steig" vorhergesagten, wie viele stimmen?
- **Recall:** Von den eigentlich "Steig" Fällen, wie viele haben wir gefunden?
- **F1-Score:** Harmonic mean von Precision & Recall
- **ROC-AUC:** Trade-off zwischen TPR und FPR

---

## ✅ Checkliste vor Submission

- [ ] Datensatz unter `data/processed/merged_data.csv`
- [ ] `pip install -r requirements.txt` erfolgreich
- [ ] Skript mit `python src/knn_analyse.py` lauffähig
- [ ] Visualisierungen in `results/knn_results.png` generiert
- [ ] Modelle in `models/` gespeichert
- [ ] Tests bestanden: `python tests/test_knn.py`
- [ ] `knn_analysis.log` dokumentiert alle Schritte

**Viel Erfolg mit deiner wissenschaftlichen Arbeit! 🚀**
