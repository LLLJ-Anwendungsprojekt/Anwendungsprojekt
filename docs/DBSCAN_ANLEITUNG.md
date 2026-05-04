# DBSCAN-Analyse: Geopolitische Konflikte & Aktienindizes

## Forschungsfrage
**Wie beeinflussen geopolitische Konflikte mit ähnlichen Mustern die Reaktion von Aktienindizes?**

## Workflow

### 1️⃣ Feature Engineering – Warum ist das kritisch?

#### 1.1 Log-Transformation der Todeszahlen
```python
df["log_deaths"] = np.log1p(df["total_deaths"])
```

**Warum?**
- **Problem:** Todeszahlen sind extrem **rechtsschief** verteilt
  - Beispiel: 1, 2, 3, 5, 10, 50 → plötzlich 1000, 5000, 10000
  - Der Mittelwert wird durch Ausreißer dominiert
  - DBSCAN würde Cluster verzerrung bekommen
- **Lösung:** Log-Transformation komprimiert die Skala
  - log(1)=0, log(10)≈2.3, log(100)≈4.6
  - Verteilung wird symmetrischer
  - Ausreißer haben weniger Einfluss

#### 1.2 Kategorische Variablen (One-Hot-Encoding)
```python
type_violence_encoded = pd.get_dummies(df["type_of_violence"], prefix="violence")
```

**Warum?**
- `type_of_violence` (z.B. "Organized violence", "One-sided violence") ist kategorisch
- DBSCAN arbeitet mit Distanzmetriken (euklidisch, Manhattan)
- Das funktioniert nur mit numerischen Variablen
- One-Hot-Encoding wandelt Kategorien in binäre Spalten um

#### 1.3 Featureliste für DBSCAN
```
latitude              # Geografische Position
longitude            # Geografische Position
log_deaths           # Konflikt-Intensität (log-transformiert)
lethality_ratio      # Tödlichkeit pro Tag
conflict_duration_days  # Konflikt-Dauer
violence_type_*      # Kategorien (One-Hot encoded)
```

---

### 2️⃣ Standardisierung – Die zentrale Anforderung

#### Das Problem ohne Standardisierung
DBSCAN berechnet Distanzen zwischen Datenpunkten:

```
Distanz = √[(lat_a - lat_b)² + (deaths_a - deaths_b)²]
```

**Beispiel ohne Standardisierung:**
```
Punkt A: latitude=52.5, deaths=10
Punkt B: latitude=52.6, deaths=50

Differenz latitude:  0.1   (Einheit: Grad)
Differenz deaths:    40    (Einheit: Todesfälle)

Distanz ≈ √[0.1² + 40²] = √1600.01 ≈ 40.0

Problem: 
→ deaths dominiert die Distanzberechnung (40.0 >> 0.1)
→ latitude wird quasi ignoriert
→ geografische Nähe spielt keine Rolle
```

#### Die Lösung: StandardScaler
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Was passiert?**
- Jede Variable wird auf **Mittelwert = 0, Standardabweichung = 1** normalisiert
- Formel: `x_norm = (x - mean) / std`
- **Ergebnis:** Alle Features haben gleiche Gewichtung

**Beispiel mit Standardisierung:**
```
Punkt A: latitude_norm=−0.5, deaths_norm=−1.0
Punkt B: latitude_norm=+0.5, deaths_norm=+1.0

Differenz latitude_norm:   1.0
Differenz deaths_norm:     2.0

Distanz = √[1.0² + 2.0²] ≈ 2.24

Resultat:
→ Beide Features tragen proportional bei
→ Geografische Nähe wird berücksichtigt
→ Todeszahlen werden berücksichtigt
→ Fair und reproduzierbar
```

**Ohne Standardisierung → falsche (oder gar keine) Cluster!**

---

### 3️⃣ Epsilon-Bestimmung – Die kritischste Entscheidung

DBSCAN hat zwei Hyperparameter:
- **eps (ε):** Radius um jeden Punkt
- **min_samples:** Minimale Punkte im Radius zum "Core Point"

#### Die ε-Strategie

**Problem:**
- ε zu klein → alle Punkte sind Noise, keine Cluster
- ε zu groß → ein massiver Cluster mit allem
- **Richtig:** Nur ähnliche Konflikte werden geclustert

#### k-distance Plot
```python
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)

distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, 4])  # k=4+1

plt.plot(distances)
plt.title("k-distance plot (k=5)")
plt.xlabel("Points (sorted)")
plt.ylabel("Distance to 5th nearest neighbor")
plt.show()
```

**Interpretation:**
- Die Kurve zeigt, wie Distanzen zu den k-nächsten Nachbarn verteilt sind
- **Der Knick (Elbow)** = Übergang von dichten zu dünn besetzten Regionen
- **ε am Knick wählen** = optimale Balance

**Beispiel:**
```
Kurve flach bei 0.1 bis Punkt 5000
Dann plötzlich steil ansteigend ab 0.5

→ Knick bei ε ≈ 0.3–0.5
→ Dies ist der "natürliche" Bruch im Datensatz
```

---

### 4️⃣ DBSCAN anwenden

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
df["cluster"] = dbscan.fit_predict(X_scaled)
```

**Ergebnis:**
- `cluster = -1`: Noise (isolierte Konflikte)
- `cluster = 0, 1, 2, ...`: Cluster-IDs (ähnliche Konflikte)

**Interpretation der Cluster:**
- Cluster 0: z.B. "kurze, gering-tödliche Konflikte in Südostasien"
- Cluster 1: z.B. "lange, hochgradig tödliche Konflikte im Naher Osten"
- Noise (-1): z.B. "völlig isolierte, anomale Konflikte"

---

### 5️⃣ Marktreaktion pro Cluster analysieren

**Die Kernfrage:**

```python
market_analysis = df.groupby("cluster").agg({
    "market_reaction": ["mean", "std", "count"],
    "volatility_change": ["mean", "std"],
    "event_return": ["mean", "std"],
    "total_deaths": ["mean", "median"]
}).round(4)
```

**Was wir lernen:**
- Reagiert der Markt anders je nach Konflikt-Cluster?
- Konflikte mit ähnlichen Mustern → ähnliche Marktreaktion?
- Oder ist der Markt-Impact zufällig/unabhängig?

**Hypothesis:**
- Cluster mit hohen Todeszahlen + langer Dauer → stärkere Marktreaktion?
- Lokalisierte Konflikte → geringere Marktreaktion?

---

### 6️⃣ Statistische Validierung

**Frage:** Sind die Unterschiede zwischen Clustern **statistisch signifikant** oder nur Zufall?

```python
from scipy.stats import kruskal

# Kruskal-Wallis Test (non-parametrisch)
h_statistic, p_value = kruskal(
    *[
        df[df.cluster == c]["market_reaction"]
        for c in df.cluster.unique()
        if c != -1
    ]
)

print(f"H-Statistik: {h_statistic}")
print(f"p-Wert: {p_value}")
print(f"Signifikant? {p_value < 0.05}")
```

**Interpretation:**
- **p < 0.05:** Die Unterschiede sind **nicht zufällig** → echte Muster!
- **p ≥ 0.05:** Könnte Zufall sein → vorsichtig interpretieren

---

### 7️⃣ Visualisierungen

#### Geografische Cluster-Karte
```python
import seaborn as sns
sns.scatterplot(x="longitude", y="latitude", hue="cluster", data=df)
plt.title("Konflikt-Cluster (geografisch)")
plt.show()
```
→ Sehen wir geografisch zusammenhängende Cluster? (z.B. Naher Osten?)

#### Marktreaktion pro Cluster (Boxplot)
```python
sns.boxplot(x="cluster", y="market_reaction", data=df)
plt.title("Durchschnittliche Marktreaktion je Cluster")
plt.show()
```
→ Cluster mit höherer Volatilität sichtbar?

#### Cluster-Größe
```python
df["cluster"].value_counts().plot(kind="bar")
plt.title("Anzahl Konflikte pro Cluster")
plt.show()
```
→ Manche Cluster viel größer als andere?

---

### 8️⃣ Qualitätsmetriken

**Silhouette Score** (nur für Cluster, nicht Noise):

```python
from sklearn.metrics import silhouette_score

clusters_only = df[df["cluster"] != -1]
if len(clusters_only) > 0:
    score = silhouette_score(
        X_scaled[df["cluster"] != -1],
        df.loc[df["cluster"] != -1, "cluster"]
    )
    print(f"Silhouette Score: {score:.3f}")
```

- **-1 bis 1** Score
- **> 0.5:** Gute Cluster
- **< 0.3:** Schwache Cluster

---

## Checkliste

- [ ] Feature Engineering (Log-Transform, One-Hot)
- [ ] Standardisierung mit StandardScaler
- [ ] k-distance Plot zur ε-Bestimmung
- [ ] DBSCAN mit gewähltem ε
- [ ] Cluster-Verteilung prüfen (keine reinen Noise-Resultate)
- [ ] Marktreaktion pro Cluster analysieren
- [ ] Visualisierungen erstellen
- [ ] Statistischer Test durchführen
- [ ] Ergebnisse dokumentieren
