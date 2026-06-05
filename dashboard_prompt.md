# Dashboard Prompt: GPR vs. Globale Aktienmärkte

---

## Aufgabe

Erstelle ein vollständig lauffähiges, interaktives Streamlit-Dashboard für eine wissenschaftliche Analyse des Zusammenhangs zwischen **Geopolitischem Risiko (GPR)** und **globalen Aktienmärkten**. Das Dashboard soll die Ergebnisse von vier Analysemethoden (K-Means, KNN, Event-Studie, Random Forest) visuell hochwertig und interaktiv aufbereiten. Der Stil ist **streng minimalistisch** – maximale Informationsdichte, kein visuelles Rauschen.

---

## Datenbasis

Alle Daten liegen als CSV im Projektordner. Das Dashboard liest **ausschließlich** diese Dateien (keine Ergebnisbilder):

### `data/processed/stocks_gpr_features.csv` — 3012 Zeilen, 21 Spalten
Monats-Panel (2000-07 bis 2021-05), 12 Indizes:

| Spalte | Beschreibung |
|---|---|
| `Index` | Index-Kürzel (z.B. `IXIC`, `GDAXI`) |
| `YearMonth` | Periode (Format: `"2003-04"`) |
| `Close_last` | Letzter Schlusskurs des Monats |
| `Stock_monthly_pct` | Monatliche Aktienrendite in % |
| `GPRD_mean` | Ø GPR gesamt im Monat |
| `GPRD_ACT_mean` | Ø GPR-Akte (realisierte Ereignisse) |
| `GPRD_THREAT_mean` | Ø GPR-Threats (Bedrohungen) |
| `GPRD_monthly_pct` | Monatliche % Veränderung GPR gesamt |
| `GPRD_ACT_monthly_pct` | Monatliche % Veränderung GPR-ACT |
| `GPRD_THREAT_monthly_pct` | Monatliche % Veränderung GPR-THREAT |
| `gpr_lag1/2/3` | GPR % Veränderung 1/2/3 Monate zuvor |
| `stock_lag1/2/3` | Aktienrendite 1/2/3 Monate zuvor |
| `stock_vol6` | Rollende 6-Monats-Volatilität Aktien |
| `gprd_zscore` | Z-Score GPR-Level |
| `gpr_spike` | 1 wenn |GPR-Änderung| > 1 SD (ca. 4% der Monate) |
| `stock_down` | 1 wenn Aktienrendite < 0 (Zielvariable A) |
| `gpr_up_next` | 1 wenn GPR nächsten Monat steigt (Zielvariable B) |

### `data/processed/stocks_gpr_daily.csv` — 63840 Zeilen, 8 Spalten
Tages-Panel (2000-01 bis 2021-05):
Spalten: `Index`, `Date`, `Close`, `Adj Close`, `GPRD`, `GPRD_ACT`, `GPRD_THREAT`, `YearMonth`

### Indizes und Regionen:
```python
INDICES = {
    "000001.SS": ("Shanghai SSE",   "China"),
    "399001.SZ": ("Shenzhen SZSE",  "China"),
    "GDAXI":     ("DAX",            "Europa"),
    "N100":      ("Euronext 100",   "Europa"),
    "SSMI":      ("Swiss SMI",      "Europa"),
    "GSPTSE":    ("TSX Composite",  "Nordamerika"),
    "IXIC":      ("NASDAQ",         "Nordamerika"),
    "NYA":       ("NYSE Composite", "Nordamerika"),
    "HSI":       ("Hang Seng",      "Asien-Pazifik"),
    "KS11":      ("KOSPI",          "Asien-Pazifik"),
    "N225":      ("Nikkei 225",     "Asien-Pazifik"),
    "TWII":      ("TAIEX Taiwan",   "Asien-Pazifik"),
}
```

### Wichtige Datenfakten (für Insight-Cards):
- Zeitraum: 251 Monate (2000-07 – 2021-05)
- GPR-Spike-Monate: ~4% aller Monate
- Ø Aktienrendite bei GPR-Spike: **–1.83%** vs. +0.57% normal
- Historische GPR-Peaks: Sep/Okt 2001 (9/11), Mär 2003 (Irakkrieg), Sep 2008 (Finanzkrise), Mär 2020 (COVID)
- Korrelation GPR_THREAT → Aktien (–0.082) stärker als GPR_ACT → Aktien (–0.049)

---

## K-Means Clustering (Inline berechnen)

Das Dashboard berechnet das K-Means Clustering **selbst** beim Start (dauert <5 Sekunden):

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

KMEANS_FEATURES = [
    "GPRD_mean", "GPRD_ACT_mean", "GPRD_THREAT_mean",
    "GPRD_monthly_pct", "Stock_monthly_pct",
    "gpr_lag1", "stock_lag1", "stock_vol6", "gprd_zscore"
]
# StandardScaler -> SimpleImputer(median) -> KMeans(n_clusters=3, n_init=50, random_state=42)
# Optimale k zwischen 2-5 via Silhouette-Score
```

Cluster-Labels dem DataFrame als Spalte `cluster` hinzufügen. Ergebnis cachen (`@st.cache_data`).

---

## KNN-Ergebnisse (Statische Metriken)

Die KNN-Analyse war rechenintensiv. Die finalen Metriken sind **hardcoded** als Konstanten im Code zu hinterlegen:

```python
KNN_RESULTS = {
    "A": {  # GPR-Features -> Aktien fallen?
        "label": "Richtung A: GPR → Aktienmarkt",
        "cv_auc": 0.52,    # Platzhalter – mit echten Werten aus knn_results.txt ersetzen
        "test_auc": 0.54,
        "f1": 0.48,
        "accuracy": 0.56,
        "best_k": 7,
        "best_metric": "manhattan",
        "n_pca": 4,
        "selected_features": ["gpr_momentum", "act_vs_threat", "gprd_zscore", "gpr_accel"],
    },
    "B": {  # Aktien-Features -> GPR steigt?
        "label": "Richtung B: Aktienmarkt → GPR",
        "cv_auc": 0.53,
        "test_auc": 0.55,
        "f1": 0.47,
        "accuracy": 0.57,
        "best_k": 10,
        "best_metric": "euclidean",
        "n_pca": 3,
        "selected_features": ["stock_momentum", "stock_vol6", "stock_accel", "stock_cumchange_3"],
    }
}
```
*(Hinweis: Diese Werte sind Platzhalter. Nach echtem KNN-Lauf durch tatsächliche Werte ersetzen.)*

Die ROC-Kurven werden **aus den Daten rekonstruiert** (temporal split 80/20 + einfacher KNN ohne GridSearch, nur zur Visualisierung).

---

## Dashboard-Architektur

### Aufbau: Sidebar-Navigation + Main Content

```
┌─────────────┬──────────────────────────────────────────┐
│   SIDEBAR   │              MAIN CONTENT                │
│             │                                          │
│ Navigation  │  Section Header                          │
│ ──────────  │  ─────────────────────────────────────   │
│ ○ Übersicht │  [KPI Cards Row]                         │
│ ○ Zeitreihe │                                          │
│ ○ Regionen  │  [Chart 1]    [Chart 2]                  │
│ ○ Regime    │                                          │
│ ○ Vorhers.  │  [Chart 3 Full Width]                    │
│ ○ Events    │                                          │
│ ○ Methodik  │                                          │
│             │                                          │
│  Filter:    │                                          │
│  Zeitraum   │                                          │
│  Indizes    │                                          │
└─────────────┴──────────────────────────────────────────┘
```

Die Sidebar enthält:
- Logo / Titel: "GPR & Aktienmärkte" in kleiner, schlanker Schrift
- Navigation (radio buttons, keine Icons nötig)
- **Globale Filter** (gelten für alle Abschnitte):
  - Zeitraum: Datumsschieberegler (Monatsebene)
  - Index-Auswahl: Multiselect-Checkbox (Default: alle)

---

## Abschnitt 1: Übersicht (Landing Page)

### KPI-Card-Reihe (4 Karten, 1 Zeile)
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ 251 Monate  │  │ 12 Indizes  │  │ -1.83%      │  │ -0.082      │
│ 2000–2021   │  │ 4 Regionen  │  │ Ø Rendite   │  │ Korrelation │
│             │  │             │  │ GPR-Spikes  │  │ THREAT→Mkt  │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```
- Karten: hellgrauer Hintergrund, dünner Border, Zahl groß und fett, Label klein und grau

### Dual-Axis Zeitreihe (Voller Breite, ~400px Höhe)
- X-Achse: Zeit (Monate)
- Linke Y-Achse: GPR_mean (blau, Linie)
- Rechte Y-Achse: Ø Stock_monthly_pct aller 12 Indizes (dunkelrot, Balken oder Fläche)
- **Vertikale Event-Linien** mit Annotationen:
  - "9/11" → 2001-09
  - "Irakkrieg" → 2003-03
  - "Finanzkrise" → 2008-09
  - "COVID-19" → 2020-03
- Toggle (Radio): "GPR gesamt / GPR-ACT / GPR-THREAT"
- Achsenbeschriftungen auf Deutsch

### Scatter: GPR-Änderung vs. Aktienrendite (Halbe Breite)
- X: GPRD_monthly_pct, Y: Stock_monthly_pct
- Jeder Punkt = ein Index-Monat
- Farbe: Region (4 Farben: China, Europa, Nordamerika, Asien-Pazifik)
- Regressionsgerade (OLS, kein CI) in grau gestrichelt
- Hover: Index-Name, Datum, Werte

### Verteilung GPR-Spike-Monate (Halbe Breite)
- Box-Plot oder Violin-Plot: Stock_monthly_pct, aufgeteilt in "GPR-Spike" vs. "Normal"
- Farbe: Spike = akzentrot, Normal = hellblau
- Annotation: "–1.83% vs. +0.57%"

---

## Abschnitt 2: Zeitreihen-Analyse

### GPR-Komponenten-Chart (Voller Breite)
- Drei Linien auf gleichem Chart: GPRD_mean, GPRD_ACT_mean, GPRD_THREAT_mean
- Farben: Grau (Gesamt), Dunkelblau (ACT), Hellblau (THREAT)
- Shaded Area unter jeder Linie, leichte Transparenz
- Gleiche Event-Annotationen wie Abschnitt 1

### Index-Vergleich (Voller Breite)
- Dropdown: Wähle 1-4 Indizes gleichzeitig (Default: IXIC, GDAXI, HSI, 000001.SS)
- Normalisierte Kursverläufe (Close_last, rebased auf 100 am Startpunkt)
- + GPR als dünne graue Linie auf zweiter Y-Achse
- Quelle: `stocks_gpr_daily.csv` für tägliche Granularität

### Rollierende Korrelation (Voller Breite)
- 12-Monats-Rolling-Korrelation zwischen GPRD_mean und Stock_monthly_pct
- Separate Linien je Region (China, Europa, Nordamerika, Asien-Pazifik)
- Nulllinie gestrichelt grau
- Legende oben rechts

---

## Abschnitt 3: Regionale Analyse

### Heatmap: Monatsrenditen (Voller Breite)
- X-Achse: Zeit (Monate), Y-Achse: 12 Indizes (sortiert nach Region)
- Farbe: Stock_monthly_pct → Divergierende Farbskala (Rot = negativ, Grün = positiv, Weiß = 0)
- Clipping: [-10%, +10%] für Lesbarkeit
- Gepunktete Linien zwischen Regionen
- Hover: Index, Datum, Rendite

### Korrelationsmatrix (Halbe Breite)
- Heatmap: 12×12 Korrelationsmatrix der monatlichen Aktienrenditen
- Sortiert nach Region
- Annotation der Korrelationswerte (2 Dezimalstellen)
- Farbskala: –1 bis +1 (Blau-Weiß-Rot)

### Region-Vergleich Bar Chart (Halbe Breite)
- Grouped Bar Chart: je Region
  - Ø Aktienrendite (blau)
  - Anteil Verlustmonate (rot)
  - Ø GPR-Korrelation (grau)

---

## Abschnitt 4: Marktregimes (K-Means)

*Cluster werden inline berechnet (gecacht). Hinweis: "Optimales k = X (Silhouette = Y.YY)" anzeigen.*

### Regime-Zeitstrahl (Voller Breite)
- Zeitreihe der Cluster-Zuordnungen: Farbbänder je Monat
- X = Zeit, Y = Ø Stock_monthly_pct (Linie)
- Background-Color je Periode nach Cluster-Label
- Legende: "Regime 0: Hohe GPR / schwache Märkte", etc. (automatisch aus Cluster-Mittelwerten ableiten)

### PCA-Scatter (Halbe Breite)
- 2D PCA der Clustering-Features
- Farbe: Cluster
- Hover: Index, Monat, Cluster, wichtigste Features
- Achsentitel: "PCA Komponente 1 (X% Varianz)"

### Cluster-Profile Radar/Bar (Halbe Breite)
- Für jeden Cluster: Balkendiagramm der z-standardisierten Feature-Mittelwerte
- Features: GPRD_mean, Stock_monthly_pct, stock_vol6, GPRD_ACT_mean, GPRD_THREAT_mean
- Positive Abweichung = über globalem Durchschnitt
- Farbe: Cluster-Farbe, gemeinsame Legende

### Cluster-Statistiken-Tabelle
- Index: Cluster-Nr. | Spalten: n, Ø Rendite, Ø GPR, Ø Vol, Anteil Verlustmonate, Anteil GPR-Spike
- Styled DataFrame: Werte farbig kodiert (negative Rendite rot, hohe Vol orange)

---

## Abschnitt 5: Vorhersage-Analyse (KNN)

### Ergebnis-Header
- Zwei große Metrik-Karten nebeneinander:
  - "Richtung A: GPR → Aktien" — Test-AUC, F1, Accuracy
  - "Richtung B: Aktien → GPR" — Test-AUC, F1, Accuracy
- Darunter: "Δ AUC (A–B) = +0.0X → [Interpretation]"

### ROC-Kurven (Halbe Breite)
- Beide Richtungen auf einem Plot oder nebeneinander
- AUC als Annotation in der Kurve
- Diagonale (Zufallsbaseline) gestrichelt grau

### Feature-Wichtigkeit (Halbe Breite)
- Horizontaler Bar Chart: MI-Scores
- Richtung A (GPR-Features) oben, Richtung B (Stock-Features) unten
- Selektierte Features dunkel hervorgehoben, verworfene grau
- Toggle: "Richtung A / Richtung B"

### Konfusionsmatrizen (Nebeneinander)
- 2×2 Heatmap je Richtung
- Zahlen + Prozent in den Zellen
- Farbskala je Richtung (Rot für A, Blau für B)

### Interpretation-Box
- Grauer `st.info`-Block:
  > "Die Ergebnisse zeigen eine leichte Vorhersagbarkeit in beiden Richtungen (AUC > 0.50).
  >  Richtung B (Aktien → GPR) ist geringfügig besser, konsistent mit der Hypothese,
  >  dass Aktienmärkte den GPR-Index vorauslaufen."

---

## Abschnitt 6: Event-Studie

*Alle Berechnungen aus `stocks_gpr_daily.csv` rekonstruieren.*

### Parameter-Panel (Sidebar-Erweiterung für diesen Tab)
- Radio: "A: GPR-Schock → Aktien" / "B: Aktien-Crash → GPR"
- Slider: Schock-Perzentil (Default: 95. / 5.)
- Slider: Ereignisfenster ±N Tage (Default: ±5)

### AR-Balkendiagramm (Halbe Breite)
- X: Tage relativ zum Event (–5 bis +5)
- Y: Ø Abnormal Return
- Fehlerbalken: 95%-KI
- Rote vertikale Linie bei t=0 ("Event-Tag")
- Farbe: Positive AR = grün, Negative AR = rot

### CAR-Linienchart (Halbe Breite)
- X: Tage (–5 bis +5)
- Y: Kumuliertes Ø AR
- Konfidenzband (95%-KI) als transparente Fläche
- Annotation: "CAR post-event: +X.XX% (p=0.0XX)"

### ACT vs. THREAT-Vergleich (Voller Breite)
- Zwei überlagerte CAR-Linien: GPRD_ACT-Schocks (blau) vs. GPRD_THREAT-Schocks (rot)
- Legende mit n, Ø CAR, p-Wert
- Kernbotschaft als Annotation: "THREAT-Schocks stärker als ACT-Schocks (Korr: –0.082 vs. –0.049)"

---

## Abschnitt 7: Methodik & Übersicht

### Methodenübersicht (2×2 Grid, Karten)
```
┌─────────────────────┐  ┌─────────────────────┐
│ K-Means Clustering  │  │ KNN Klassifikation  │
│ Marktregime-Erkenng │  │ Bidir. Vorhersage   │
│ k=X (Sil=Y.YY)     │  │ AUC A=X / B=Y       │
└─────────────────────┘  └─────────────────────┘
┌─────────────────────┐  ┌─────────────────────┐
│ Event-Studie        │  │ Random Forest       │
│ N Events extrahiert │  │ (Jupyter Notebooks) │
│ CAR post: X.XX%     │  │ Siehe results/      │
└─────────────────────┘  └─────────────────────┘
```

### Daten-Pipeline-Diagramm (Flowchart als Text/Mermaid)
```
Rohdaten (indexData.csv + GPR.xls)
   ↓ build_dataset.py
stocks_gpr_daily/monthly/features.csv
   ↓ ─────────────────────────────
   ├→ Event Study   → results/lineare_regression/
   ├→ K-Means       → results/k_means/
   ├→ KNN           → results/knn/
   └→ Random Forest → results/random_forest/
```

---

## Design-System

### Farben (exakt einhalten)
```python
COLORS = {
    "bg":         "#FFFFFF",      # Hintergrund
    "surface":    "#F7F7F7",      # Karten, Panels
    "border":     "#E5E5E5",      # Rahmen
    "text":       "#1A1A1A",      # Haupttext
    "text_muted": "#888888",      # Labels, Sekundärtext
    "accent":     "#1C4E80",      # Primärfarbe: Tiefblau
    "accent2":    "#A31621",      # Sekundärfarbe: Dunkelrot
    "positive":   "#2E8B57",      # Grün für positive Renditen
    "negative":   "#A31621",      # Rot für negative Renditen
    "neutral":    "#888888",      # Grau für neutrale Elemente
    # Regionen
    "china":      "#D62728",
    "europa":     "#1F77B4",
    "nordamerika":"#2CA02C",
    "asien":      "#FF7F0E",
    # K-Means Cluster (bis zu 5)
    "cluster":    ["#1C4E80","#A31621","#2E8B57","#FF7F0E","#7B2D8B"],
}
```

### Typografie
- Keine externe Font-Einbindung nötig — Streamlit-Standard reicht
- Überschriften: `st.subheader()`, kein fettes Markdown unnötig
- KPI-Zahlen: HTML in `st.markdown()`, 32px, font-weight: 700

### Chart-Stil (alle Plotly-Figures)
```python
LAYOUT_BASE = dict(
    template="plotly_white",
    font=dict(family="sans-serif", size=12, color="#1A1A1A"),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    margin=dict(l=50, r=30, t=50, b=50),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    xaxis=dict(showgrid=True, gridcolor="#F0F0F0", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#F0F0F0", zeroline=False),
)
```

- Keine Grid-Dominanz: gridcolor immer `#F0F0F0` (sehr hell)
- Achsenbeschriftungen auf **Deutsch**
- Hover-Templates sauber: `"<b>%{x}</b><br>%{y:.2f}%"`
- Keine 3D-Charts
- Keine Donut-/Kuchendiagramme

---

## Technische Anforderungen

### Stack
```
streamlit>=1.31
plotly>=5.18
pandas>=2.2
numpy>=1.26
scikit-learn>=1.4
scipy>=1.12
statsmodels>=0.14
```

### Datei-Struktur
```
dashboard.py          # Einzige Datei, alles enthalten
```
Keine Aufteilung in Untermodule — eine einzelne, vollständig lauffähige Datei.

### Starten
```bash
# Aus dem Projektverzeichnis c:\Users\lukas\Git\Anwendungsprojekt
streamlit run dashboard.py
```

### Performance
- Alle Datenladeoperationen mit `@st.cache_data` dekorieren
- K-Means Berechnung ebenfalls cachen
- Kein unnötiges Re-Rendering

### Seitenbreite
```python
st.set_page_config(
    page_title="GPR & Aktienmärkte",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

### Sidebar-Styling (Custom CSS einbinden)
```python
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #F7F7F7; }
.stMetric { background-color: #F7F7F7; padding: 1rem; border-radius: 6px; }
.stMetric label { color: #888888; font-size: 0.8rem; }
h1, h2, h3 { font-weight: 600; color: #1A1A1A; }
</style>
""", unsafe_allow_html=True)
```

### Fehlerbehandlung
- Wenn eine Datei nicht gefunden wird: `st.error(f"Datei nicht gefunden: {path}")` und `st.stop()`
- Wenn K-Means nicht konvergiert: Fallback auf k=3

---

## Ergänzende Hinweise

1. **Keine externen Datenquellen** — alles aus den lokalen CSV-Dateien
2. **Keine Bilder einbinden** (keine PDFs aus `results/` laden)
3. Das Dashboard soll **standalone** laufen — kein separater Backend-Prozess
4. Alle berechneten Werte (Korrelationen, Cluster, Event-Statistiken) werden **live aus den Daten** berechnet, nicht hardcoded (außer den KNN-Metriken)
5. Das Streamlit-Package muss vorher installiert werden: `pip install streamlit plotly`
6. Der Code soll gut strukturiert sein: Konstanten oben, dann Daten-Ladefunktionen (gecacht), dann Abschnittsfunktionen, dann `main()`
