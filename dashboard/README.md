# GPR & Aktienmärkte — HTML-Dashboard

Statisches, interaktives BI-Dashboard zur Analyse des Zusammenhangs zwischen
**Geopolitischem Risiko (GPR)** und **12 globalen Aktienindizes** (2000–2021).

## Öffnen

Doppelklick auf **`index.html`** — läuft in jedem modernen Browser, **ohne Server
und ohne Internetverbindung** (Plotly.js ist lokal eingebunden).

## Inhalt des Ordners

| Datei | Zweck |
|---|---|
| `index.html` | Layout + Sidebar-Navigation (öffnen) |
| `styles.css` | Design-System (minimalistisch) |
| `app.js` | Rendering aller Diagramme mit Plotly.js |
| `data.js` | Vorberechnete Analyse-Ergebnisse (`window.DASHBOARD_DATA`) |
| `plotly.min.js` | Plotly.js v2.30 (lokal, für Offline-Betrieb) |
| `build_dashboard.py` | Erzeugt `data.js` aus den CSV-Dateien |

## Abschnitte

Die vier Analyseverfahren werden gleichberechtigt in je einem Tab dargestellt,
gerahmt von Übersicht und Synthese (6 Tabs):

1. **Übersicht** — KPIs, GPR vs. Aktien (Dual-Axis), Streudiagramm, Spike-Verteilung
2. **K-Means** — Regime-Zeitstrahl, PCA-Projektion, Cluster-Profile, Cluster-Statistiken
3. **KNN** — Kennzahlen-Tabelle, ROC, Feature-Wichtigkeit (MI), Konfusionsmatrizen (out-of-sample)
4. **Random Forest** — Kennzahlen-Tabelle, ROC, Feature-Wichtigkeit (Gini), Konfusionsmatrizen (in-sample)
5. **Event-Studie** — AR/CAR um GPR-Schocks, ACT vs. THREAT
6. **Synthese** — Richtungsevidenz aller Verfahren, AUC-Vergleich, Verdict, Regime-Kontext

> KNN und Random Forest haben bewusst eine **parallele Struktur** (gleiche
> Kennzahlen, ROC, Feature-Wichtigkeit, Konfusion), damit der Kontrast
> *out-of-sample* (KNN) ↔ *in-sample* (RF) direkt ablesbar ist.
>
> Hinweis: `build_dashboard.py` berechnet weiterhin **alle** Analysewerte (auch
> regionale Heatmaps, rollierende Korrelation etc.) und legt sie in `data.js` ab.
> Nicht jede Größe wird gerendert — zusätzliche Grafiken lassen sich ohne erneuten
> Build wieder einblenden.

## Daten neu berechnen

Wenn sich die zugrunde liegenden CSV-Dateien ändern, `data.js` neu erzeugen
(aus dem **Projektverzeichnis** ausführen, da relative Pfade verwendet werden):

```bash
python dashboard/build_dashboard.py
```

Das Skript liest `data/processed/stocks_gpr_features.csv` und
`stocks_gpr_daily.csv`, berechnet **alle Werte live** (Korrelationen, K-Means,
KNN-GridSearch, Event-Studie) und überschreibt `data.js`. Laufzeit ca. 1–2 Min.

### Methodische Hinweise

- **K-Means**: StandardScaler → KMeans, optimales *k* via Silhouette-Score (2–5),
  Cluster nach mittlerer Rendite sortiert. Identisches Feature-Set wie die Projekt-Skripte.
- **KNN**: echte Pipeline `StandardScaler → SelectKBest(MI) → PCA(0.95) → KNN`
  mit `TimeSeriesSplit` und temporalem 20 %-Holdout. Reduziertes GridSearch-Raster
  (schnell), daher können die AUC-Werte minimal vom vollen Lauf in
  `src/knn/knn_gpr_analysis.py` abweichen.
- **Event-Studie**: exakte Replikation von `src/lineare_regression/event_regression.py`
  (95.-Perzentil-Schock, ±5-Tage-Fenster, 25-Tage-Schätzfenster).
