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

1. **Übersicht** — KPIs, GPR vs. Aktien (Dual-Axis), Streudiagramm, Spike-Verteilung
2. **Zeitreihen** — GPR-Komponenten, rebasierter Index-Vergleich, rollierende Korrelation
3. **Regionen** — Renditen-Heatmap, Korrelationsmatrix, Regionen-Vergleich
4. **Marktregimes** — K-Means (live berechnet), Zeitstrahl, PCA, Cluster-Profile
5. **Vorhersage** — KNN bidirektional: ROC, Feature-Wichtigkeit, Konfusionsmatrizen
6. **Event-Studie** — AR/CAR um GPR-Schocks, ACT vs. THREAT, Regression B
7. **Methodik** — Verfahrensübersicht und Datenpipeline

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
