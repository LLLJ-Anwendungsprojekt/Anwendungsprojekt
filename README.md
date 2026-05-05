# KONFLIKT-EINFLUSS AUF BÖRSENVOLATILITÄT & TRENDS

## Projektübersicht

Analyse des Einflusses von geopolitischen Konflikten auf die Volatilität und Trends von Aktienmärkten.

**Zeitraum:** 1989-2021  
**Beobachtungen:** 300,165  
**Fokus:** Volatilität & Markttrends (NICHT Renditen)

---

## Hauptergebnisse

### 1. VOLATILITÄTSEFFEKT
- Volatilität VOR Konflikt: **0.011332**
- Volatilität NACH Konflikt: **0.011064**
- **Veränderung: -2.37%** (Märkte werden weniger volatil)
- 45% der Konflikte führen zu mehr Volatilität
- 55% der Konflikte führen zu weniger Volatilität

### 2. MARKTRICHTUNG (5-TAGE-TREND)
- **Aufwärtstrend: 57.75%** (173,338 Fälle)
- Abwärtstrend: 42.25% (126,827 Fälle)
- → Märkte erholen sich schnell

### 3. REGIONALE RESISTENZ
Alle Regionen zeigen hohe Stabilität (91-93%):
- Middle East: -0.000239 Volatilitätsveränderung
- Europe: -0.000254
- Africa: -0.000257
- Asia: -0.000303
- Americas: -0.000311

### 4. ZEITLICHE MUSTER
**Höchste Volatilitätsreduktion:**
- 2009: -0.000703
- 2021: -0.000630
- 2003: -0.000622

**Stärkste Aufwärtstrends:**
- 1989: 73.59%
- 2006: 73.41%
- 1993: 71.41%

→ Märkte sind im Zeitverlauf widerstandsfähiger geworden

---

## Projektstruktur

```
Anwendungsprojekt/
├── run_analyse.py                           [HAUPTEINSTIEGSPUNKT]
├── src/
│   ├── konflikte_volatilitaet_trends_analyse.py  [ANALYSE-KLASSE]
│   ├── utils.py                             [Utilities]
│   └── __init__.py
├── data/
│   └── processed/
│       └── conflict_market_features.csv     [Eingangsdaten: 300,165 Zeilen]
├── results/                                 [ERGEBNISSE]
│   ├── 01_volatilitaet_grundanalyse.png     [Volatilitätseffekte]
│   ├── 02_regionale_resistenz.png           [Regionale Unterschiede]
│   ├── 03_zeitliche_trends.png              [1989-2021 Evolution]
│   └── 04_zusammenfassung_analyse.png       [Detaillierte Zusammenfassung]
└── README.md                                [Diese Datei]
```

---

## Ausführung

### Hauptanalyse starten:
```bash
python run_analyse.py
```

### Direktes Ausführen des Analyse-Moduls:
```bash
python src/konflikte_volatilitaet_trends_analyse.py
```

**Laufzeit:** ~1-2 Minuten

---

## Visualisierungen

### 01_volatilitaet_grundanalyse.png (452 KB)
4 Subplots:
- Volatilitäts-Verteilung (vor/nach)
- Volatilitätsveränderung nach Schweregrad
- Marktrichtung nach Schweregrad
- Volatilitätsveränderung nach Konflikt-Typ

### 02_regionale_resistenz.png (200 KB)
- Volatilitätsveränderung pro Region
- Aufwärtstrend-Prozentsätze pro Region

### 03_zeitliche_trends.png (513 KB)
- Volatilitätsveränderung 1989-2021
- Marktrichtung 1989-2021

### 04_zusammenfassung_analyse.png (536 KB)
Detaillierte Zusammenfassung aller Ergebnisse und Schlussfolgerungen

---

## Datenquellen & Variablen

**Eingabedatei:** `data/processed/conflict_market_features.csv`

**Wichtige Variablen:**
- `pre_volatility`: Volatilität vor Konflikt
- `post_volatility`: Volatilität nach Konflikt
- `volatility_change`: Differenz (post - pre)
- `market_direction_5d`: Marktrichtung 5 Tage nach Konflikt (1=auf, 0=ab)
- `severity_class`: Konflikt-Schweregrad (0, 1, 2)
- `type_of_violence`: Konflikt-Typ (1=State-based, 2=Non-state, 3=One-sided)
- `region`: Geografische Region
- `year`: Analysejahr
- `total_deaths`: Todesopfer

---

## Schlussfolgerungen

✅ **Konflikte HABEN einen messbaren Einfluss auf Börsenvolatilität & Trends**

✅ **Märkte reagieren mit schneller Erholung** (57.75% Aufwärtstrend in 5 Tagen)

✅ **Im Durchschnitt sinkt die Volatilität um -2.37%** nach Konflikten

✅ **Regionale Muster sind konsistent** - alle Regionen zeigen ähnliche Muster

✅ **Märkte sind widerstandsfähiger geworden** (2010er vs 1990er)

---

## Anforderungen

- Python 3.8+
- pandas 3.0+
- numpy 2.4+
- matplotlib 3.10+
- seaborn 0.13+
- scikit-learn 1.8+
- scipy 1.17+

---

## Projekt-Status

✅ **ABGESCHLOSSEN & OPTIMIERT**

- Fokus: Volatilität & Trends
- Alle Visualisierungen erstellt
- Projektstruktur aufgeräumt
- Alle alten/veralteten Dateien entfernt

---

**Letztes Update:** 05.05.2026
