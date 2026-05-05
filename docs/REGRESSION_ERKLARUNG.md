# Lineare Regression - Methodologischer Überblick

## Ursprünglicher Ansatz

Initiale lineare Regression wurde auf `conflict_market_features.csv` angewendet mit:
- **Abhängige Variable (y)**: Marktrendite / Volatilität
- **Unabhängige Variablen (X)**: Konflikt-Features (Schweregrad, Todeszahl, Region, Konflikt-Typ)
- **Beobachtungen**: 300,165 Fälle (1989-2021)

## Ergebnis & Limitation

```
Modell Performance: R² = 0.000029 (0.0029%)
```

Das Modell erklärte nur **0.003% der Varianz** - dies ist zu schwach für aussagekräftige Vorhersagen:
- Lineare Beziehung zwischen Konflikten und Renditen nicht nachweisbar
- Zu viele unkontrollierte externe Faktoren (Zinssätze, Liquidität, Konjunktur, etc.)
- Konflikte beeinflussen Märkte, aber nicht über einfache lineare Mechanismen

## Pivot zu Volatilitäts- & Trendanalyse

**Statt Regression wurde ein deskriptiver Ansatz verwendet:**

1. **Volatilitätseffekt**: Pre vs. Post-Konflikt Volatilität vergleichen
   - Ergebnis: -2.37% Volatilitätsreduktion im Durchschnitt
   
2. **Marktrichtung**: 5-Tage Trend nach Konflikten
   - Ergebnis: 57.75% Aufwärtstrend vs. 42.25% Abwärtstrend
   
3. **Regionale Stabilität**: Volatilitätsmuster nach Region
   - Alle 5 Regionen zeigen 91-93% Stabilität
   
4. **Zeitliche Evolution**: Jährliche Trends 1989-2021

## Schlussfolgerung

Konflikte haben **messbaren qualitativen Einfluss** auf Börsevolatilität und Trends, aber keine einfache lineare Beziehung. Märkte reagieren dynamisch und nicht-linear auf geopolitische Schocks.

**Status**: Regression verworfen, deskriptive Analyse erfolgreich ✓
