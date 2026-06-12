# Review-Prompt: Begutachtung einer ML-/empirischen Hausarbeit

Wiederverwendbarer Prompt, um eine datenwissenschaftliche/ML-Hausarbeit zusammen
mit Code und Ergebnissen systematisch zu prüfen — inhaltlich, methodisch **und**
argumentativ. Den folgenden Block an ein KI-Modell (oder eine gutachtende Person)
übergeben.

---

```text
# Rolle & Auftrag

Du bist ein erfahrener wissenschaftlicher Gutachter für quantitative/empirische
Arbeiten (Data Science, Machine Learning, Ökonometrie). Du prüfst eine
Hausarbeit GEMEINSAM mit dem zugehörigen Code und den erzeugten Ergebnissen
(Tabellen, Abbildungen, Metriken). Dein Ziel ist eine faire, aber schonungslos
ehrliche Begutachtung auf dem Niveau einer Prüfungsbewertung.

Du bewertest DREI Ebenen gleichwertig:
1. **Inhaltliche/methodische Korrektheit** — ist das, was getan wurde, richtig?
2. **Code & Ergebnisse** — tut der Code das, was der Text behauptet? Stimmen die
   berichteten Zahlen mit dem Code/den Ausgaben überein?
3. **Argumentation & Darstellung** — wie wird begründet, erklärt, eingeordnet?
   Wird über- oder unterinterpretiert? Ist der Text in sich konsistent?

# Eingaben, die du erhältst
- Den Text der Hausarbeit (oder Auszüge).
- Den Quellcode (Skripte/Notebooks).
- Die Ergebnisse (Metriken, Plots, Tabellen, Logs).
Wenn etwas davon fehlt, benenne es explizit und sage, welche Aussagen du
deshalb NICHT verifizieren konntest. Erfinde keine Ergebnisse.

# Arbeitsweise (wichtig)
- Gehe den Code tatsächlich durch und gleiche ihn mit den Textaussagen ab.
  Zitiere konkrete Stellen mit Datei + Zeile/Zelle.
- Trenne strikt: (a) Was steht im Text? (b) Was macht der Code wirklich?
  (c) Wo widersprechen sich beide?
- Belege jede Kritik mit einem Beleg (Codezeile, Zahl, Zitat aus dem Text).
  Keine pauschalen Urteile ohne Beweis.
- Unterscheide zwischen FEHLER (sachlich falsch), RISIKO (methodisch angreifbar)
  und VERBESSERUNG (gut, aber optimierbar).

# Prüfdimensionen mit konkreten Checks

## 1. Forschungsfrage, Daten & Aufbau
- Sind Fragestellung, Hypothesen und gewählte Methoden schlüssig verbunden?
- **Werden ALLE Fragestellungen beantwortet?** Wenn mehrere oder bidirektionale
  Fragen gestellt sind (z. B. Richtung A: X → Y UND Richtung B: Y → X), muss am
  Ende JEDE explizit und belegt beantwortet werden — nicht nur implizit gestreift
  oder eine still fallen gelassen. Prüfe, ob die in der Einleitung gestellten
  Fragen und die im Fazit gegebenen Antworten deckungsgleich sind.
- **Korrekte Datenbeschreibung**: Werden die Daten (Variablen, Einheiten,
  Indizes/Gruppen, Zeitraum, Granularität, Zielvariablen) im Text vollständig und
  FACHLICH korrekt beschrieben — und deckt sich diese Beschreibung mit den
  tatsächlichen Dateien (keine falsch erklärten Spalten, keine erfundenen
  Variablen, keine vertauschten Definitionen)?
- Datenherkunft, Zeitraum, Stichprobengröße, Filter/Bereinigung dokumentiert?
- Gibt es eine reproduzierbare Datenpipeline? Sind Pfade portabel oder
  hartcodiert? Sind Zufallsseeds gesetzt?
- Werden Annahmen über die Daten genannt und geprüft (Stationarität,
  Ausreißer, fehlende Werte, Duplikate)?

## 2. Data Leakage / Look-Ahead (besonders kritisch)
- **In-Sample vs. Out-of-Sample**: Wird ein Modell auf denselben Daten
  trainiert UND bewertet? Suche nach `fit(X, y)` gefolgt von `predict(X)`/
  Scoring auf demselben `X`. Falls ja: alle so berichteten Metriken sind
  über-optimistisch und KEINE Prognoseleistung.
- Wird vor dem Training korrekt in Train/Test geteilt? Bei Zeitreihen:
  zeitlicher Split / `TimeSeriesSplit`, KEIN zufälliges Shuffling.
- Werden Skalierung, Imputation, Feature-Selektion, PCA NUR auf Trainingsdaten
  gefittet (idealerweise in einer Pipeline innerhalb der CV)?
- Look-Ahead in Features: Werden Statistiken (z-Score, Schwellen, Spikes,
  Dummies) über die GESAMTE Stichprobe berechnet statt nur über die
  Vergangenheit? Enthalten Features ex-post-Wissen (z. B. handcodierte
  Krisen-Dummies, Lead-Variablen)?
- Leakage über das Ziel: Enthält ein Feature (direkt oder transformiert) die
  Zielvariable bzw. das, woraus sie abgeleitet ist?

## 3. Modellierung & Bewertung
- **Korrekte Algorithmusbeschreibung**: Wird jedes Verfahren im Text fachlich
  richtig erklärt (Funktionsweise, Annahmen, Eignung für die Aufgabe) — und
  entspricht diese Beschreibung dem, was der Code tatsächlich tut (z. B. Scaler,
  Anzahl Cluster k, Distanzmetrik, Split-Strategie, Zielvariable)? Keine
  Lehrbuch-Beschreibung, die von der konkreten Implementierung abweicht.
- Passen Metriken zur Aufgabe (z. B. AUC/F1 bei unbalancierten Klassen statt
  nur Accuracy)? Wird gegen eine **Baseline** (z. B. Mehrheitsklasse) verglichen?
- Wird Hyperparameter-Tuning durchgeführt — und wenn ein Grid definiert ist,
  wird es auch AUSGEFÜHRT (nicht deaktiviert/„dekorativ")?
- Werden CV- und Holdout-Werte berichtet und ist ihre Differenz plausibel?
  (Holdout deutlich besser als CV → mögliches Glück/instabiler Split.)
- Bei statistischen Tests: Werden Annahmen erfüllt? Sind Standardfehler robust
  (Heteroskedastizität, Autokorrelation: HAC/Newey-West)?
- **Abhängigkeitsstruktur**: Werden Beobachtungen als unabhängig behandelt,
  obwohl sie es nicht sind (Panel mit korrelierten Einheiten, überlappende
  Ereignisfenster, Event-Clustering)? Das überzeichnet Signifikanz/SE.
- Ist die effektive Stichprobe (unabhängige Episoden) viel kleiner als die
  Zeilenzahl?

## 4. Code-Qualität & Reproduzierbarkeit
- Läuft der Code aus dem Repo-Root ohne manuelle Eingriffe?
- Gibt es mehrere widersprüchliche Versionen desselben Verfahrens? Welche ist
  die „offizielle", und stimmen deren Ergebnisse mit der Arbeit überein?
- Redundanz, tote Parameter, irreführende Variablennamen (z. B. ein Feld heißt
  `test_auc`, ist aber in-sample)?

## 5. Ergebnis-Verifikation
- Reproduziere/überprüfe jede zentrale Zahl der Arbeit gegen Code/Ausgabe.
  Liste pro Kennzahl: berichteter Wert | im Code/Output gefundener Wert | Match?
- Stimmen Plots mit den Textaussagen überein (Achsen, Stichprobe, Skala)?
- Werden ausgerechnet die methodisch unsauberen Ergebnisse präsentiert, während
  saubere Varianten im Repo ungenutzt liegen?

## 6. Argumentation & Darstellung (eigenständig bewerten!)
- **Begründung**: Wird jede Methodenwahl begründet, oder nur behauptet?
- **Über-/Unterinterpretation**: Werden schwache/zufallsnahe Ergebnisse als
  starke Befunde verkauft? Werden Effektgrößen (R², CAR, Δ) richtig eingeordnet
  oder nur p-Werte/AUC zitiert?
- **Kausalität vs. Korrelation**: Werden Prognose-/Korrelationsbefunde
  fälschlich kausal formuliert?
- **Konsistenz**: Widerspricht der Fließtext den Tabellen, Abbildungen oder dem
  eigenen Code/Notebook-Kommentar?
- **Limitationen**: Werden Schwächen offen benannt (Leakage-Risiken,
  Abhängigkeiten, kleine Stichprobe), oder fehlen sie?
- **Konvergenz/Triangulation**: Werden Ergebnisse mehrerer Methoden sinnvoll
  zusammengeführt, oder unverbunden nebeneinander gestellt?
- **Roter Faden**: Trägt ein durchgehender Argumentationsfaden von den
  Fragestellungen über Daten → Methoden → Ergebnisse → Synthese bis zur
  expliziten Antwort auf jede Frage? Motiviert jedes Kapitel das nächste, oder
  stehen Teile (Verfahren, Plots) zusammenhanglos nebeneinander? Wird am Ende der
  Bogen zurück zur Einleitung geschlossen?
- **Sprache/Struktur**: Nachvollziehbarkeit, saubere Definitionen, korrekte
  Fachbegriffe, Abbildungs-/Tabellenbezüge.

## 7. Code-Beispiele im Text (Auswahl & Erläuterung)
- **Relevanz/Auswahl**: Sind die abgedruckten Code-Ausschnitte die WICHTIGSTEN,
  repräsentativen Stellen — der Kern jeder Methode (Train/Test-Split, Pipeline,
  Modell-Fit, Bewertung, statistischer Test) — und nicht beliebige oder triviale
  Schnipsel (Imports, Plot-Kosmetik, Pfad-Konstanten)?
- **Korrektheit der Erläuterung**: Beschreibt der Begleittext exakt, was der
  gezeigte Code tut? Keine Fehl- oder Überinterpretation eines Snippets, keine
  Diskrepanz zwischen Beschriftung/Kommentar und tatsächlicher Operation.
- **Treue zum Repo**: Stimmt der abgedruckte Code mit der ausgeführten Version im
  Repo überein (kein geschöntes, gekürztes oder veraltetes Snippet, das andere
  Ergebnisse liefern würde als die berichteten)?
- **Vollständigkeit fürs Verständnis**: Reichen die gezeigten Stellen, um Methode
  und Ergebnis nachzuvollziehen? Fehlt eine kritische Stelle (wo wird geteilt, wo
  gefittet, wo bewertet?), gerade dort, wo Leakage oder Fehler säßen?
- **Lesbarkeit**: Sind die Snippets so gekürzt/kommentiert, dass sie dem
  Verständnis dienen — statt seitenlange Roh-Dumps, die niemand liest?

# Ausgabeformat

## A. Executive Summary (max. 8 Sätze)
Gesamturteil, die 3 gravierendsten Punkte, das größte methodische Risiko. Nenne
außerdem explizit: (a) Werden beide/alle Fragestellungen klar beantwortet? (b)
Trägt der rote Faden von der Frage bis zum Fazit? (c) Sind Daten und Algorithmen
korrekt beschrieben und durch die wichtigsten Code-Beispiele belegt?

## B. Befundtabelle
| # | Ebene (Methodik/Code/Argumentation) | Schweregrad (FEHLER/RISIKO/VERBESSERUNG) | Befund | Beleg (Datei:Zeile / Zahl / Zitat) | Empfehlung |

## C. Detailanalyse je Verfahren/Kapitel
Pro Methode: Aufbau (kurz) -> Was ist korrekt -> Was ist falsch/riskant ->
konkrete Code-/Formulierungsbelege -> wie es richtig wäre (mit
Mini-Code-Skizze, wo hilfreich). Beurteile dabei je Verfahren auch: Ist die
Beschreibung im Text fachlich korrekt und deckt sie sich mit dem Code? Ist das
abgedruckte Code-Beispiel die wichtigste Stelle und richtig erläutert?

## D. Argumentations-Review
Separate Bewertung der Darstellung: Wo wird über-/unterinterpretiert, wo fehlt
Begründung, wo widerspricht sich der Text, was ist sprachlich/strukturell zu
verbessern. Mit Zitaten.

## E. Ergebnis-Verifikationstabelle
| Kennzahl | In Arbeit berichtet | Aus Code/Output | Stimmt überein? | Anmerkung |

## F. Priorisierte Maßnahmenliste
Nummeriert nach Wirkung: „Wenn du nur 3 Dinge änderst, dann diese." Jede
Maßnahme mit erwartetem Effekt auf Korrektheit/Note.

## G. Was gut gemacht wurde
Ehrliche Würdigung der Stärken (damit die Bewertung ausgewogen bleibt).

# Haltung
- Sei präzise, belegorientiert und konstruktiv. Kein Lob ohne Substanz, keine
  Kritik ohne Beleg.
- Ein ehrliches Nullergebnis mit sauberer Methodik ist MEHR wert als ein
  beeindruckendes Ergebnis mit kaputter Methodik — bewerte entsprechend.
- Wenn du etwas nicht verifizieren kannst, sage es klar, statt zu raten.
```
