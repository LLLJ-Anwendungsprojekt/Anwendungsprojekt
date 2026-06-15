"""Führt erst die Tests aus und danach die komplette Analyse-Pipeline.

Ablauf: pytest -> Datenaufbereitung -> Lineare Regression -> K-Means -> KNN.
Schlagen die Tests fehl, wird abgebrochen. Schlägt ein Analyseschritt fehl,
laufen die übrigen weiter; am Ende folgt eine Zusammenfassung.

Random Forest ist NICHT enthalten – die RF-Notebooks in src/random_forest/
werden manuell als letzter Schritt ausgeführt (siehe README).

In results/ verbleiben anschließend ausschließlich die Grafiken der vier
Verfahren; von den Skripten erzeugte Zwischendateien (CSV/TXT/...) werden
danach entfernt.

Aufruf (von der Repo-Wurzel):  python run_all.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
RESULTS = ROOT / "results"
KEEP_SUFFIXES = {".pdf", ".png"}  # in results/ erwünschte Grafikformate

# (Beschriftung, [Befehle]) – der Reihe nach abgearbeitet. K-Means ist eine
# Kette: erst das Tuning (best_kmeans_pca.pdf), dann die Boxplot-Reports.
STEPS = [
    ("Datenaufbereitung", [[PY, "src/build_dataset.py"]]),
    ("Lineare Regression", [[PY, "src/lineare_regression/event_regression.py"]]),
    ("K-Means", [[PY, "src/kmeans/kmeans_tune.py"],
                 [PY, "src/kmeans/generate_tuning_reports.py"]]),
    ("KNN", [[PY, "src/knn/knn_gpr_analysis.py"]]),
]


def run(label, commands):
    """Führt alle Befehle eines Schritts aus; True nur, wenn alle erfolgreich sind."""
    print(f"\n{'=' * 70}\n>>> {label}\n{'=' * 70}")
    return all(subprocess.run(cmd, cwd=ROOT).returncode == 0 for cmd in commands)


def clean_results():
    """Entfernt aus results/ alle Nicht-Grafik-Dateien (CSV/TXT/PKL/...)."""
    removed = 0
    for path in RESULTS.rglob("*"):
        if path.is_file() and path.suffix.lower() not in KEEP_SUFFIXES:
            path.unlink()
            removed += 1
    return removed


def main():
    # 1. Tests – Voraussetzung für den Rest
    if not run("Tests (pytest)", [[PY, "-m", "pytest", "tests/", "-q"]]):
        print("\nTests fehlgeschlagen – Pipeline wird nicht gestartet.")
        return 1

    # 2. Analyseschritte
    failed = [label for label, cmds in STEPS if not run(label, cmds)]

    # 3. results/ auf reine Grafiken reduzieren
    removed = clean_results()

    # 4. Zusammenfassung
    print(f"\n{'=' * 70}\nZusammenfassung")
    for label, _ in STEPS:
        print(f"  {'FEHLER' if label in failed else 'OK    '}  {label}")
    print(f"  {removed} Zwischendatei(en) aus results/ entfernt")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
