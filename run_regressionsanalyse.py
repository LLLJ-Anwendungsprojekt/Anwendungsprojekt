"""
Master-Skript für Regressions-Analysen auf conflict_market_features.csv
"""

import sys
from pathlib import Path
import pandas as pd

# Füge src zum Python-Pfad hinzu
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lineare_regression_analyse import LinereRegression
from erweiterte_regression_analyse import ErweiterteRegression

def main():
    # Pfade
    data_path = Path(__file__).parent / 'data' / 'processed' / 'conflict_market_features.csv'
    results_linear = Path(__file__).parent / 'results' / 'results_lineare_regression'
    
    print(f"\n{'='*70}")
    print("REGRESSIONS-ANALYSE: conflict_market_features.csv")
    print(f"{'='*70}\n")
    
    # Prüfe ob Datei existiert
    if not data_path.exists():
        print(f"❌ Datei nicht gefunden: {data_path}")
        return
    
    # Prüfe Dateigröße
    file_size_mb = data_path.stat().st_size / (1024 * 1024)
    print(f"Datei: {data_path.name}")
    print(f"Größe: {file_size_mb:.1f} MB\n")
    
    # ====== ANALYSE 1: LINEARE REGRESSION ======
    print(f"{'='*70}")
    print("ANALYSE 1: LINEARE REGRESSION")
    print(f"{'='*70}\n")
    
    try:
        lin_reg = LinereRegression(str(data_path))
        lin_reg.load_data()
        lin_reg.explore_data()
        
        X, y = lin_reg.prepare_conflict_market_data()
        lin_reg.linear_regression_model(X, y)
        lin_reg.plot_regression_results(output_dirs=[str(results_linear)])
        
        print("\n✓ Lineare Regression erfolgreich abgeschlossen\n")
    except Exception as e:
        print(f"\n❌ Fehler bei Linearer Regression: {e}\n")
        import traceback
        traceback.print_exc()
    
    # ====== ANALYSE 2: ERWEITERTE REGRESSION ======
    print(f"\n{'='*70}")
    print("ANALYSE 2: ERWEITERTE REGRESSION")
    print(f"{'='*70}\n")
    
    try:
        erw_reg = ErweiterteRegression(str(data_path))
        erw_reg.load_and_prepare()
        
        # Führe alle Analysen durch
        erw_reg.analyse_boersenindex_spezifisch()
        erw_reg.analyse_volatilitaet()
        erw_reg.analyse_zeitliche_trends()
        erw_reg.analyse_konflikttyp()
        
        # Generiere Visualisierungen
        erw_reg.generate_visualizations(output_dirs=[str(results_linear)])
        
        print("\n✓ Erweiterte Regression erfolgreich abgeschlossen\n")
    except Exception as e:
        print(f"\n❌ Fehler bei Erweiterter Regression: {e}\n")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("ANALYSE ABGESCHLOSSEN")
    print(f"{'='*70}")
    print(f"Ergebnisse gespeichert in: {results_linear}\n")

if __name__ == '__main__':
    main()
