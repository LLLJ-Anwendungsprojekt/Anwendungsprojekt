#!/usr/bin/env python
"""
EINSTIEGSPUNKT: Konflikt-Einfluss auf Börsenvolatilität & Trends

Ausführung:
    python run_analyse.py
"""

from pathlib import Path
import sys

# Importiere die Analyse-Klasse
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from konflikte_volatilitaet_trends_analyse import KonfliktVolatilitaetsTrendsAnalyse


def main():
    """Hauptfunktion"""
    data_path = Path(__file__).parent / 'data' / 'processed' / 'conflict_market_features.csv'
    
    if not data_path.exists():
        print(f"FEHLER: Datendatei nicht gefunden: {data_path}")
        return 1
    
    # Führe Analyse durch
    analysis = KonfliktVolatilitaetsTrendsAnalyse(str(data_path))
    analysis.run()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
