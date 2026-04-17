"""
KNN Klassifikation - Beispiel Pipeline
=======================================
Demonstriert wie man die KNN-Analyse verwendet

Verwendung:
    python knn_example.py
    
Oder einzelne Funktionen in Python importieren:
    from knn_analyse import KNNAnalyzer
"""

import sys
import pandas as pd
from pathlib import Path
from knn_analyse import KNNAnalyzer
from utils import (
    load_config, create_target_variable, 
    handle_missing_values, remove_outliers_iqr,
    select_features, save_dataset
)


def example_full_pipeline():
    """
    Beispiel 1: Vollständige Pipeline mit Data Preprocessing
    ========================================================
    Voraussetzung: Du hast bereits ein merged Dataset mit:
    - Konflikt-Features (von UCDP Daten)
    - Stock-Preis Daten (von Kaggle)
    """
    
    print("="*70)
    print("BEISPIEL 1: Vollständige KNN Pipeline")
    print("="*70)
    
    # 1. Raw Data laden (Beispiel)
    raw_data_path = 'data/raw/merged_conflicts_stocks.csv'
    
    print(f"\n[1/5] Lade Rohdaten von {raw_data_path}...")
    try:
        df = pd.read_csv(raw_data_path)
        print(f"  → {df.shape[0]} Zeilen, {df.shape[1]} Spalten geladen")
    except FileNotFoundError:
        print(f"  ✗ Datei nicht gefunden. Erstelle Beispiel-Datensatz...")
        df = create_dummy_data()
    
    # 2. Data Cleaning
    print("\n[2/5] Data Cleaning...")
    df = handle_missing_values(df, strategy='drop')
    df = remove_outliers_iqr(df, threshold=3.0)
    print(f"  → Nach Cleaning: {df.shape[0]} Zeilen")
    
    # 3. Target Variable erstellen
    print("\n[3/5] Erstelle Target-Variable...")
    if 'close' in df.columns or 'Close' in df.columns:
        price_col = 'close' if 'close' in df.columns else 'Close'
        df = create_target_variable(df, price_col=price_col)
    else:
        print("  ⚠ 'close' Spalte nicht gefunden - skippe Target-Erstellung")
    
    # 4. Speichere verarbeitete Daten
    processed_path = 'data/processed/knn_dataset.csv'
    save_dataset(df, processed_path)
    
    # 5. KNN Analyse
    print("\n[4/5] Starte KNN Klassifikation...")
    analyzer = KNNAnalyzer(random_state=42)
    
    results = analyzer.run_pipeline(
        data_path=processed_path,
        target_col='market_direction',
        k_range=range(1, 21)
    )
    
    print("\n[5/5] Pipeline abgeschlossen! ✓")
    print(f"Ergebnisse in: models/ und results/")


def example_with_preprocessed_data():
    """
    Beispiel 2: KNN nur mit bereits bereinigten Daten
    =================================================
    Wenn du die Daten bereits vorbereitet hast
    """
    
    print("\n" + "="*70)
    print("BEISPIEL 2: KNN mit vorbereiteten Daten")
    print("="*70)
    
    data_path = 'data/processed/merged_data.csv'  # Deine bereinigten Daten
    
    analyzer = KNNAnalyzer(random_state=42)
    
    try:
        results = analyzer.run_pipeline(
            data_path=data_path,
            target_col='market_direction',
            k_range=range(3, 16)  # Reduzierter Bereich
        )
        
        print("\n✓ KNN Analyse erfolgreich!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Fehler: {e}")
        print(f"Stelle sicher, dass die Datei unter {data_path} existiert")


def example_custom_config():
    """
    Beispiel 3: Mit Custom Hyperparametern
    ======================================
    Passe k und andere Parameter an deine Anforderungen an
    """
    
    print("\n" + "="*70)
    print("BEISPIEL 3: KNN mit Custom Hyperparametern")
    print("="*70)
    
    data_path = 'data/processed/merged_data.csv'
    
    analyzer = KNNAnalyzer(random_state=42)
    
    # Lade und bereite Daten vor
    df = analyzer.load_data(data_path)
    analyzer.prepare_data(df, target_col='market_direction', test_size=0.25)
    
    # Finde bestes k
    best_k = analyzer.find_optimal_k(k_range=range(1, 31))  # Größerer Range
    
    # Trainiere mit bestem k
    analyzer.train(k=best_k)
    
    # Evaluiere
    metrics = analyzer.evaluate()
    analyzer.print_results()
    
    # Visualisiere
    analyzer.visualize_results()
    analyzer.save_model()


def example_prediction():
    """
    Beispiel 4: Vorhersagen mit trainiertem Modell
    ==============================================
    Lade das Modell und mache neue Vorhersagen
    """
    
    print("\n" + "="*70)
    print("BEISPIEL 4: Neue Vorhersagen mit geladenem Modell")
    print("="*70)
    
    import pickle
    
    # Lade Modell und Scaler
    model_path = 'models/knn_model_20240101_120000.pkl'  # Anpassen!
    
    try:
        model = pickle.load(open(model_path, 'rb'))
        scaler_path = model_path.replace('knn_model_', 'scaler_')
        scaler = pickle.load(open(scaler_path, 'rb'))
        
        # Neue Daten vorbereiten
        new_data = pd.read_csv('data/new_conflicts_stocks.csv')
        X_new = new_data.drop(columns=['market_direction'])  # Falls Target vorhanden
        
        # Skaliere neue Daten
        X_scaled = scaler.transform(X_new)
        
        # Mache Vorhersagen
        predictions = model.predict(X_scaled)
        prob = model.predict_proba(X_scaled)
        
        # Zeige Ergebnisse
        results_df = new_data.copy()
        results_df['prediction'] = predictions
        results_df['prob_down'] = prob[:, 0]
        results_df['prob_up'] = prob[:, 1]
        
        print("\nVorhersagen:")
        print(results_df[['prediction', 'prob_down', 'prob_up']].head(10))
        
    except FileNotFoundError as e:
        print(f"✗ Modell nicht gefunden: {e}")
        print("Trainiere erst ein Modell mit beispiel_full_pipeline()")


def create_dummy_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Erstelle Dummy-Datensatz für Testing
    """
    import numpy as np
    
    print("  Erstelle Test-Datensatz...")
    
    dates = pd.date_range('2020-01-01', periods=n_samples)
    
    data = {
        'date': dates,
        'close': np.cumsum(np.random.randn(n_samples)) + 100,
        'volume': np.random.randint(1000000, 10000000, n_samples),
        'conflict_count': np.random.poisson(2, n_samples),
        'deaths': np.random.exponential(50, n_samples),
        'displaced': np.random.exponential(500, n_samples),
    }
    
    df = pd.DataFrame(data)
    print(f"  → {len(df)} Zeilen erstellt")
    
    return df


def main():
    """
    Hauptfunktion - Wähle welches Beispiel zu laufen
    """
    
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║  KNN Klassifikation - Beispiel Pipeline                       ║")
    print("║  Geopolitische Konflikte → Aktienmärkte Vorhersage            ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    
    examples = {
        '1': ('Vollständige Pipeline (Data Cleaning + KNN)', example_full_pipeline),
        '2': ('Nur KNN mit vorbereiteten Daten', example_with_preprocessed_data),
        '3': ('KNN mit Custom Hyperparametern', example_custom_config),
        '4': ('Vorhersagen mit geladenem Modell', example_prediction),
    }
    
    print("\nWähle ein Beispiel:")
    for key, (desc, _) in examples.items():
        print(f"  {key}. {desc}")
    print("\nOder drücke Enter für Beispiel 2...")
    
    choice = input("\nDeine Wahl (1-4): ").strip() or '2'
    
    if choice in examples:
        _, func = examples[choice]
        func()
    else:
        print(f"Ungültige Wahl. Starte Beispiel 2...")
        example_with_preprocessed_data()


if __name__ == '__main__':
    # Uncomment um direkt ein Beispiel zu laufen:
    # example_full_pipeline()
    # example_with_preprocessed_data()
    # example_custom_config()
    # example_prediction()
    
    # Oder interaktiv:
    main()
