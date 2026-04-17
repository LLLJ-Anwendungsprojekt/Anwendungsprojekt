"""
Utility-Funktionen für KNN Klassifikation
==========================================
Helpers für Data Loading, Preprocessing und Persistence
"""

import os
import pickle
import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def load_config(config_path: str = 'configs/knn_config.yaml') -> Dict[str, Any]:
    """
    Lade YAML Konfiguration
    
    Args:
        config_path: Pfad zu YAML-Datei
        
    Returns:
        Config Dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Konfiguration geladen: {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Config nicht gefunden: {config_path}")
        return {}


def merge_conflict_and_stock_data(conflict_file: str, stock_file: str,
                                   merge_on: str = 'date',
                                   target_window: int = 1) -> pd.DataFrame:
    """
    Merge UCDP Conflict Daten mit Stock Exchange Daten
    
    Args:
        conflict_file: Path zu Konflikt-CSV (UCDP)
        stock_file: Path zu Stock-CSV (Kaggle)
        merge_on: Spalte zum Mergen (meist 'date')
        target_window: Fenster in Tagen zur Konflikt-Vorhersage
        
    Returns:
        Merged DataFrame mit Features und Target
    """
    logger.info("Merge Konflikt- und Stock-Daten...")
    
    try:
        conflicts = pd.read_csv(conflict_file)
        stocks = pd.read_csv(stock_file)
        
        # Konvertiere zu datetime falls nötig
        if merge_on in conflicts.columns:
            conflicts[merge_on] = pd.to_datetime(conflicts[merge_on], errors='coerce')
        if merge_on in stocks.columns:
            stocks[merge_on] = pd.to_datetime(stocks[merge_on], errors='coerce')
        
        # Merge on date
        merged = pd.merge(stocks, conflicts, on=merge_on, how='left')
        
        # Fülle Konflikt-Spalten mit 0 für Tage ohne Konflikte
        conflict_cols = [col for col in merged.columns if col not in stocks.columns]
        for col in conflict_cols:
            merged[col].fillna(0, inplace=True)
        
        logger.info(f"Merged Shape: {merged.shape}")
        return merged
        
    except Exception as e:
        logger.error(f"Fehler beim Mergen: {e}")
        raise


def create_target_variable(df: pd.DataFrame, price_col: str = 'close',
                          target_col_name: str = 'market_direction') -> pd.DataFrame:
    """
    Erstelle binäre Target-Variable für Klassifikation
    
    Args:
        df: Input DataFrame
        price_col: Name der Preis-Spalte
        target_col_name: Name der neuen Target-Spalte
        
    Returns:
        DataFrame mit Target-Spalte (1 = Preis steigt, 0 = Preis fällt)
    """
    logger.info("Erstelle Target-Variable (market_direction)...")
    
    if price_col not in df.columns:
        logger.error(f"Spalte '{price_col}' nicht gefunden")
        raise ValueError(f"Spalte '{price_col}' nicht vorhanden")
    
    # Berechne tägliche Rendite
    df['daily_return'] = df[price_col].pct_change()
    
    # Binäre Variable: 1 wenn Preis steigt, 0 wenn fällt
    df[target_col_name] = (df['daily_return'] > 0).astype(int)
    
    logger.info(f"Target-Verteilung:\n{df[target_col_name].value_counts()}")
    
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop',
                         threshold: float = 0.5) -> pd.DataFrame:
    """
    Handle Missing Values (NaN)
    
    Args:
        df: Input DataFrame
        strategy: 'drop' oder 'forward_fill' oder 'mean'
        threshold: Anteil von NaN zum Droppen einer Spalte
        
    Returns:
        DataFrame ohne oder mit gehandhabten NaN
    """
    logger.info(f"Handle NaN-Werte (strategy: {strategy})...")
    
    initial_nulls = df.isnull().sum().sum()
    
    # Entferne Spalten mit zu vielen NaN
    null_ratio = df.isnull().sum() / len(df)
    cols_to_drop = null_ratio[null_ratio > threshold].index
    df = df.drop(columns=cols_to_drop)
    
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'forward_fill':
        df = df.fillna(method='ffill').dropna()
    elif strategy == 'mean':
        df = df.fillna(df.mean())
    
    final_nulls = df.isnull().sum().sum()
    logger.info(f"NaN-Werte: {initial_nulls} → {final_nulls}")
    
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list = None,
                       threshold: float = 3.0) -> pd.DataFrame:
    """
    Entferne Ausreißer mit IQR-Methode
    
    Args:
        df: Input DataFrame
        columns: Spalten für Ausreißer-Detection (None = alle numerisch)
        threshold: IQR-Multiplier (3.0 = extreme Outliers)
        
    Returns:
        DataFrame ohne Ausreißer
    """
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    logger.info(f"Entferne Ausreißer (threshold: {threshold})...")
    initial_rows = len(df)
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    removed = initial_rows - len(df)
    logger.info(f"Zeilen entfernt: {removed}")
    
    return df


def select_features(df: pd.DataFrame, exclude_cols: list = None,
                   feature_importance: Dict = None) -> Tuple[pd.DataFrame, list]:
    """
    Wähle relevante Features für Modell
    
    Args:
        df: Input DataFrame
        exclude_cols: Spalten zum Ausschließen
        feature_importance: Dict mit Feature-Wichtigkeit für Selektion
        
    Returns:
        (DataFrame mit Features, NamenkListe der Features)
    """
    if exclude_cols is None:
        exclude_cols = ['date', 'datetime', 'index', 'target', 'market_direction']
    
    # Nur numerische Spalten
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    features = [col for col in numeric_cols if col not in exclude_cols]
    
    logger.info(f"Wähle {len(features)} Features aus {len(numeric_cols)}")
    
    return df[features], features


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Speichere verarbeiteten Datensatz
    
    Args:
        df: DataFrame zum Speichern
        output_path: Output-Pfad (CSV)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Datensatz gespeichert: {output_path}")


def load_model_artifacts(model_path: str) -> Tuple[Any, Any, Dict]:
    """
    Lade trainiertes Modell, Scaler und Ergebnisse
    
    Args:
        model_path: Pfad zum Modell (.pkl)
        
    Returns:
        (model, scaler, results)
    """
    model_dir = os.path.dirname(model_path)
    base_name = os.path.basename(model_path).replace('knn_model_', '').replace('.pkl', '')
    
    scaler_path = os.path.join(model_dir, f'scaler_{base_name}.pkl')
    results_path = os.path.join(model_dir, f'results_{base_name}.pkl')
    
    model = pickle.load(open(model_path, 'rb'))
    scaler = pickle.load(open(scaler_path, 'rb'))
    results = pickle.load(open(results_path, 'rb'))
    
    logger.info(f"Modell-Artefakte geladen von {model_dir}")
    
    return model, scaler, results


def make_predictions(model, scaler, X_new: pd.DataFrame) -> np.ndarray:
    """
    Mache Vorhersagen mit geladenem Modell
    
    Args:
        model: Trainiertes KNN-Modell
        scaler: Fitted Scaler
        X_new: Neue Features (DataFrame)
        
    Returns:
        Vorhersagen (Array)
    """
    X_scaled = scaler.transform(X_new)
    predictions = model.predict(X_scaled)
    
    return predictions
