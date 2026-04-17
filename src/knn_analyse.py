"""
K-Nearest Neighbors (KNN) Klassifikation für Aktienmarkt-Vorhersage
=====================================================================
Vorhersage ob der Aktienmarkt bei Geopolitischen Konflikten steigt/fällt

Autor: Johannes
Modul: Anwendungsprojekt - CRISP-DM
Phase: Modeling & Evaluation
"""

import os
import sys
import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Konfiguration
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('knn_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class KNNAnalyzer:
    """
    Production-ready KNN Klassifikations-Analyzer für 
    Konflikt-Aktienmärkte Vorhersage.
    """
    
    def __init__(self, data_path: str = None, random_state: int = 42):
        """
        Initialisiere KNN Analyzer
        
        Args:
            data_path: Pfad zur bereinigten CSV-Datei
            random_state: Für Reproduzierbarkeit
        """
        self.random_state = random_state
        self.data_path = data_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.model = None
        self.results = {}
        logger.info("KNN Analyzer initialisiert")
    
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Lädt die bereinigten Daten
        
        Args:
            filepath: Pfad zur Datei (CSV)
            
        Returns:
            DataFrame mit Daten
        """
        if filepath is None:
            filepath = self.data_path
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Daten geladen: {filepath} ({df.shape[0]} Zeilen, {df.shape[1]} Spalten)")
            return df
        except FileNotFoundError:
            logger.error(f"Datei nicht gefunden: {filepath}")
            raise
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'market_direction',
                     test_size: float = 0.2) -> None:
        """
        Teile Daten in Train/Test und standardisiere
        
        Args:
            df: DataFrame mit Features und Target
            target_col: Name der Zielkolumne ('up'/'down' oder 1/0)
            test_size: Anteil der Testdaten
        """
        logger.info("Starte Datenvorbereitung...")
        
        # Validierung
        if target_col not in df.columns:
            logger.error(f"Target kolumne '{target_col}' nicht gefunden")
            raise ValueError(f"Kolumne '{target_col}' nicht vorhanden")
        
        # Entferne Rows mit Missing Values
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Zeilen nach NaN-Entfernung: {initial_rows} → {len(df)}")
        
        # Trennung Features und Target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Konvertiere Target zu numerisch wenn nötig
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            logger.info(f"Target-Klassen: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Train-Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y
        )
        
        # Standardisierung (WICHTIG für KNN!)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info(f"Train-Test Split: {len(self.X_train)} Train, {len(self.X_test)} Test")
        logger.info(f"Features: {X.shape[1]}")
        logger.info(f"Klassenverteilung: {np.bincount(self.y_train)}")
    
    def find_optimal_k(self, k_range: range = range(1, 21)) -> int:
        """
        Finde optimale k mit GridSearchCV und Cross-Validation
        
        Args:
            k_range: Bereich von k-Werten zu testen
            
        Returns:
            Beste k
        """
        logger.info(f"Starte Hyperparameter-Tuning für k in {k_range}...")
        
        param_grid = {'n_neighbors': list(k_range)}
        
        # StratifiedKFold für unbalancierte Daten
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            KNeighborsClassifier(metric='euclidean'),
            param_grid,
            cv=kfold,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        best_k = grid_search.best_params_['n_neighbors']
        best_score = grid_search.best_score_
        
        logger.info(f"Beste k: {best_k} (F1-Score: {best_score:.4f})")
        
        # Speichere CV-Ergebnisse
        self.results['cv_scores'] = grid_search.cv_results_
        
        return best_k
    
    def train(self, k: int = 5) -> None:
        """
        Trainiere KNN Modell
        
        Args:
            k: Anzahl der Nachbarn
        """
        logger.info(f"Trainiere KNN Modell mit k={k}...")
        
        self.model = KNeighborsClassifier(
            n_neighbors=k,
            metric='euclidean',
            weights='distance',  # Gewichte Nachbarn nach Distanz
            algorithm='auto'
        )
        
        self.model.fit(self.X_train, self.y_train)
        logger.info("Modell-Training abgeschlossen")
    
    def evaluate(self) -> dict:
        """
        Evaluiere Modell auf Test-Daten
        
        Returns:
            Dictionary mit allen Metriken
        """
        logger.info("Starte Modell-Evaluation...")
        
        # Vorhersagen
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1] if len(np.unique(self.y_test)) == 2 else None
        
        # Metriken
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
        
        # Cross-Validation Score
        cv_scores = cross_val_score(
            self.model, self.X_train, self.y_train,
            cv=5, scoring='f1_weighted'
        )
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        self.results['metrics'] = metrics
        self.results['y_test'] = self.y_test
        self.results['y_pred'] = y_pred
        self.results['y_pred_proba'] = y_pred_proba
        self.results['confusion_matrix'] = confusion_matrix(self.y_test, y_pred)
        
        logger.info("Evaluation abgeschlossen")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1-Score: {metrics['f1']:.4f}")
        logger.info(f"CV-Score: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        
        return metrics
    
    def print_results(self) -> None:
        """Drucke detaillierte Ergebnisse"""
        logger.info("\n" + "="*60)
        logger.info("FINALE ERGEBNISSE")
        logger.info("="*60)
        
        metrics = self.results['metrics']
        print(f"\nMetriken:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  CV-Score:  {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        
        print(f"\nKlassifikationsreport:")
        print(classification_report(self.results['y_test'], self.results['y_pred']))
    
    def visualize_results(self, output_dir: str = 'results') -> None:
        """
        Erstelle Visualisierungen für wissenschaftliche Arbeit
        
        Args:
            output_dir: Verzeichnis für PNG-Export
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Erstelle Visualisierungen in {output_dir}/")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('KNN Klassifikations-Ergebnisse', fontsize=16, fontweight='bold')
        
        # 1. Verwirrungsmatrix Heatmap
        cm = self.results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. Metriken-Balkendiagramm
        metrics = self.results['metrics']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                        metrics['recall'], metrics['f1']]
        axes[0, 1].bar(metric_names, metric_values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].set_title('Performance Metriken')
        axes[0, 1].set_ylabel('Score')
        for i, v in enumerate(metric_values):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 3. ROC-Kurve (für binäre Klassifikation)
        if self.results['y_pred_proba'] is not None and len(np.unique(self.results['y_test'])) == 2:
            fpr, tpr, _ = roc_curve(self.results['y_test'], self.results['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            axes[1, 0].plot(fpr, tpr, color='#e74c3c', lw=2, 
                          label=f'ROC Kurve (AUC = {roc_auc:.3f})')
            axes[1, 0].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Zufälliger Klassifikator')
            axes[1, 0].set_xlim([0.0, 1.0])
            axes[1, 0].set_ylim([0.0, 1.05])
            axes[1, 0].set_xlabel('False Positive Rate')
            axes[1, 0].set_ylabel('True Positive Rate')
            axes[1, 0].set_title('ROC-Kurve')
            axes[1, 0].legend(loc="lower right")
        
        # 4. Predictions Distribution
        y_test = self.results['y_test']
        y_pred = self.results['y_pred']
        classes = np.unique(y_test)
        
        actual_counts = [np.sum(y_test == c) for c in classes]
        pred_counts = [np.sum(y_pred == c) for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        axes[1, 1].bar(x - width/2, actual_counts, width, label='Actual', color='#3498db')
        axes[1, 1].bar(x + width/2, pred_counts, width, label='Predicted', color='#e74c3c')
        axes[1, 1].set_xlabel('Klasse')
        axes[1, 1].set_ylabel('Anzahl')
        axes[1, 1].set_title('Vorhersage-Verteilung')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(['Down', 'Up'])
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/knn_results.png', dpi=300, bbox_inches='tight')
        logger.info(f"Visualisierung gespeichert: {output_dir}/knn_results.png")
        plt.close()
    
    def save_model(self, output_dir: str = 'models') -> None:
        """
        Speichere trainiertes Modell und Scaler
        
        Args:
            output_dir: Verzeichnis für Model-Speicherung
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = f'{output_dir}/knn_model_{timestamp}.pkl'
        scaler_path = f'{output_dir}/scaler_{timestamp}.pkl'
        results_path = f'{output_dir}/results_{timestamp}.pkl'
        
        pickle.dump(self.model, open(model_path, 'wb'))
        pickle.dump(self.scaler, open(scaler_path, 'wb'))
        pickle.dump(self.results, open(results_path, 'wb'))
        
        logger.info(f"Modell gespeichert: {model_path}")
        logger.info(f"Scaler gespeichert: {scaler_path}")
        logger.info(f"Ergebnisse gespeichert: {results_path}")
    
    def run_pipeline(self, data_path: str, target_col: str = 'market_direction',
                    k_range: range = range(1, 21)) -> dict:
        """
        Komplette Analytics Pipeline
        
        Args:
            data_path: Pfad zu CSV-Datei
            target_col: Name der Zielkolumne
            k_range: Range für k-Tuning
            
        Returns:
            Results Dictionary
        """
        logger.info("Starte KNN Analytics Pipeline...")
        
        # 1. Daten laden
        df = self.load_data(data_path)
        
        # 2. Daten vorbereiten
        self.prepare_data(df, target_col)
        
        # 3. Optimales k finden
        best_k = self.find_optimal_k(k_range)
        
        # 4. Modell trainieren
        self.train(k=best_k)
        
        # 5. Evaluation
        metrics = self.evaluate()
        
        # 6. Ergebnisse anzeigen
        self.print_results()
        
        # 7. Visualisierungen
        self.visualize_results()
        
        # 8. Modell speichern
        self.save_model()
        
        logger.info("Pipeline abgeschlossen!")
        return self.results


def main():
    """
    Hauptfunktion - Rufe KNN Pipeline auf.
    
    Hinweis: Passe data_path und target_col an deine Dateistruktur an!
    """
    
    # ============================================
    # KONFIGURATION - HIER ANPASSEN!
    # ============================================
    data_path = '../data/processed/merged_data.csv'  # Pfad zu bereinigtem Datensatz
    target_col = 'market_direction'  # Name der Zielspalte
    k_range = range(1, 21)  # K-Werte zum Testen
    
    try:
        # Analyzer erstellen
        analyzer = KNNAnalyzer(random_state=42)
        
        # Pipeline laufen lassen
        results = analyzer.run_pipeline(
            data_path=data_path,
            target_col=target_col,
            k_range=k_range
        )
        
        logger.info("KNN Analyse erfolgreich abgeschlossen!")
        
    except FileNotFoundError as e:
        logger.error(f"Datei nicht gefunden: {e}")
        logger.error(f"Bitte stelle sicher, dass die Datei unter {data_path} existiert")
        logger.error("Beispielformat: CSV mit Features und Target-Spalte")
    except Exception as e:
        logger.error(f"Fehler während Pipeline: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

