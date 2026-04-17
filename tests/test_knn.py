"""
Tests für KNN Klassifikations-Modul
===================================
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import sys
sys.path.insert(0, '../src')

from knn_analyse import KNNAnalyzer
from utils import (
    handle_missing_values, create_target_variable,
    remove_outliers_iqr
)


class TestKNNAnalyzer(unittest.TestCase):
    """Test-Suite für KNNAnalyzer Klasse"""
    
    def setUp(self):
        """Erstelle Test-Datensatz"""
        np.random.seed(42)
        self.n_samples = 100
        
        # Dummy-Daten mit Features und Target
        self.X = np.random.randn(self.n_samples, 5)
        self.y = np.random.randint(0, 2, self.n_samples)
        
        # DataFrame für Testing
        self.df = pd.DataFrame(
            self.X,
            columns=[f'feature_{i}' for i in range(5)]
        )
        self.df['target'] = self.y
    
    def test_knn_analyzer_initialization(self):
        """Test ob KNNAnalyzer richtig initialisiert wird"""
        analyzer = KNNAnalyzer(random_state=42)
        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.random_state, 42)
    
    def test_prepare_data(self):
        """Test ob Daten richtig vorbereitet werden"""
        analyzer = KNNAnalyzer()
        analyzer.prepare_data(self.df, target_col='target', test_size=0.2)
        
        # Überprüfe Shapes
        self.assertEqual(len(analyzer.X_train) + len(analyzer.X_test), self.n_samples)
        self.assertTrue(len(analyzer.X_train) > len(analyzer.X_test))
    
    def test_train_model(self):
        """Test ob Modell trainiert werden kann"""
        analyzer = KNNAnalyzer()
        analyzer.prepare_data(self.df, target_col='target')
        analyzer.train(k=5)
        
        self.assertIsNotNone(analyzer.model)
        self.assertIsInstance(analyzer.model, KNeighborsClassifier)
    
    def test_evaluation(self):
        """Test ob Evaluation Metriken berechnet werden"""
        analyzer = KNNAnalyzer()
        analyzer.prepare_data(self.df, target_col='target')
        analyzer.train(k=5)
        metrics = analyzer.evaluate()
        
        self.assertIn('accuracy', metrics)
        self.assertIn('f1', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)


class TestUtilsFunctions(unittest.TestCase):
    """Test-Suite für Utility-Funktionen"""
    
    def setUp(self):
        """Erstelle Test-DataFrame"""
        self.df = pd.DataFrame({
            'feature_1': [1.0, 2.0, np.nan, 4.0],
            'feature_2': [10.0, 20.0, 30.0, 40.0],
            'price': [100.0, 105.0, 102.0, 107.0]
        })
    
    def test_handle_missing_values_drop(self):
        """Test Handling von NaN mit drop strategy"""
        df_clean = handle_missing_values(self.df, strategy='drop')
        self.assertEqual(len(df_clean), 3)  # Eine Zeile mit NaN entfernt
    
    def test_handle_missing_values_fillna(self):
        """Test Handling von NaN mit fill strategy"""
        df_clean = handle_missing_values(self.df, strategy='mean')
        self.assertEqual(len(df_clean), 4)  # Alle Zeilen behalten
        self.assertFalse(df_clean.isnull().any().any())
    
    def test_create_target_variable(self):
        """Test ob Target-Variable richtig erstellt wird"""
        df = self.df.copy()
        df = create_target_variable(df, price_col='price')
        
        self.assertIn('market_direction', df.columns)
        self.assertTrue(df['market_direction'].isin([0, 1]).all())
    
    def test_remove_outliers(self):
        """Test Ausreißer-Entfernung"""
        # DataFrame mit Ausreißern
        df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 1000],  # 1000 ist Ausreißer
            'feature_2': [10, 20, 30, 40, 50]
        })
        
        df_clean = remove_outliers_iqr(df, threshold=3.0)
        self.assertLess(len(df_clean), len(df))


class TestDataIntegration(unittest.TestCase):
    """Integration Tests für komplette Pipeline"""
    
    def test_full_pipeline(self):
        """Test komplette KNN Pipeline"""
        # Erstelle Test-Datensatz
        np.random.seed(42)
        df = pd.DataFrame({
            f'feature_{i}': np.random.randn(200) for i in range(10)
        })
        df['conflict_count'] = np.random.poisson(2, 200)
        df['market_direction'] = np.random.randint(0, 2, 200)
        
        # Speichere temporär
        df.to_csv('test_data.csv', index=False)
        
        # Starte Pipeline
        analyzer = KNNAnalyzer(random_state=42)
        df = analyzer.load_data('test_data.csv')
        analyzer.prepare_data(df)
        analyzer.train(k=5)
        metrics = analyzer.evaluate()
        
        # Überprüfe Ergebnisse
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
        
        # Aufräumen
        import os
        os.remove('test_data.csv')


def run_tests():
    """Führe alle Tests aus"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestKNNAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilsFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestDataIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
