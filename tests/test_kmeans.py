"""Tests fuer das K-Means Modul."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from kmeans_analyse import KMeansAnalyzer, KMeansResults


class TestKMeansAnalyzer(unittest.TestCase):
    """Test-Suite fuer den KMeansAnalyzer."""

    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "f1": [1.0, 1.1, 0.9, 8.0, 8.2, 7.8],
                "f2": [1.0, 0.8, 1.2, 8.1, 7.9, 8.3],
                "drop_me": [10, 11, 12, 13, 14, 15],
                "category": ["a", "a", "a", "b", "b", "b"],
            }
        )

    def test_initialization(self):
        analyzer = KMeansAnalyzer(random_state=123)
        self.assertEqual(analyzer.random_state, 123)
        self.assertIsNone(analyzer.model)

    def test_prepare_features_include_and_exclude(self):
        analyzer = KMeansAnalyzer(random_state=42)
        x = analyzer.prepare_features(
            self.df,
            include_columns=["f1", "f2", "category"],
            exclude_columns=["f2"],
        )

        self.assertEqual(x.shape, (6, 1))
        self.assertEqual(analyzer.numeric_columns, ["f1"])

    def test_find_best_k_returns_valid_values(self):
        analyzer = KMeansAnalyzer(random_state=42)
        x = analyzer.prepare_features(self.df, include_columns=["f1", "f2"])

        best_k, inertia, silhouette = analyzer.find_best_k(x, k_min=2, k_max=4)

        self.assertIn(best_k, [2, 3, 4])
        self.assertGreater(inertia, 0)
        self.assertGreaterEqual(silhouette, -1.0)
        self.assertLessEqual(silhouette, 1.0)

    def test_run_pipeline_creates_outputs(self):
        analyzer = KMeansAnalyzer(random_state=42)

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_path = tmp_path / "data.csv"
            output_dir = tmp_path / "results"

            self.df.to_csv(data_path, index=False)

            results = analyzer.run_pipeline(
                data_path=str(data_path),
                output_dir=str(output_dir),
                include_columns=["f1", "f2"],
                output_prefix="_NEU_DB",
                k_min=2,
                k_max=4,
            )

            self.assertIsInstance(results, KMeansResults)
            self.assertTrue((output_dir / "kmeans_cluster_assignments_NEU_DB.csv").exists())
            self.assertTrue((output_dir / "kmeans_clusters_pca_NEU_DB.png").exists())
            self.assertTrue((output_dir / "kmeans_summary_NEU_DB.txt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
