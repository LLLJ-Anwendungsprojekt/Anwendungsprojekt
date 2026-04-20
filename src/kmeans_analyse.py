"""
K-Means Clustering fuer Konflikt- und Boersendaten
==================================================
Fuehrt unsupervised Clustering auf numerischen Features aus.
Unterstuetzt CSV sowie ZIP-Dateien mit einer CSV im Archiv.
"""

import argparse
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 7)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("kmeans_analysis.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class KMeansResults:
    best_k: int
    inertia: float
    silhouette: float
    n_samples: int
    n_features: int
    feature_columns: List[str]


class KMeansAnalyzer:
    """Kapselt Datenvorbereitung, Modelltraining und Visualisierung fuer K-Means."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.model: Optional[KMeans] = None
        self.numeric_columns: List[str] = []

    def load_data(self, path: str) -> pd.DataFrame:
        """Laedt CSV oder ZIP (mit genau einer CSV) in ein DataFrame."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Datei nicht gefunden: {path}")

        lower_path = path.lower()
        if lower_path.endswith(".zip"):
            df = pd.read_csv(path, compression="zip")
        else:
            df = pd.read_csv(path)

        logger.info("Daten geladen: %s (%s Zeilen, %s Spalten)", path, df.shape[0], df.shape[1])
        return df

    def prepare_features(
        self,
        df: pd.DataFrame,
        include_columns: Optional[Sequence[str]] = None,
        exclude_columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Waehlt numerische Spalten, imputiert Missing Values und skaliert."""
        numeric_df = df.select_dtypes(include=[np.number]).copy()

        if include_columns:
            include_columns = [c for c in include_columns if c in numeric_df.columns]
            if not include_columns:
                raise ValueError("Keine der include_columns ist numerisch vorhanden.")
            numeric_df = numeric_df[list(include_columns)]

        if exclude_columns:
            to_drop = [c for c in exclude_columns if c in numeric_df.columns]
            numeric_df = numeric_df.drop(columns=to_drop)

        if numeric_df.empty:
            raise ValueError("Keine numerischen Features nach Filterung verfuegbar.")

        self.numeric_columns = list(numeric_df.columns)

        x_imputed = self.imputer.fit_transform(numeric_df)
        x_scaled = self.scaler.fit_transform(x_imputed)

        logger.info("Feature-Matrix vorbereitet: %s Samples, %s Features", x_scaled.shape[0], x_scaled.shape[1])
        return x_scaled

    def find_best_k(self, x: np.ndarray, k_min: int = 2, k_max: int = 10) -> Tuple[int, float, float]:
        """Bestimmt das beste k mittels Silhouette-Score."""
        if k_min < 2:
            raise ValueError("k_min muss mindestens 2 sein.")
        if k_max <= k_min:
            raise ValueError("k_max muss groesser als k_min sein.")

        sample_size = x.shape[0]
        k_max_allowed = min(k_max, sample_size - 1)
        if k_max_allowed <= k_min:
            raise ValueError("Zu wenige Datenpunkte fuer den gewaehlten k-Bereich.")

        best_k = k_min
        best_silhouette = -1.0
        best_inertia = float("inf")

        for k in range(k_min, k_max_allowed + 1):
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=20)
            labels = model.fit_predict(x)
            sil = silhouette_score(x, labels)
            logger.info("k=%s | silhouette=%.4f | inertia=%.2f", k, sil, model.inertia_)

            if sil > best_silhouette:
                best_k = k
                best_silhouette = sil
                best_inertia = model.inertia_

        return best_k, best_inertia, best_silhouette

    def fit(self, x: np.ndarray, n_clusters: int) -> np.ndarray:
        """Trainiert K-Means und liefert Cluster-Labels zurueck."""
        self.model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=20)
        labels = self.model.fit_predict(x)
        return labels

    def plot_clusters(self, x: np.ndarray, labels: np.ndarray, output_path: str) -> None:
        """Visualisiert Cluster in 2D via PCA."""
        pca = PCA(n_components=2, random_state=self.random_state)
        x_2d = pca.fit_transform(x)

        plot_df = pd.DataFrame({
            "pca_1": x_2d[:, 0],
            "pca_2": x_2d[:, 1],
            "cluster": labels,
        })

        plt.figure(figsize=(12, 7))
        sns.scatterplot(
            data=plot_df,
            x="pca_1",
            y="pca_2",
            hue="cluster",
            palette="tab10",
            s=20,
            alpha=0.75,
            linewidth=0,
        )
        plt.title("K-Means Cluster (PCA-Projektion)")
        plt.xlabel("PCA Komponente 1")
        plt.ylabel("PCA Komponente 2")
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=180)
        plt.close()
        logger.info("Cluster-Plot gespeichert: %s", output_path)

    def run_pipeline(
        self,
        data_path: str,
        output_dir: str = "results",
        include_columns: Optional[Sequence[str]] = None,
        exclude_columns: Optional[Sequence[str]] = None,
        k_min: int = 2,
        k_max: int = 10,
    ) -> KMeansResults:
        """Fuehrt die komplette K-Means-Pipeline aus und speichert Ergebnisse."""
        df = self.load_data(data_path)
        x = self.prepare_features(df, include_columns=include_columns, exclude_columns=exclude_columns)

        best_k, best_inertia, best_silhouette = self.find_best_k(x, k_min=k_min, k_max=k_max)
        labels = self.fit(x, n_clusters=best_k)

        result_df = df.copy()
        result_df["cluster"] = labels

        os.makedirs(output_dir, exist_ok=True)
        assignments_path = os.path.join(output_dir, "kmeans_cluster_assignments.csv")
        result_df.to_csv(assignments_path, index=False)
        logger.info("Cluster-Zuordnungen gespeichert: %s", assignments_path)

        plot_path = os.path.join(output_dir, "kmeans_clusters_pca.png")
        self.plot_clusters(x, labels, plot_path)

        summary_path = os.path.join(output_dir, "kmeans_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("K-Means Analyse Summary\n")
            f.write("=======================\n")
            f.write(f"best_k: {best_k}\n")
            f.write(f"silhouette: {best_silhouette:.4f}\n")
            f.write(f"inertia: {best_inertia:.2f}\n")
            f.write(f"samples: {x.shape[0]}\n")
            f.write(f"features: {x.shape[1]}\n")
            f.write(f"feature_columns: {', '.join(self.numeric_columns)}\n")
        logger.info("Summary gespeichert: %s", summary_path)

        return KMeansResults(
            best_k=best_k,
            inertia=best_inertia,
            silhouette=best_silhouette,
            n_samples=x.shape[0],
            n_features=x.shape[1],
            feature_columns=self.numeric_columns,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-Means Clustering auf Konflikt/Marktdaten")
    parser.add_argument(
        "--data-path",
        default="data/processed/ConfilicsIndex2010.zip",
        help="Pfad zur CSV oder ZIP-Datei",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Verzeichnis fuer Outputs",
    )
    parser.add_argument(
        "--include-columns",
        nargs="*",
        default=None,
        help="Nur diese numerischen Spalten nutzen",
    )
    parser.add_argument(
        "--exclude-columns",
        nargs="*",
        default=["id", "conflict_new_id", "dyad_new_id", "country_id", "latitude", "longitude"],
        help="Diese numerischen Spalten ausschliessen",
    )
    parser.add_argument("--k-min", type=int, default=2, help="Minimaler k-Wert")
    parser.add_argument("--k-max", type=int, default=10, help="Maximaler k-Wert")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = KMeansAnalyzer(random_state=42)
    results = analyzer.run_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        include_columns=args.include_columns,
        exclude_columns=args.exclude_columns,
        k_min=args.k_min,
        k_max=args.k_max,
    )

    print("\nK-Means abgeschlossen")
    print(f"Bestes k: {results.best_k}")
    print(f"Silhouette: {results.silhouette:.4f}")
    print(f"Inertia: {results.inertia:.2f}")
    print(f"Samples: {results.n_samples}")
    print(f"Features: {results.n_features}")


if __name__ == "__main__":
    main()
