"""
Lineare Regression Analyse - Konflikt- und Börsen-Daten
========================================================

Dieses Modul führt eine lineare Regressionsanalyse auf den Konflikt- und 
Börsen-Daten durch. Es analysiert die Beziehung zwischen Konflikten und 
Marktpreisen verschiedener Börsenindizes weltweit.

Verwendet:
- pandas für Datenmanipulation
- numpy für numerische Operationen
- scikit-learn für Modellierung
- matplotlib/seaborn für Visualisierung
- statsmodels für detaillierte Regressions-Statistiken
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Stil-Einstellungen für Visualisierungen
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class LinereRegression:
    """
    Klasse für lineare Regressionsanalyse auf Konflikt- und Börsen-Daten.
    Vergleicht den Einfluss von Konflikten auf Börsenpreise.
    """

    def __init__(self, data_path: str):
        """
        Initialisiert die Regressionsanalyse.

        Parameters
        ----------
        data_path : str
            Pfad zur CSV-Datei mit den Konflikt- und Börsen-Daten
        """
        self.data_path = data_path
        self.data = None
        self.model_results = {}
        self.scaler = StandardScaler()
        self.conflict_impact = None

    def load_data(self) -> pd.DataFrame:
        """
        Lädt die Daten aus der CSV-Datei.

        Returns
        -------
        pd.DataFrame
            Geladene Datenframe
        """
        print(f"Lade Daten von: {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        print(f"Daten geladen: {self.data.shape[0]} Zeilen, {self.data.shape[1]} Spalten")
        return self.data

    def explore_data(self) -> None:
        """
        Erkundet grundlegende Eigenschaften der Daten.
        """
        if self.data is None:
            self.load_data()

        print("\n" + "="*60)
        print("DATENEXPLORATION")
        print("="*60)
        print(f"\nDatentypen:\n{self.data.dtypes}\n")
        print(f"Fehlende Werte:\n{self.data.isnull().sum()}\n")
        print(f"Grundlegende Statistiken:\n{self.data.describe()}\n")

    def prepare_conflict_market_data(self):
        """
        Bereitet Daten für Regressionsanalyse vor.
        Vereinfachtes Modell: Nur Konflikte vs. Börsenpreis (Close)
        
        X: active_year (0=kein Konflikt, 1=Konflikt vorhanden)
        y: Close (Börsenschlusspreis)

        Returns
        -------
        tuple
            (X, y) - Prädiktoren und Zielvariate
        """
        if self.data is None:
            self.load_data()

        # Konvertiere Date in datetime
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])

        # Nur Konflikt-Indikator und Close-Preis
        X = self.data[['active_year']].copy()
        y = self.data['Close'].copy()
        
        # Entferne NaN-Werte
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"\n{'='*60}")
        print("MODELL-KONFIGURATION")
        print(f"{'='*60}")
        print(f"Prädiktor: active_year (Konflikt aktiv: 0=Nein, 1=Ja)")
        print(f"Zielvariate: Close (Börsenschlusspreis)")
        print(f"Datensatz-Größe: {len(X)} Zeilen\n")
        
        # Analysiere Konflikt-Verteilung
        conflict_yes = (X['active_year'] == 1).sum()
        conflict_no = (X['active_year'] == 0).sum()
        print(f"Konflikte vorhanden: {conflict_yes} Tage ({conflict_yes/len(X)*100:.1f}%)")
        print(f"Keine Konflikte: {conflict_no} Tage ({conflict_no/len(X)*100:.1f}%)\n")

        return X, y

    def linear_regression_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Erstellt und trainiert ein lineares Regressionsmodell.
        Vergleicht: Konflikte beeinflussen Börsenpreise?

        Parameters
        ----------
        X : pd.DataFrame
            Feature-Matrix (active_year)
        y : pd.Series
            Zielvariate (Close)
        """
        # Sklearn Modell
        model_sklearn = LinearRegression()
        model_sklearn.fit(X, y)
        y_pred = model_sklearn.predict(X)

        # Metriken
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        print("="*60)
        print("LINEARE REGRESSION: KONFLIKT-EINFLUSS AUF BÖRSENPREISE")
        print("="*60)
        print(f"\nModell-Performance:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}\n")

        # Koeffizienten interpretieren
        coefficient = model_sklearn.coef_[0]
        intercept = model_sklearn.intercept_

        print(f"Regressionsformal: Close = {intercept:.2f} + {coefficient:.4f} × Konflikt\n")
        print("INTERPRETATION:")
        print(f"  Intercept (kein Konflikt): {intercept:.2f}")
        print(f"  Koeffizient (wenn Konflikt=1): {coefficient:.4f}")
        
        if coefficient < 0:
            print(f"  → Konflikte REDUZIEREN Börsenpreise um durchschnittlich {abs(coefficient):.2f}")
        elif coefficient > 0:
            print(f"  → Konflikte ERHÖHEN Börsenpreise um durchschnittlich {coefficient:.2f}")
        else:
            print(f"  → Konflikte haben KEINEN signifikanten Einfluss")
        
        # Berechne Durchschnittspreise mit/ohne Konflikt
        conflict_data = X[X['active_year'] == 1]
        no_conflict_data = X[X['active_year'] == 0]
        
        if len(conflict_data) > 0:
            avg_price_conflict = y[conflict_data.index].mean()
        else:
            avg_price_conflict = 0
            
        if len(no_conflict_data) > 0:
            avg_price_no_conflict = y[no_conflict_data.index].mean()
        else:
            avg_price_no_conflict = 0

        print(f"\nDURCHSCHNITTSPREISE:")
        print(f"  Mit Konflikt: {avg_price_conflict:.2f}")
        print(f"  Ohne Konflikt: {avg_price_no_conflict:.2f}")
        if avg_price_no_conflict > 0:
            price_diff = avg_price_conflict - avg_price_no_conflict
            pct_change = (price_diff / avg_price_no_conflict) * 100
            print(f"  Differenz: {price_diff:.2f} ({pct_change:.2f}%)\n")

        self.model_results = {
            'model': model_sklearn,
            'X': X,
            'y': y,
            'y_pred': y_pred,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'coefficient': coefficient,
            'intercept': intercept,
            'avg_price_conflict': avg_price_conflict,
            'avg_price_no_conflict': avg_price_no_conflict
        }

        return model_sklearn

    def plot_regression_results(self, output_dirs: list = None):
        """
        Erstellt Visualisierungen der Regressionsergebnisse.

        Parameters
        ----------
        output_dirs : list
            Liste von Verzeichnissen zum Speichern der Diagramme
        """
        if not self.model_results:
            raise ValueError("Kein Modell trainiert. Führe linear_regression_model aus.")

        if output_dirs is None:
            output_dirs = ['../results/', '../results_lineare_regression/']

        # Erstelle alle Output-Verzeichnisse
        output_paths = [Path(d) for d in output_dirs]
        for path in output_paths:
            path.mkdir(parents=True, exist_ok=True)

        model = self.model_results['model']
        X = self.model_results['X']
        y = self.model_results['y']
        y_pred = self.model_results['y_pred']
        coefficient = self.model_results['coefficient']
        intercept = self.model_results['intercept']

        # ========== DIAGRAMM 1: Konflikt vs. Börsenpreis ==========
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1.1: Streudiagramm mit Regressionslinie
        ax1 = axes[0, 0]
        ax1.scatter(X['active_year'], y, alpha=0.3, s=20, label='Daten')
        x_line = np.array([0, 1])
        y_line = intercept + coefficient * x_line
        ax1.plot(x_line, y_line, 'r-', linewidth=2, label='Regressionslinie')
        ax1.set_xlabel('Konflikt (0=Nein, 1=Ja)', fontsize=11)
        ax1.set_ylabel('Börsenschlusspreis (Close)', fontsize=11)
        ax1.set_title('Konflikt-Einfluss auf Börsenpreis', fontsize=12, fontweight='bold')
        ax1.set_xticks([0, 1])
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 1.2: Box-Plot Vergleich
        ax2 = axes[0, 1]
        price_with_conflict = y[X['active_year'] == 1]
        price_no_conflict = y[X['active_year'] == 0]
        bp = ax2.boxplot([price_no_conflict, price_with_conflict], 
                         labels=['Kein Konflikt', 'Mit Konflikt'],
                         patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
            patch.set_facecolor(color)
        ax2.set_ylabel('Börsenschlusspreis (Close)', fontsize=11)
        ax2.set_title('Preisverteilung: Mit vs. Ohne Konflikt', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # 1.3: Residuen
        ax3 = axes[1, 0]
        residuals = y - y_pred
        ax3.scatter(y_pred, residuals, alpha=0.3, s=20)
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Vorhersagte Werte', fontsize=11)
        ax3.set_ylabel('Residuen', fontsize=11)
        ax3.set_title('Residuen-Diagramm', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 1.4: Residuen-Verteilung
        ax4 = axes[1, 1]
        ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax4.set_xlabel('Residuen', fontsize=11)
        ax4.set_ylabel('Häufigkeit', fontsize=11)
        ax4.set_title('Verteilung der Residuen', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        
        # Speichere in alle Output-Verzeichnisse
        for path in output_paths:
            plot_path = path / 'konflikt_vs_boersen_preis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Diagramm gespeichert: {plot_path}")
        plt.close()

        # ========== DIAGRAMM 2: Durchschnittspreis-Vergleich ==========
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Kein Konflikt', 'Mit Konflikt']
        prices = [self.model_results['avg_price_no_conflict'], 
                 self.model_results['avg_price_conflict']]
        colors = ['#2ecc71', '#e74c3c']
        
        bars = ax.bar(categories, prices, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        # Beschriftung auf den Balken
        for bar, price in zip(bars, prices):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{price:.2f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylabel('Durchschnittlicher Börsenschlusspreis', fontsize=12)
        ax.set_title('Durchschnittliche Börsenpreise: Mit vs. Ohne Konflikt', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Speichere in alle Output-Verzeichnisse
        for path in output_paths:
            plot_path = path / 'durchschnittspreis_vergleich.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Diagramm gespeichert: {plot_path}")
        plt.close()

        # ========== DIAGRAMM 3: Modell-Gleichung und Statistiken ==========
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        text_content = f"""
LINEARE REGRESSIONSANALYSE: KONFLIKT-EINFLUSS AUF BÖRSENPREISE

Modell-Gleichung:
Close = {intercept:.2f} + {coefficient:.4f} × Konflikt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODELL-METRIKEN:
  • R² Score: {self.model_results['r2']:.4f} ({self.model_results['r2']*100:.2f}% Varianz erklärt)
  • RMSE: {self.model_results['rmse']:.2f}
  • MAE: {self.model_results['mae']:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DURCHSCHNITTLICHE BÖRSENPREISE:
  • Ohne Konflikt: {self.model_results['avg_price_no_conflict']:.2f}
  • Mit Konflikt: {self.model_results['avg_price_conflict']:.2f}
  • Differenz: {self.model_results['avg_price_conflict'] - self.model_results['avg_price_no_conflict']:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTERPRETATION:
{'Konflikte SENKEN die Börsenpreise deutlich.' if coefficient < -10 else 'Konflikte SENKEN die Börsenpreise leicht.' if coefficient < 0 else 'Konflikte ERHÖHEN die Börsenpreise leicht.' if coefficient > 0 else 'Konflikte haben keinen signifikanten Einfluss.'}
        """
        
        ax.text(0.05, 0.95, text_content, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Speichere in alle Output-Verzeichnisse
        for path in output_paths:
            plot_path = path / 'modell_zusammenfassung.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Zusammenfassung gespeichert: {plot_path}")
        plt.close()
        
        print("\n✅ Alle Diagramme wurden erfolgreich gespeichert!")

    def summary_statistics(self) -> None:
        """
        Gibt detaillierte Zusammenfassung der Regressionsstatistiken aus.
        """
        if not self.model_results:
            raise ValueError("Kein Modell trainiert.")

        print("\n" + "="*60)
        print("DETAILLIERTE REGRESSIONSSTATISTIKEN")
        print("="*60)

        r2 = self.model_results['r2']
        rmse = self.model_results['rmse']
        mae = self.model_results['mae']

        print(f"R² (Bestimmtheitsmaß): {r2:.4f}")
        print(f"  → {r2*100:.2f}% der Varianz wird erklärt")
        print(f"\nRMSE (Wurzel Mittlerer Quadratischer Fehler): {rmse:.4f}")
        print(f"MAE (Mittlerer Absoluter Fehler): {mae:.4f}")
        print("\nModell-Interpretation:")
        print(f"  - Eine perfekte Anpassung: R² = 1.0")
        print(f"  - Zufällige Vorhersagen: R² = 0.0")
        print(f"  - Schlechte Anpassung: R² < 0.0")


def main():
    """
    Hauptfunktion zum Ausführen der linearen Regressionsanalyse.
    Vergleicht: Konflikte vs. Börsenpreise
    """
    # Pfad zu den Daten
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'conflict_market_features.csv'

    if not data_path.exists():
        print(f"Fehler: Datendatei nicht gefunden: {data_path}")
        return

    # Initialisiere Analyse
    regression = LinereRegression(str(data_path))
    regression.load_data()
    regression.explore_data()

    # Vorbereitung und Modellierung
    X, y = regression.prepare_conflict_market_data()
    regression.linear_regression_model(X, y)

    # Visualisierungen (in beide Verzeichnisse speichern)
    output_dirs = [
        str(Path(__file__).parent.parent / 'results' / 'results_lineare_regression'),
        str(Path(__file__).parent.parent / 'results')
    ]
    regression.plot_regression_results(output_dirs=output_dirs)
    regression.summary_statistics()

    print("\n✅ Lineare Regressionsanalyse abgeschlossen!")


if __name__ == "__main__":
    main()
