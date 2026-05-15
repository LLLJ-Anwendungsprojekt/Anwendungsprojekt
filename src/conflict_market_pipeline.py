import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. Pfade & Konfiguration definieren
# ==========================================
# Passe diese Pfade an deine lokale Ordnerstruktur an
BASE_DIR = Path(__file__).parent.parent  # Gehe ein Level hoch von src/ zu Projekt-Root
INDEX_PATH = BASE_DIR / "data" / "raw" / "indexData.csv"
# Beachte: Obwohl die Datei .xls im Namen hat, ist es vom Typ her eine CSV!
GPR_PATH   = BASE_DIR / "data" / "raw" / "data_gpr_daily_recent.xls"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "model_ready_data.csv"

# ==========================================
# 2. Daten exakt nach Dateistruktur laden
# ==========================================
print("Lade Daten...")
# Index-Daten einlesen (Date-Spalte hat Format YYYY-MM-DD)
df_market = pd.read_csv(INDEX_PATH)
df_market['Date'] = pd.to_datetime(df_market['Date'])
df_market = df_market.sort_values(by=['Index', 'Date']).reset_index(drop=True)

# GPR-Daten einlesen (Excel-Format statt CSV)
df_gpr = pd.read_excel(GPR_PATH)
# Umwandlung des numerischen Tagesformats in ein echtes Datum
df_gpr['Date'] = pd.to_datetime(df_gpr['DAY'].astype(str), format='%Y%m%d')

# Wir behalten nur das Datum und die drei Kern-Indizes
df_gpr = df_gpr[['Date', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT']].copy()
df_gpr = df_gpr.sort_values(by='Date').reset_index(drop=True)

# ==========================================
# 3. Das Wochenend-Problem lösen (GPR Preprocessing)
# ==========================================
print("Berechne Wochenend-Überträge (Rolling Max)...")
# GPR ist 7 Tage die Woche, Aktienmärkte nur 5 Tage.
# Wir speichern das Maximum der letzten 3 Tage, um Wochenend-Eskalationen am Montag abzubilden.
gpr_columns = ['GPRD', 'GPRD_ACT', 'GPRD_THREAT']

for col in gpr_columns:
    df_gpr[f'{col}_3d_max'] = df_gpr[col].rolling(window=3, min_periods=1).max()

# ==========================================
# 4. Der globale Join
# ==========================================
print("Führe Join durch...")
# Da wir keine Ländercodes im GPR-Datensatz haben, mappen wir die globalen 
# Werte (Threat vs. Act) auf alle Indizes. 
cols_to_merge = ['Date', 'GPRD_3d_max', 'GPRD_ACT_3d_max', 'GPRD_THREAT_3d_max']
df_gpr_subset = df_gpr[cols_to_merge]

# Left Join über das Datum
df_final = pd.merge(df_market, df_gpr_subset, on='Date', how='left')

# Spalten leserlicher benennen
df_final = df_final.rename(columns={
    'GPRD_3d_max': 'gpr_global',
    'GPRD_ACT_3d_max': 'gpr_act',
    'GPRD_THREAT_3d_max': 'gpr_threat'
})

# ==========================================
# 5. Feature Engineering (Schocks, Lags, Vola)
# ==========================================
print("Berechne GPR-Features (Schocks, Lags und Threats vs. Acts)...")

def calculate_gpr_features(df):
    df = df.copy()
    
    # Für alle drei GPR-Kategorien berechnen wir die Schocks
    for prefix in ['gpr_global', 'gpr_act', 'gpr_threat']:
        
        # 1. Baseline: 21 Handelstage (ca. 1 Monat) gleitender Durchschnitt
        df[f'{prefix}_ma21'] = df[prefix].rolling(window=21, min_periods=5).mean()
        
        # 2. Der Schock: Heutiges Risiko minus Gewohnheit
        df[f'{prefix}_shock'] = df[prefix] - df[f'{prefix}_ma21']
        
        # 3. Nachrichten-Volatilität (Unsicherheit) - 7 Tage
        df[f'{prefix}_volatility_7d'] = df[prefix].rolling(window=7, min_periods=3).std()
        
        # 4. Zeitliche Verzögerungen (Lags) für die Modelle
        for lag in [1, 2, 3, 5]:
            df[f'{prefix}_shock_lag{lag}'] = df[f'{prefix}_shock'].shift(lag)
            
    # SPEZIAL-FEATURE: Die "Angst-Prämie"
    # Wenn Threats (Drohungen) überproportional höher sind als Acts (Taten), 
    # gerät der Markt oft stärker unter Druck, weil unklar ist, was noch kommt.
    df['gpr_fear_premium'] = df['gpr_threat'] - df['gpr_act']
            
    return df

# WICHTIG: Die Features müssen separat PRO INDEX berechnet werden, 
# damit die gleitenden Durchschnitte nicht z.B. vom GDAXI in den IXIC hineinlaufen!
df_final = df_final.groupby('Index', group_keys=False).apply(calculate_gpr_features)

# NaN-Werte entfernen (die ersten ca. 21 Handelstage pro Index, wegen des Moving Average)
# Wir droppen nur Zeilen, in denen unser weitester Lag (Lag 5) fehlt.
df_final = df_final.dropna(subset=['gpr_global_shock_lag5'])

print(f"Datenaufbereitung abgeschlossen! Form: {df_final.shape}")
print("\nNeue spannende GPR-Features für dein Machine Learning:")
print(df_final.filter(like='gpr_').columns.tolist()[:10] + ['...'])

# Speichern für die Modellierung
df_final.to_csv(OUTPUT_PATH, index=False)
print(f"\n✓ Fertige Datei gespeichert: {OUTPUT_PATH}")