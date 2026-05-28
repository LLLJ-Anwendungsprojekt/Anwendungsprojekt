import sys
from pathlib import Path
import traceback

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report, roc_curve, auc)
import joblib

BASE_DIR = Path(r"d:/Anwendungsprojekt")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = PROCESSED_DIR / "dataset_2001_2021.csv"

try:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print('Loaded final dataset shape:', df.shape)

    # Ensure sorting
    df = df.sort_values(['Index','YearMonth']).reset_index(drop=True)

    # Add GPR lags if missing
    for lag in (1,2,3):
        col = f'gpr_lag{lag}'
        if col not in df.columns:
            df[col] = df.groupby('Index')['gprd_ret'].shift(lag)

    # Add stock_vol6 if missing
    if 'stock_vol6' not in df.columns:
        df['stock_vol6'] = df.groupby('Index')['stock_ret'].transform(lambda s: s.rolling(window=6, min_periods=6).std())

    # Prepare features
    features_A = [
        'gprd_ret',
        'gpr_lag1',
        'gpr_lag2',
        'gpr_lag3',
        'gprd_act_ret',
        'gprd_threat_ret',
        'GPR_zscore',
        'GPR_spike',
        'Crisis_dummy',
        'Region_encoded',
    ]
    features_B = [
        'stock_ret',
        'stock_ret_lag1',
        'stock_ret_lag2',
        'stock_ret_lag3',
        'stock_vol6',
        'GPR_spike',
        'Crisis_dummy',
        'Region_encoded',
    ]

    # Derive targets
    if 'target_stock_down' not in df.columns:
        df['target_stock_down'] = (df['stock_ret'] < 0).astype(int)

    # Drop NA conservatively
    needed_A = features_A + ['target_stock_down']
    needed_B = features_B + ['target_gpr_up_lead1']

    df_A = df.dropna(subset=needed_A).copy()
    df_B = df.dropna(subset=needed_B).copy()

    X_A_full = df_A[features_A].copy()
    y_A_full = df_A['target_stock_down'].copy()
    X_B_full = df_B[features_B].copy()
    y_B_full = df_B['target_gpr_up_lead1'].copy()

    print('Final shapes:')
    print('Direction A X,y:', X_A_full.shape, y_A_full.shape)
    print('Direction B X,y:', X_B_full.shape, y_B_full.shape)

    rf_params = dict(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=20,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    rf_A_full = RandomForestClassifier(**rf_params)
    rf_B_full = RandomForestClassifier(**rf_params)

    print('Fitting rf_A_full...')
    rf_A_full.fit(X_A_full, y_A_full)
    print('Fitting rf_B_full...')
    rf_B_full.fit(X_B_full, y_B_full)

    def eval_and_save(model, X, y, direction_label, target_name, title_label):
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:,1] if hasattr(model, 'predict_proba') else np.zeros(len(y))
        metrics = dict(
            Sample='Full Period 2001-2021 (in-sample)',
            Direction=direction_label,
            Target=target_name,
            Accuracy=accuracy_score(y, y_pred),
            Precision=precision_score(y, y_pred, zero_division=0),
            Recall=recall_score(y, y_pred, zero_division=0),
            F1=f1_score(y, y_pred, zero_division=0),
            ROC_AUC=roc_auc_score(y, y_proba) if len(np.unique(y))>1 else np.nan,
        )
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {title_label}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_png = PLOTS_DIR / f'rf_confusion_matrix_{direction_label}_full_period.png'
        cm_pdf = PLOTS_DIR / f'rf_confusion_matrix_{direction_label}_full_period.pdf'
        plt.savefig(cm_png, bbox_inches='tight')
        plt.savefig(cm_pdf, bbox_inches='tight')
        plt.close()

        # ROC
        try:
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(6,5))
            plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
            plt.plot([0,1],[0,1],'--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC-Kurve: {title_label}')
            plt.legend(loc='lower right')
            roc_png = PLOTS_DIR / f'rf_roc_curve_{direction_label}_full_period.png'
            roc_pdf = PLOTS_DIR / f'rf_roc_curve_{direction_label}_full_period.pdf'
            plt.savefig(roc_png, bbox_inches='tight')
            plt.savefig(roc_pdf, bbox_inches='tight')
            plt.close()
        except Exception as exc:
            print('ROC curve could not be computed:', exc)

        creport = classification_report(y, y_pred, zero_division=0)

        fi_path = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            fi = pd.DataFrame({'feature': X.columns, 'importance': importances})
            fi = fi.sort_values('importance', ascending=False)
            fi_path = TABLES_DIR / f'rf_feature_importance_{direction_label}_full_period.csv'
            fi.to_csv(fi_path, index=False)
            plt.figure(figsize=(8,6))
            sns.barplot(data=fi, x='importance', y='feature', palette='viridis')
            plt.title(f'Feature Importance: {title_label}')
            fig_png = PLOTS_DIR / f'rf_feature_importance_{direction_label}_full_period.png'
            fig_pdf = PLOTS_DIR / f'rf_feature_importance_{direction_label}_full_period.pdf'
            plt.tight_layout()
            plt.savefig(fig_png, bbox_inches='tight')
            plt.savefig(fig_pdf, bbox_inches='tight')
            plt.close()

        model_path = MODELS_DIR / f'random_forest_{direction_label}_full_period.pkl'
        joblib.dump(model, model_path)
        return metrics, creport

    metrics_A, report_A = eval_and_save(rf_A_full, X_A_full, y_A_full, 'direction_A', 'target_stock_down', 'GPR → Aktien, 2001–2021')
    metrics_B, report_B = eval_and_save(rf_B_full, X_B_full, y_B_full, 'direction_B', 'target_gpr_up_lead1', 'Aktien → GPR, 2001–2021')

    metrics_df = pd.DataFrame([metrics_A, metrics_B])
    metrics_out_path = TABLES_DIR / 'rf_metrics_full_period_2001_2021.csv'
    metrics_df.to_csv(metrics_out_path, index=False)
    print('Saved full-period metrics to', metrics_out_path)

    print('\nClassification report Direction A:\n')
    print(report_A)
    print('\nClassification report Direction B:\n')
    print(report_B)

    # Print top features
    try:
        fiA = pd.read_csv(TABLES_DIR / 'rf_feature_importance_direction_A_full_period.csv')
        fiB = pd.read_csv(TABLES_DIR / 'rf_feature_importance_direction_B_full_period.csv')
        print('\nTop features Direction A:')
        print(fiA.head(10).to_string(index=False))
        print('\nTop features Direction B:')
        print(fiB.head(10).to_string(index=False))
    except Exception:
        pass

    print('\nDone successfully.')

except Exception as e:
    print('Error during execution:')
    traceback.print_exc()
    sys.exit(1)
