"""
Comprehensive 5-Model Comparative Evaluation
Evaluates LSTM, GRU, GRU+Attention, XGBoost, and LightGBM
Outputs results to results/metrics/comparative_evaluation.csv
"""

from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import tensorflow as tf
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Select dataset
if (PROCESSED_DIR / "yoochoose_temporal_split.npz").exists():
    SEQ_DATA_PATH = PROCESSED_DIR / "yoochoose_temporal_split.npz"
else:
    SEQ_DATA_PATH = PROCESSED_DIR / "yoochoose_balanced_train.npz"

TAB_DATA_PATH = PROCESSED_DIR / "yoochoose_ml_temporal.npz"


def main():
    print("=" * 70)
    print("UNIFIED 5-MODEL BENCHMARK EVALUATION")
    print("=" * 70)

    # 1. Load Sequence Data
    data_seq = np.load(SEQ_DATA_PATH, allow_pickle=True)
    test_seq = data_seq["test_sequences"]
    y_test_seq = data_seq["test_labels"]

    # Sample for DL if > 50,000 to keep evaluation runtime fast
    if len(y_test_seq) > 50000:
        np.random.seed(42)
        sample_idx = np.random.choice(len(y_test_seq), size=50000, replace=False)
        test_seq_eval = test_seq[sample_idx]
        y_test_dl = y_test_seq[sample_idx]
    else:
        test_seq_eval = test_seq
        y_test_dl = y_test_seq

    results = []

    # 2. Evaluate DL Models
    dl_configs = [
        ("LSTM", "lstm_model.keras", 0.63),
        ("GRU", "gru_model.keras", 0.60),
        ("GRU_Attention", "gru_attention_model.keras", 0.60),
    ]

    for name, fname, tau in dl_configs:
        model_path = MODELS_DIR / "deep_learning" / fname
        if model_path.exists():
            print(f"Evaluating {name}...")
            t0 = time.perf_counter()
            model = tf.keras.models.load_model(model_path, compile=False)
            probs = model.predict(test_seq_eval, verbose=0).flatten()
            latency = time.perf_counter() - t0

            preds = (probs >= tau).astype(int)
            results.append({
                "model": name,
                "threshold": tau,
                "accuracy": round(float(accuracy_score(y_test_dl, preds)), 4),
                "precision": round(float(precision_score(y_test_dl, preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test_dl, preds, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test_dl, preds, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_test_dl, probs)), 4),
                "pr_auc": round(float(average_precision_score(y_test_dl, probs)), 4),
                "latency_seconds": round(float(latency), 4),
            })

    # 3. Evaluate Tree Models
    if TAB_DATA_PATH.exists():
        data_tab = np.load(TAB_DATA_PATH, allow_pickle=True)
        X_test_tab = data_tab["X_test"]
        y_test_tab = data_tab["y_test"]

        # XGBoost
        xgb_path = MODELS_DIR / "machine_learning" / "xgboost_model.json"
        if xgb_path.exists():
            print("Evaluating XGBoost...")
            booster = xgb.Booster()
            booster.load_model(str(xgb_path))
            dtest = xgb.DMatrix(X_test_tab)
            t0 = time.perf_counter()
            probs_xgb = booster.predict(dtest)
            latency = time.perf_counter() - t0

            preds_xgb = (probs_xgb >= 0.64).astype(int)
            results.append({
                "model": "XGBoost",
                "threshold": 0.64,
                "accuracy": round(float(accuracy_score(y_test_tab, preds_xgb)), 4),
                "precision": round(float(precision_score(y_test_tab, preds_xgb, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test_tab, preds_xgb, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test_tab, preds_xgb, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_test_tab, probs_xgb)), 4),
                "pr_auc": round(float(average_precision_score(y_test_tab, probs_xgb)), 4),
                "latency_seconds": round(float(latency), 4),
            })

        # LightGBM
        lgb_path = MODELS_DIR / "machine_learning" / "lightgbm_model.txt"
        if lgb_path.exists():
            print("Evaluating LightGBM...")
            bst = lgb.Booster(model_file=str(lgb_path))
            t0 = time.perf_counter()
            probs_lgb = bst.predict(X_test_tab)
            latency = time.perf_counter() - t0

            preds_lgb = (probs_lgb >= 0.71).astype(int)
            results.append({
                "model": "LightGBM",
                "threshold": 0.71,
                "accuracy": round(float(accuracy_score(y_test_tab, preds_lgb)), 4),
                "precision": round(float(precision_score(y_test_tab, preds_lgb, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test_tab, preds_lgb, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test_tab, preds_lgb, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_test_tab, probs_lgb)), 4),
                "pr_auc": round(float(average_precision_score(y_test_tab, probs_lgb)), 4),
                "latency_seconds": round(float(latency), 4),
            })

    df = pd.DataFrame(results)
    out_file = METRICS_DIR / "comparative_evaluation.csv"
    df.to_csv(out_file, index=False)

    print("\n" + "=" * 70)
    print("FINAL 5-MODEL BENCHMARK RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\n[SAVED] {out_file}")


if __name__ == "__main__":
    main()