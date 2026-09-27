"""
Early Purchase-Intent Prediction Analysis
Evaluates predictive performance at prefix milestones: 20%, 40%, 60%, 80%, and 100% of sessions.
Uses the true test distribution from yoochoose_balanced_train.npz.
"""

from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import tensorflow as tf
import xgboost as xgb

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "yoochoose_balanced_train.npz"
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    "interaction_count",
    "unique_item_count",
    "repeated_item_count",
    "first_item_id",
    "last_item_id",
    "mean_item_id",
    "std_item_id",
    "min_item_id",
    "max_item_id",
    "total_time_delta",
    "mean_time_delta",
    "max_time_delta",
    "std_time_delta",
    "nonzero_time_intervals",
]


# =============================================================================
# FEATURE EXTRACTION HELPERS
# =============================================================================

def extract_features(sequences, time_deltas):
    """Dynamically extracts the 14 tabular features from sequential arrays."""
    n_sessions = sequences.shape[0]
    features = np.zeros((n_sessions, 14), dtype=np.float32)

    for i in range(n_sessions):
        items = sequences[i]
        valid_items = items[items != 0]

        if len(valid_items) == 0:
            features[i] = 0
            continue

        deltas = time_deltas[i]
        valid_deltas = deltas[deltas > 0]

        total_time = np.sum(valid_deltas) if len(valid_deltas) > 0 else 0.0
        mean_time = np.mean(valid_deltas) if len(valid_deltas) > 0 else 0.0
        max_time = np.max(valid_deltas) if len(valid_deltas) > 0 else 0.0
        std_time = np.std(valid_deltas) if len(valid_deltas) > 0 else 0.0

        features[i] = [
            len(valid_items),
            len(np.unique(valid_items)),
            len(valid_items) - len(np.unique(valid_items)),
            valid_items[0],
            valid_items[-1],
            np.mean(valid_items),
            np.std(valid_items),
            np.min(valid_items),
            np.max(valid_items),
            total_time,
            mean_time,
            max_time,
            std_time,
            len(valid_deltas),
        ]

    return features


def truncate_sequences(sequences, time_deltas, fraction):
    """Truncates interaction events to simulate early in-session prediction."""
    truncated_seq = np.zeros_like(sequences)
    truncated_deltas = np.zeros_like(time_deltas)

    for i in range(len(sequences)):
        items = sequences[i]
        valid_indices = np.where(items != 0)[0]
        n_valid = len(valid_indices)

        if n_valid == 0:
            continue

        keep_count = max(1, int(np.ceil(n_valid * fraction)))
        kept_indices = valid_indices[:keep_count]

        # Right-aligned padding to maintain temporal order
        truncated_seq[i, -keep_count:] = items[kept_indices]
        truncated_deltas[i, -keep_count:] = time_deltas[i, kept_indices]

    return truncated_seq, truncated_deltas


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 60)
    print("EARLY PURCHASE-INTENT PREDICTION (BALANCED PIPELINE)")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Required dataset not found: {DATA_PATH}")

    print(f"\nLoading dataset from: {DATA_PATH}")
    data = np.load(DATA_PATH, allow_pickle=True)
    test_seq = data["test_sequences"]
    test_deltas = data["test_time_deltas"]
    y_test = data["test_labels"]

    print(f"Test sequences : {test_seq.shape}")
    print(f"Test labels    : {y_test.shape}")

    # Load All 5 Models
    print("\nLoading trained model artifacts...")
    xgb_model = xgb.Booster()
    xgb_model.load_model(str(MODELS_DIR / "machine_learning" / "xgboost_model.json"))

    lgb_model = lgb.Booster(
        model_file=str(MODELS_DIR / "machine_learning" / "lightgbm_model.txt")
    )

    lstm_model = tf.keras.models.load_model(
        MODELS_DIR / "deep_learning" / "lstm_model.keras", compile=False
    )
    gru_model = tf.keras.models.load_model(
        MODELS_DIR / "deep_learning" / "gru_model.keras", compile=False
    )
    attention_model = tf.keras.models.load_model(
        MODELS_DIR / "deep_learning" / "gru_attention_model.keras",
        compile=False,
    )
    print("All 5 models loaded successfully.")

    fractions = [0.20, 0.40, 0.60, 0.80, 1.00]
    results = []

    for frac in fractions:
        pct = int(frac * 100)
        n_events = int(round(frac * 20))
        print(f"\n{'-' * 60}")
        print(f"Using first {pct}% ({n_events}/20 events)")
        print(f"{'-' * 60}")

        sub_seq, sub_deltas = truncate_sequences(test_seq, test_deltas, frac)
        sub_tabular = extract_features(sub_seq, sub_deltas)

        # 1. LSTM
        t0 = time.perf_counter()
        prob_lstm = lstm_model.predict(sub_seq, verbose=0).flatten()
        lat_lstm = time.perf_counter() - t0
        pred_lstm = (prob_lstm >= 0.50).astype(int)

        results.append({
            "model": "LSTM",
            "session_percentage": pct,
            "precision": float(precision_score(y_test, pred_lstm, zero_division=0)),
            "recall": float(recall_score(y_test, pred_lstm, zero_division=0)),
            "f1_score": float(f1_score(y_test, pred_lstm, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob_lstm)),
            "prediction_time_seconds": lat_lstm,
        })
        print(f"LSTM            Precision={results[-1]['precision']:.4f} Recall={results[-1]['recall']:.4f} F1={results[-1]['f1_score']:.4f} ROC-AUC={results[-1]['roc_auc']:.4f}")

        # 2. GRU
        t0 = time.perf_counter()
        prob_gru = gru_model.predict(sub_seq, verbose=0).flatten()
        lat_gru = time.perf_counter() - t0
        pred_gru = (prob_gru >= 0.50).astype(int)

        results.append({
            "model": "GRU",
            "session_percentage": pct,
            "precision": float(precision_score(y_test, pred_gru, zero_division=0)),
            "recall": float(recall_score(y_test, pred_gru, zero_division=0)),
            "f1_score": float(f1_score(y_test, pred_gru, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob_gru)),
            "prediction_time_seconds": lat_gru,
        })
        print(f"GRU             Precision={results[-1]['precision']:.4f} Recall={results[-1]['recall']:.4f} F1={results[-1]['f1_score']:.4f} ROC-AUC={results[-1]['roc_auc']:.4f}")

        # 3. GRU + Attention
        t0 = time.perf_counter()
        prob_att = attention_model.predict(sub_seq, verbose=0).flatten()
        lat_att = time.perf_counter() - t0
        pred_att = (prob_att >= 0.50).astype(int)

        results.append({
            "model": "GRU + Attention",
            "session_percentage": pct,
            "precision": float(precision_score(y_test, pred_att, zero_division=0)),
            "recall": float(recall_score(y_test, pred_att, zero_division=0)),
            "f1_score": float(f1_score(y_test, pred_att, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob_att)),
            "prediction_time_seconds": lat_att,
        })
        print(f"GRU + Attention Precision={results[-1]['precision']:.4f} Recall={results[-1]['recall']:.4f} F1={results[-1]['f1_score']:.4f} ROC-AUC={results[-1]['roc_auc']:.4f}")

        # 4. XGBoost
        t0 = time.perf_counter()
        dmatrix = xgb.DMatrix(sub_tabular, feature_names=FEATURE_NAMES)
        prob_xgb = xgb_model.predict(dmatrix)
        lat_xgb = time.perf_counter() - t0
        pred_xgb = (prob_xgb >= 0.64).astype(int)

        results.append({
            "model": "XGBoost",
            "session_percentage": pct,
            "precision": float(precision_score(y_test, pred_xgb, zero_division=0)),
            "recall": float(recall_score(y_test, pred_xgb, zero_division=0)),
            "f1_score": float(f1_score(y_test, pred_xgb, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob_xgb)),
            "prediction_time_seconds": lat_xgb,
        })
        print(f"XGBoost         Precision={results[-1]['precision']:.4f} Recall={results[-1]['recall']:.4f} F1={results[-1]['f1_score']:.4f} ROC-AUC={results[-1]['roc_auc']:.4f}")

        # 5. LightGBM
        t0 = time.perf_counter()
        prob_lgb = lgb_model.predict(sub_tabular)
        lat_lgb = time.perf_counter() - t0
        pred_lgb = (prob_lgb >= 0.71).astype(int)

        results.append({
            "model": "LightGBM",
            "session_percentage": pct,
            "precision": float(precision_score(y_test, pred_lgb, zero_division=0)),
            "recall": float(recall_score(y_test, pred_lgb, zero_division=0)),
            "f1_score": float(f1_score(y_test, pred_lgb, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob_lgb)),
            "prediction_time_seconds": lat_lgb,
        })
        print(f"LightGBM        Precision={results[-1]['precision']:.4f} Recall={results[-1]['recall']:.4f} F1={results[-1]['f1_score']:.4f} ROC-AUC={results[-1]['roc_auc']:.4f}")

    df_results = pd.DataFrame(results)
    out_csv = METRICS_DIR / "early_prediction_metrics.csv"
    df_results.to_csv(out_csv, index=False)

    print(f"\n[SAVED] {out_csv}")
    print("=" * 60)
    print("EARLY PREDICTION EXPERIMENT COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()