"""
Anonymous User Purchase-Intent Prediction
Evaluates models strictly on current-session interactions with zero persistent user identifiers.
Guarantees out-of-vocabulary tokens are mapped to UNK (index 1) to prevent embedding lookup errors.
"""

from pathlib import Path
import time
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "yoochoose_temporal_split.npz"
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


def extract_features(sequences, time_deltas):
    n_sessions = sequences.shape[0]
    features = np.zeros((n_sessions, 14), dtype=np.float32)

    for i in range(n_sessions):
        items = sequences[i]
        valid_items = items[items != 0]

        if len(valid_items) == 0:
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


def load_lightgbm_model():
    joblib_p = MODELS_DIR / "machine_learning" / "lightgbm.joblib"
    txt_p = MODELS_DIR / "machine_learning" / "lightgbm_model.txt"
    if joblib_p.exists():
        return joblib.load(joblib_p)
    return lgb.Booster(model_file=str(txt_p))


def main():
    print("=" * 60)
    print("ANONYMOUS USER PURCHASE-INTENT PREDICTION")
    print("=" * 60)

    # 1. Load Data
    data = np.load(DATA_PATH, allow_pickle=True)
    test_seq = data["test_sequences"]
    test_deltas = data["test_time_deltas"]

    # Sample 15,000 anonymous test sessions
    n_samples = min(15000, len(test_seq))
    np.random.seed(42)
    sample_indices = np.random.choice(len(test_seq), size=n_samples, replace=False)

    anon_seq = test_seq[sample_indices]
    anon_deltas = test_deltas[sample_indices]

    print(f"Loaded anonymous sessions: {n_samples:,}")

    # 2. Load Models
    print("Loading models...")
    lstm_model = tf.keras.models.load_model(
        MODELS_DIR / "deep_learning" / "lstm_model.keras", compile=False
    )
    gru_model = tf.keras.models.load_model(
        MODELS_DIR / "deep_learning" / "gru_model.keras", compile=False
    )
    attention_model = tf.keras.models.load_model(
        MODELS_DIR / "deep_learning" / "gru_attention_model.keras", compile=False
    )

    xgb_model = xgb.Booster()
    xgb_model.load_model(str(MODELS_DIR / "machine_learning" / "xgboost_model.json"))
    lgb_model = load_lightgbm_model()

    # 3. Guard against OOV IDs for Keras Embeddings (Vocabulary size = 47,682)
    vocab_size = 47682
    anon_seq_safe = np.copy(anon_seq)
    oov_mask = (anon_seq_safe >= vocab_size) | (anon_seq_safe < 0)
    anon_seq_safe[oov_mask] = 1  # Map to UNK token

    # 4. Extract Tabular Features
    t0_feat = time.perf_counter()
    anon_tabular = extract_features(anon_seq_safe, anon_deltas)
    feat_time = time.perf_counter() - t0_feat

    results = []

    # LSTM
    t0 = time.perf_counter()
    lstm_probs = lstm_model.predict(anon_seq_safe, batch_size=256, verbose=0).flatten()
    lstm_time = time.perf_counter() - t0
    lstm_preds = (lstm_probs >= 0.63).astype(int)
    results.append({
        "model": "LSTM",
        "anonymous_sessions": n_samples,
        "predicted_purchase": int(np.sum(lstm_preds == 1)),
        "predicted_no_purchase": int(np.sum(lstm_preds == 0)),
        "mean_purchase_probability": float(np.mean(lstm_probs)),
        "predicted_purchase_rate": float(np.mean(lstm_preds)),
        "prediction_time_seconds": float(lstm_time),
    })

    # GRU
    t0 = time.perf_counter()
    gru_probs = gru_model.predict(anon_seq_safe, batch_size=256, verbose=0).flatten()
    gru_time = time.perf_counter() - t0
    gru_preds = (gru_probs >= 0.60).astype(int)
    results.append({
        "model": "GRU",
        "anonymous_sessions": n_samples,
        "predicted_purchase": int(np.sum(gru_preds == 1)),
        "predicted_no_purchase": int(np.sum(gru_preds == 0)),
        "mean_purchase_probability": float(np.mean(gru_probs)),
        "predicted_purchase_rate": float(np.mean(gru_preds)),
        "prediction_time_seconds": float(gru_time),
    })

    # GRU + Attention
    t0 = time.perf_counter()
    att_probs = attention_model.predict(anon_seq_safe, batch_size=256, verbose=0).flatten()
    att_time = time.perf_counter() - t0
    att_preds = (att_probs >= 0.60).astype(int)
    results.append({
        "model": "GRU + Attention",
        "anonymous_sessions": n_samples,
        "predicted_purchase": int(np.sum(att_preds == 1)),
        "predicted_no_purchase": int(np.sum(att_preds == 0)),
        "mean_purchase_probability": float(np.mean(att_probs)),
        "predicted_purchase_rate": float(np.mean(att_preds)),
        "prediction_time_seconds": float(att_time),
    })

    # XGBoost
    dmatrix = xgb.DMatrix(anon_tabular, feature_names=FEATURE_NAMES)
    t0 = time.perf_counter()
    xgb_probs = xgb_model.predict(dmatrix)
    xgb_time = time.perf_counter() - t0
    xgb_preds = (xgb_probs >= 0.71).astype(int)
    results.append({
        "model": "XGBoost",
        "anonymous_sessions": n_samples,
        "predicted_purchase": int(np.sum(xgb_preds == 1)),
        "predicted_no_purchase": int(np.sum(xgb_preds == 0)),
        "mean_purchase_probability": float(np.mean(xgb_probs)),
        "predicted_purchase_rate": float(np.mean(xgb_preds)),
        "prediction_time_seconds": float(xgb_time),
    })

    # LightGBM
    t0 = time.perf_counter()
    lgb_probs = lgb_model.predict(anon_tabular)
    lgb_time = time.perf_counter() - t0
    lgb_preds = (lgb_probs >= 0.72).astype(int)
    results.append({
        "model": "LightGBM",
        "anonymous_sessions": n_samples,
        "predicted_purchase": int(np.sum(lgb_preds == 1)),
        "predicted_no_purchase": int(np.sum(lgb_preds == 0)),
        "mean_purchase_probability": float(np.mean(lgb_probs)),
        "predicted_purchase_rate": float(np.mean(lgb_preds)),
        "prediction_time_seconds": float(lgb_time),
    })

    df = pd.DataFrame(results)
    out_csv = METRICS_DIR / "anonymous_user_prediction.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print("ANONYMOUS USER INTENT SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n[SAVED] {out_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()