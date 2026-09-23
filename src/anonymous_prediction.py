import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/yoochoose_dl_subset.npz"

LSTM_MODEL_PATH = "models/deep_learning/lstm_model.keras"
GRU_MODEL_PATH = "models/deep_learning/gru_model.keras"
ATTENTION_MODEL_PATH = "models/deep_learning/gru_attention_model.keras"

XGB_MODEL_PATH = "models/machine_learning/xgboost_model.json"
LGBM_MODEL_PATH = "models/machine_learning/lightgbm_model.txt"

OUTPUT_PATH = "results/metrics/anonymous_user_prediction.csv"

THRESHOLD = 0.5


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs("results/metrics", exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("ANONYMOUS USER PURCHASE-INTENT PREDICTION")
print("=" * 60)


# ============================================================
# LOAD CURRENT-SESSION DATA
# ============================================================

print("\nLoading current-session data...")

data = np.load(DATA_PATH, allow_pickle=True)

test_sequences = data["test_sequences"]
test_time_deltas = data["test_time_deltas"]
test_labels = data["test_labels"]

print("Test sessions       :", len(test_sequences))
print("Sequence shape      :", test_sequences.shape)
print("Time-delta shape    :", test_time_deltas.shape)

print("\nAnonymous-user assumption:")
print("- No user ID is used")
print("- No previous-session history is used")
print("- Only current-session interaction data is used")


# ============================================================
# AGGREGATED CURRENT-SESSION FEATURES
# Same 14 features used by XGBoost and LightGBM
# ============================================================

def create_features(sequences, time_deltas):

    features = []

    for seq, times in zip(sequences, time_deltas):

        seq = np.asarray(seq)
        times = np.asarray(times)

        nonzero_items = seq[seq > 0]

        if len(nonzero_items) == 0:
            nonzero_items = np.array([0])

        interaction_count = len(nonzero_items)
        unique_item_count = len(np.unique(nonzero_items))
        repeated_item_count = interaction_count - unique_item_count

        first_item_id = nonzero_items[0]
        last_item_id = nonzero_items[-1]

        mean_item_id = np.mean(nonzero_items)
        std_item_id = np.std(nonzero_items)
        min_item_id = np.min(nonzero_items)
        max_item_id = np.max(nonzero_items)

        total_time_delta = np.sum(times)
        mean_time_delta = np.mean(times)
        max_time_delta = np.max(times)
        std_time_delta = np.std(times)
        nonzero_time_intervals = np.sum(times > 0)

        features.append([
            interaction_count,
            unique_item_count,
            repeated_item_count,
            first_item_id,
            last_item_id,
            mean_item_id,
            std_item_id,
            min_item_id,
            max_item_id,
            total_time_delta,
            mean_time_delta,
            max_time_delta,
            std_time_delta,
            nonzero_time_intervals
        ])

    return np.asarray(features, dtype=np.float32)


# ============================================================
# LOAD EXISTING MODELS
# ============================================================

print("\nLoading existing trained models...")

lstm_model = tf.keras.models.load_model(
    LSTM_MODEL_PATH
)

gru_model = tf.keras.models.load_model(
    GRU_MODEL_PATH
)

attention_model = tf.keras.models.load_model(
    ATTENTION_MODEL_PATH
)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_MODEL_PATH)

lgbm_model = lgb.Booster(
    model_file=LGBM_MODEL_PATH
)

print("All models loaded successfully.")


# ============================================================
# CREATE CURRENT-SESSION FEATURES
# ============================================================

print("\nExtracting current-session features...")

start = time.perf_counter()

aggregated_features = create_features(
    test_sequences,
    test_time_deltas
)

feature_time = time.perf_counter() - start

print("Feature matrix     :", aggregated_features.shape)
print("Feature extraction :", f"{feature_time:.4f} seconds")


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "-" * 60)
print("GENERATING ANONYMOUS-USER PREDICTIONS")
print("-" * 60)


results = []


# ------------------------------------------------------------
# LSTM
# ------------------------------------------------------------

start = time.perf_counter()

lstm_probabilities = lstm_model.predict(
    test_sequences,
    verbose=0
).ravel()

lstm_time = time.perf_counter() - start

lstm_predictions = (
    lstm_probabilities >= THRESHOLD
).astype(int)

results.append([
    "LSTM",
    len(test_sequences),
    int(np.sum(lstm_predictions == 1)),
    int(np.sum(lstm_predictions == 0)),
    float(np.mean(lstm_probabilities)),
    float(np.mean(lstm_probabilities >= THRESHOLD)),
    lstm_time
])

print(
    f"LSTM            : "
    f"{np.sum(lstm_predictions == 1)} predicted Purchase, "
    f"{np.sum(lstm_predictions == 0)} predicted No Purchase"
)


# ------------------------------------------------------------
# GRU
# ------------------------------------------------------------

start = time.perf_counter()

gru_probabilities = gru_model.predict(
    test_sequences,
    verbose=0
).ravel()

gru_time = time.perf_counter() - start

gru_predictions = (
    gru_probabilities >= THRESHOLD
).astype(int)

results.append([
    "GRU",
    len(test_sequences),
    int(np.sum(gru_predictions == 1)),
    int(np.sum(gru_predictions == 0)),
    float(np.mean(gru_probabilities)),
    float(np.mean(gru_probabilities >= THRESHOLD)),
    gru_time
])

print(
    f"GRU             : "
    f"{np.sum(gru_predictions == 1)} predicted Purchase, "
    f"{np.sum(gru_predictions == 0)} predicted No Purchase"
)


# ------------------------------------------------------------
# GRU + ATTENTION
# ------------------------------------------------------------

start = time.perf_counter()

attention_probabilities = attention_model.predict(
    test_sequences,
    verbose=0
).ravel()

attention_time = time.perf_counter() - start

attention_predictions = (
    attention_probabilities >= THRESHOLD
).astype(int)

results.append([
    "GRU + Attention",
    len(test_sequences),
    int(np.sum(attention_predictions == 1)),
    int(np.sum(attention_predictions == 0)),
    float(np.mean(attention_probabilities)),
    float(np.mean(attention_probabilities >= THRESHOLD)),
    attention_time
])

print(
    f"GRU + Attention  : "
    f"{np.sum(attention_predictions == 1)} predicted Purchase, "
    f"{np.sum(attention_predictions == 0)} predicted No Purchase"
)


# ------------------------------------------------------------
# XGBOOST
# ------------------------------------------------------------

start = time.perf_counter()

xgb_probabilities = xgb_model.predict_proba(
    aggregated_features
)[:, 1]

xgb_time = time.perf_counter() - start

xgb_predictions = (
    xgb_probabilities >= THRESHOLD
).astype(int)

results.append([
    "XGBoost",
    len(test_sequences),
    int(np.sum(xgb_predictions == 1)),
    int(np.sum(xgb_predictions == 0)),
    float(np.mean(xgb_probabilities)),
    float(np.mean(xgb_probabilities >= THRESHOLD)),
    xgb_time
])

print(
    f"XGBoost         : "
    f"{np.sum(xgb_predictions == 1)} predicted Purchase, "
    f"{np.sum(xgb_predictions == 0)} predicted No Purchase"
)


# ------------------------------------------------------------
# LIGHTGBM
# ------------------------------------------------------------

start = time.perf_counter()

lgbm_probabilities = lgbm_model.predict(
    aggregated_features
)

lgbm_time = time.perf_counter() - start

lgbm_predictions = (
    lgbm_probabilities >= THRESHOLD
).astype(int)

results.append([
    "LightGBM",
    len(test_sequences),
    int(np.sum(lgbm_predictions == 1)),
    int(np.sum(lgbm_predictions == 0)),
    float(np.mean(lgbm_probabilities)),
    float(np.mean(lgbm_probabilities >= THRESHOLD)),
    lgbm_time
])

print(
    f"LightGBM        : "
    f"{np.sum(lgbm_predictions == 1)} predicted Purchase, "
    f"{np.sum(lgbm_predictions == 0)} predicted No Purchase"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

columns = [
    "model",
    "anonymous_sessions",
    "predicted_purchase",
    "predicted_no_purchase",
    "mean_purchase_probability",
    "predicted_purchase_rate",
    "prediction_time_seconds"
]

results_df = pd.DataFrame(
    results,
    columns=columns
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

sample_size = min(10, len(test_sequences))

sample_df = pd.DataFrame({
    "anonymous_session": np.arange(1, sample_size + 1),
    "actual_label": test_labels[:sample_size],
    "LSTM_probability": lstm_probabilities[:sample_size],
    "GRU_probability": gru_probabilities[:sample_size],
    "GRU_Attention_probability": attention_probabilities[:sample_size],
    "XGBoost_probability": xgb_probabilities[:sample_size],
    "LightGBM_probability": lgbm_probabilities[:sample_size]
})

print("\n" + "-" * 60)
print("SAMPLE ANONYMOUS-SESSION PREDICTIONS")
print("-" * 60)

print(sample_df.to_string(index=False))


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ANONYMOUS USER EXPERIMENT COMPLETED")
print("=" * 60)

print("\nPrediction summary:")
print(results_df.to_string(index=False))

print(
    f"\nSaved results -> {OUTPUT_PATH}"
)

print("\nImportant:")
print("Actual labels were used only for experimental evaluation.")
print("No user identity or historical user information was used.")