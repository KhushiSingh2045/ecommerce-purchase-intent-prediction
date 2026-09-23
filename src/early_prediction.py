import os
import time
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
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

OUTPUT_PATH = "results/metrics/early_prediction_metrics.csv"

SEQUENCE_LENGTH = 20

# Percentages of the session used for prediction
PERCENTAGES = [20, 40, 60, 80, 100]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("results/metrics", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("EARLY PURCHASE-INTENT PREDICTION")
print("=" * 60)

print("\nLoading dataset...")

data = np.load(DATA_PATH, allow_pickle=True)

test_sequences = data["test_sequences"]
test_time_deltas = data["test_time_deltas"]
test_labels = data["test_labels"]

print("Test sequences :", test_sequences.shape)
print("Test labels    :", test_labels.shape)


# ============================================================
# AGGREGATED FEATURES
# SAME 14 FEATURES USED BY XGBOOST/LIGHTGBM
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
# EVALUATION FUNCTION
# ============================================================

def evaluate_predictions(y_true, probabilities):

    predictions = (probabilities >= 0.5).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    return precision, recall, f1, roc_auc


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading trained models...")

lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
gru_model = tf.keras.models.load_model(GRU_MODEL_PATH)
attention_model = tf.keras.models.load_model(ATTENTION_MODEL_PATH)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_MODEL_PATH)

lgbm_model = lgb.Booster(model_file=LGBM_MODEL_PATH)

print("All models loaded successfully.")


# ============================================================
# EARLY PREDICTION EXPERIMENT
# ============================================================

results = []


for percentage in PERCENTAGES:

    number_of_events = max(
        1,
        int(SEQUENCE_LENGTH * percentage / 100)
    )

    print("\n" + "-" * 60)
    print(
        f"Using first {percentage}% "
        f"({number_of_events}/{SEQUENCE_LENGTH} events)"
    )
    print("-" * 60)

    partial_sequences = test_sequences.copy()
    partial_times = test_time_deltas.copy()

    # Keep only the first part of each session
    partial_sequences[:, number_of_events:] = 0
    partial_times[:, number_of_events:] = 0

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    start = time.perf_counter()

    lstm_prob = lstm_model.predict(
        partial_sequences,
        verbose=0
    ).ravel()

    lstm_time = time.perf_counter() - start

    precision, recall, f1, auc = evaluate_predictions(
        test_labels,
        lstm_prob
    )

    results.append([
        "LSTM",
        percentage,
        precision,
        recall,
        f1,
        auc,
        lstm_time
    ])

    print(
        f"LSTM           "
        f"Precision={precision:.4f} "
        f"Recall={recall:.4f} "
        f"F1={f1:.4f} "
        f"ROC-AUC={auc:.4f}"
    )


    # --------------------------------------------------------
    # GRU
    # --------------------------------------------------------

    start = time.perf_counter()

    gru_prob = gru_model.predict(
        partial_sequences,
        verbose=0
    ).ravel()

    gru_time = time.perf_counter() - start

    precision, recall, f1, auc = evaluate_predictions(
        test_labels,
        gru_prob
    )

    results.append([
        "GRU",
        percentage,
        precision,
        recall,
        f1,
        auc,
        gru_time
    ])

    print(
        f"GRU            "
        f"Precision={precision:.4f} "
        f"Recall={recall:.4f} "
        f"F1={f1:.4f} "
        f"ROC-AUC={auc:.4f}"
    )


    # --------------------------------------------------------
    # GRU + ATTENTION
    # --------------------------------------------------------

    start = time.perf_counter()

    attention_prob = attention_model.predict(
        partial_sequences,
        verbose=0
    ).ravel()

    attention_time = time.perf_counter() - start

    precision, recall, f1, auc = evaluate_predictions(
        test_labels,
        attention_prob
    )

    results.append([
        "GRU + Attention",
        percentage,
        precision,
        recall,
        f1,
        auc,
        attention_time
    ])

    print(
        f"GRU + Attention "
        f"Precision={precision:.4f} "
        f"Recall={recall:.4f} "
        f"F1={f1:.4f} "
        f"ROC-AUC={auc:.4f}"
    )


    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    xgb_features = create_features(
        partial_sequences,
        partial_times
    )

    start = time.perf_counter()

    xgb_prob = xgb_model.predict_proba(
        xgb_features
    )[:, 1]

    xgb_time = time.perf_counter() - start

    precision, recall, f1, auc = evaluate_predictions(
        test_labels,
        xgb_prob
    )

    results.append([
        "XGBoost",
        percentage,
        precision,
        recall,
        f1,
        auc,
        xgb_time
    ])

    print(
        f"XGBoost        "
        f"Precision={precision:.4f} "
        f"Recall={recall:.4f} "
        f"F1={f1:.4f} "
        f"ROC-AUC={auc:.4f}"
    )


    # --------------------------------------------------------
    # LIGHTGBM
    # --------------------------------------------------------

    lgb_features = create_features(
        partial_sequences,
        partial_times
    )

    start = time.perf_counter()

    lgb_prob = lgbm_model.predict(
        lgb_features
    )

    lgb_time = time.perf_counter() - start

    precision, recall, f1, auc = evaluate_predictions(
        test_labels,
        lgb_prob
    )

    results.append([
        "LightGBM",
        percentage,
        precision,
        recall,
        f1,
        auc,
        lgb_time
    ])

    print(
        f"LightGBM       "
        f"Precision={precision:.4f} "
        f"Recall={recall:.4f} "
        f"F1={f1:.4f} "
        f"ROC-AUC={auc:.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

columns = [
    "model",
    "session_percentage",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
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

print("\n" + "=" * 60)
print("EARLY PREDICTION EXPERIMENT COMPLETED")
print("=" * 60)

print("\nResults:")
print(results_df.to_string(index=False))

print(
    f"\nSaved results -> {OUTPUT_PATH}"
)