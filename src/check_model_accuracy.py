

import os
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "yoochoose_dl_subset.npz"
)

LSTM_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "deep_learning",
    "lstm_model.keras"
)

GRU_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "deep_learning",
    "gru_model.keras"
)

GRU_ATTENTION_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "deep_learning",
    "gru_attention_model.keras"
)

XGBOOST_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "machine_learning",
    "xgboost_model.json"
)

LIGHTGBM_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "machine_learning",
    "lightgbm_model.txt"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "metrics"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "final_model_accuracy_comparison.csv"
)


# ============================================================
# 2. CHECK FILES
# ============================================================

def check_required_files():
    """
    Check whether all required data and model files exist.
    """

    required_files = {
        "Test dataset": DATA_FILE,
        "LSTM model": LSTM_MODEL,
        "GRU model": GRU_MODEL,
        "GRU + Attention model": GRU_ATTENTION_MODEL,
        "XGBoost model": XGBOOST_MODEL,
        "LightGBM model": LIGHTGBM_MODEL,
    }

    print("\n" + "=" * 80)
    print("CHECKING REQUIRED FILES")
    print("=" * 80)

    missing = []

    for name, path in required_files.items():

        if os.path.exists(path):
            print(f"[OK] {name}")
            print(f"     {path}")
        else:
            print(f"[MISSING] {name}")
            print(f"          {path}")
            missing.append(name)

    if missing:
        print("\nERROR: Some required files are missing.")
        print("Please make sure the models have already been trained.")
        raise FileNotFoundError(
            "Missing required files: " + ", ".join(missing)
        )

    print("\nAll required files are available.")


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

def load_test_data():
    """
    Load the test sequences and labels from the existing NPZ file.
    """

    print("\n" + "=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)

    data = np.load(DATA_FILE, allow_pickle=True)

    required_keys = [
        "test_sequences",
        "test_time_deltas",
        "test_labels",
    ]

    for key in required_keys:
        if key not in data:
            raise KeyError(
                f"Required key '{key}' was not found in {DATA_FILE}"
            )

    X_test = data["test_sequences"].astype(np.int32)
    time_deltas = data["test_time_deltas"].astype(np.float32)
    y_test = data["test_labels"].astype(np.int32)

    print(f"Test sequences shape : {X_test.shape}")
    print(f"Time deltas shape    : {time_deltas.shape}")
    print(f"Test labels shape    : {y_test.shape}")

    print("\nTest class distribution:")

    unique, counts = np.unique(y_test, return_counts=True)

    for label, count in zip(unique, counts):
        percentage = count / len(y_test) * 100
        print(
            f"  Class {label}: {count:,} "
            f"({percentage:.2f}%)"
        )

    return X_test, time_deltas, y_test


# ============================================================
# 4. CREATE AGGREGATED FEATURES
# ============================================================

def create_aggregated_features(sequences, time_deltas):
    """
    Create the same 14 aggregated behavioral features used by
    the XGBoost and LightGBM models.

    Features:

    1. interaction_count
    2. unique_item_count
    3. repeated_item_count
    4. first_item_id
    5. last_item_id
    6. mean_item_id
    7. std_item_id
    8. min_item_id
    9. max_item_id
    10. total_time_delta
    11. mean_time_delta
    12. max_time_delta
    13. std_time_delta
    14. nonzero_time_intervals
    """

    print("\nCreating aggregated behavioral features...")

    n_samples = sequences.shape[0]

    features = np.zeros(
        (n_samples, 14),
        dtype=np.float32
    )

    for i in range(n_samples):

        seq = sequences[i]
        times = time_deltas[i]

        # ----------------------------------------------------
        # Padding value is assumed to be 0.
        # ----------------------------------------------------

        valid_mask = seq != 0

        items = seq[valid_mask]

        if len(items) == 0:
            items = np.array([0], dtype=np.int32)

        valid_times = times[valid_mask]

        # ----------------------------------------------------
        # Basic interaction features
        # ----------------------------------------------------

        interaction_count = len(items)

        unique_items = np.unique(items)

        unique_item_count = len(unique_items)

        repeated_item_count = (
            interaction_count - unique_item_count
        )

        # ----------------------------------------------------
        # Item statistics
        # ----------------------------------------------------

        first_item_id = float(items[0])

        last_item_id = float(items[-1])

        mean_item_id = float(np.mean(items))

        std_item_id = float(np.std(items))

        min_item_id = float(np.min(items))

        max_item_id = float(np.max(items))

        # ----------------------------------------------------
        # Time statistics
        # ----------------------------------------------------

        if len(valid_times) == 0:
            total_time_delta = 0.0
            mean_time_delta = 0.0
            max_time_delta = 0.0
            std_time_delta = 0.0
            nonzero_time_intervals = 0.0

        else:

            total_time_delta = float(
                np.sum(valid_times)
            )

            mean_time_delta = float(
                np.mean(valid_times)
            )

            max_time_delta = float(
                np.max(valid_times)
            )

            std_time_delta = float(
                np.std(valid_times)
            )

            nonzero_time_intervals = float(
                np.count_nonzero(valid_times)
            )

        # ----------------------------------------------------
        # Store features
        # ----------------------------------------------------

        features[i] = [
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
            nonzero_time_intervals,
        ]

    print(
        f"Aggregated feature matrix shape: {features.shape}"
    )

    return features


# ============================================================
# 5. CALCULATE METRICS
# ============================================================

def calculate_metrics(
    model_name,
    y_true,
    probabilities,
    prediction_time
):
    """
    Calculate all important classification metrics.
    """

    probabilities = np.asarray(probabilities).reshape(-1)

    # --------------------------------------------------------
    # Standard classification threshold
    # --------------------------------------------------------

    threshold = 0.50

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        predictions
    )

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

    try:
        roc_auc = roc_auc_score(
            y_true,
            probabilities
        )
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(
            y_true,
            probabilities
        )
    except ValueError:
        pr_auc = np.nan

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    print("\n" + "-" * 80)
    print(model_name)
    print("-" * 80)

    print(f"Accuracy   : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print(f"ROC-AUC    : {roc_auc:.4f}")
    print(f"PR-AUC     : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(f"TN = {tn:,}")
    print(f"FP = {fp:,}")
    print(f"FN = {fn:,}")
    print(f"TP = {tp:,}")

    print(
        f"\nPrediction time: {prediction_time:.4f} seconds"
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "prediction_time_seconds": prediction_time,
        "decision_threshold": threshold,
    }


# ============================================================
# 6. LOAD TENSORFLOW MODELS
# ============================================================

def load_tensorflow_models():
    """
    Load the three saved Keras models.
    """

    print("\n" + "=" * 80)
    print("LOADING DEEP LEARNING MODELS")
    print("=" * 80)

    import tensorflow as tf

    print("[1/3] Loading LSTM...")
    lstm = tf.keras.models.load_model(
        LSTM_MODEL
    )

    print("[2/3] Loading GRU...")
    gru = tf.keras.models.load_model(
        GRU_MODEL
    )

    print("[3/3] Loading GRU + Attention...")
    gru_attention = tf.keras.models.load_model(
        GRU_ATTENTION_MODEL
    )

    print("All deep learning models loaded.")

    return lstm, gru, gru_attention


# ============================================================
# 7. DEEP LEARNING PREDICTION
# ============================================================

def predict_deep_learning(
    model,
    X_test,
    model_name
):
    """
    Generate probabilities using a saved Keras model.
    """

    print("\n" + "=" * 80)
    print(f"GENERATING PREDICTIONS: {model_name}")
    print("=" * 80)

    start_time = time.perf_counter()

    probabilities = model.predict(
        X_test,
        batch_size=64,
        verbose=0
    )

    end_time = time.perf_counter()

    prediction_time = (
        end_time - start_time
    )

    probabilities = np.asarray(
        probabilities
    ).reshape(-1)

    print(
        f"Generated {len(probabilities):,} predictions."
    )

    return probabilities, prediction_time


# ============================================================
# 8. LOAD XGBOOST
# ============================================================

def load_xgboost():
    """
    Load saved XGBoost model.
    """

    print("\n" + "=" * 80)
    print("LOADING XGBOOST")
    print("=" * 80)

    from xgboost import XGBClassifier

    model = XGBClassifier()

    model.load_model(
        XGBOOST_MODEL
    )

    print("XGBoost model loaded.")

    return model


# ============================================================
# 9. LOAD LIGHTGBM
# ============================================================

def load_lightgbm():
    """
    Load saved LightGBM model.
    """

    print("\n" + "=" * 80)
    print("LOADING LIGHTGBM")
    print("=" * 80)

    import lightgbm as lgb

    model = lgb.Booster(
        model_file=LIGHTGBM_MODEL
    )

    print("LightGBM model loaded.")

    return model


# ============================================================
# 10. XGBOOST PREDICTION
# ============================================================

def predict_xgboost(
    model,
    X_aggregated
):
    """
    Generate XGBoost purchase probabilities.
    """

    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS: XGBoost")
    print("=" * 80)

    start_time = time.perf_counter()

    probabilities = model.predict_proba(
        X_aggregated
    )[:, 1]

    end_time = time.perf_counter()

    prediction_time = (
        end_time - start_time
    )

    return (
        np.asarray(probabilities),
        prediction_time
    )


# ============================================================
# 11. LIGHTGBM PREDICTION
# ============================================================

def predict_lightgbm(
    model,
    X_aggregated
):
    """
    Generate LightGBM purchase probabilities.
    """

    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS: LightGBM")
    print("=" * 80)

    start_time = time.perf_counter()

    probabilities = model.predict(
        X_aggregated
    )

    end_time = time.perf_counter()

    prediction_time = (
        end_time - start_time
    )

    return (
        np.asarray(probabilities),
        prediction_time
    )


# ============================================================
# 12. PRINT FINAL COMPARISON
# ============================================================

def print_final_comparison(results):
    """
    Print a clean final comparison table.
    """

    df = pd.DataFrame(results)

    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
    ]

    display_df = df[display_columns].copy()

    for column in display_columns[1:]:
        display_df[column] = display_df[column].map(
            lambda x: f"{x:.4f}"
        )

    print("\n" + "=" * 100)
    print("FINAL MODEL PERFORMANCE COMPARISON")
    print("=" * 100)

    print(display_df.to_string(index=False))

    print("\nAccuracy in percentage:")

    for _, row in df.iterrows():

        print(
            f"{row['model']:<20} "
            f"{row['accuracy'] * 100:.2f}%"
        )

    return df


# ============================================================
# 13. SAVE RESULTS
# ============================================================

def save_results(df):
    """
    Save the final model accuracy comparison CSV.
    """

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 80)
    print("RESULT SAVED")
    print("=" * 80)

    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )


# ============================================================
# 14. MAIN
# ============================================================

def main():

    total_start = time.perf_counter()

    print("\n")
    print("=" * 80)
    print("STEP 18 - FINAL MODEL ACCURACY CHECK")
    print("=" * 80)

    print("\nIMPORTANT:")
    print("- No model will be trained.")
    print("- Existing saved models will be loaded.")
    print("- Predictions will be generated on the existing test set.")
    print("- Accuracy and other evaluation metrics will be calculated.")
    print("=" * 80)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    check_required_files()

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    X_test, time_deltas, y_test = load_test_data()

    # --------------------------------------------------------
    # Create aggregated features
    # --------------------------------------------------------

    X_aggregated = create_aggregated_features(
        X_test,
        time_deltas
    )

    results = []

    # ========================================================
    # DEEP LEARNING MODELS
    # ========================================================

    lstm, gru, gru_attention = load_tensorflow_models()

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    lstm_probabilities, lstm_time = predict_deep_learning(
        lstm,
        X_test,
        "LSTM"
    )

    lstm_result = calculate_metrics(
        "LSTM",
        y_test,
        lstm_probabilities,
        lstm_time
    )

    results.append(lstm_result)

    # --------------------------------------------------------
    # GRU
    # --------------------------------------------------------

    gru_probabilities, gru_time = predict_deep_learning(
        gru,
        X_test,
        "GRU"
    )

    gru_result = calculate_metrics(
        "GRU",
        y_test,
        gru_probabilities,
        gru_time
    )

    results.append(gru_result)

    # --------------------------------------------------------
    # GRU + Attention
    # --------------------------------------------------------

    attention_probabilities, attention_time = predict_deep_learning(
        gru_attention,
        X_test,
        "GRU + Attention"
    )

    attention_result = calculate_metrics(
        "GRU + Attention",
        y_test,
        attention_probabilities,
        attention_time
    )

    results.append(attention_result)

    # ========================================================
    # MACHINE LEARNING MODELS
    # ========================================================

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    xgb_model = load_xgboost()

    xgb_probabilities, xgb_time = predict_xgboost(
        xgb_model,
        X_aggregated
    )

    xgb_result = calculate_metrics(
        "XGBoost",
        y_test,
        xgb_probabilities,
        xgb_time
    )

    results.append(xgb_result)

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    lgb_model = load_lightgbm()

    lgb_probabilities, lgb_time = predict_lightgbm(
        lgb_model,
        X_aggregated
    )

    lgb_result = calculate_metrics(
        "LightGBM",
        y_test,
        lgb_probabilities,
        lgb_time
    )

    results.append(lgb_result)

    # ========================================================
    # FINAL TABLE
    # ========================================================

    final_df = print_final_comparison(
        results
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    save_results(
        final_df
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    total_time = (
        time.perf_counter() - total_start
    )

    print("\n" + "=" * 80)
    print("WHAT THESE METRICS MEAN")
    print("=" * 80)

    print(
        """
Accuracy:
    Percentage of all test sessions classified correctly.

Precision:
    Of the sessions predicted as purchase,
    how many were actually purchases?

Recall:
    Of the actual purchase sessions,
    how many did the model identify?

F1 Score:
    Balance between Precision and Recall.

ROC-AUC:
    Measures how well the model separates purchase
    and non-purchase sessions across thresholds.

PR-AUC:
    Particularly useful when the purchase class is
    much smaller than the non-purchase class.

Confusion Matrix:
    TN = correctly predicted no-purchase
    FP = incorrectly predicted purchase
    FN = missed purchase
    TP = correctly predicted purchase
"""
    )

    print("=" * 80)
    print(
        f"Total evaluation time: {total_time:.2f} seconds"
    )
    print("=" * 80)

    print("\nDONE.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()