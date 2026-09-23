"""
XGBoost Purchase Intent Prediction
==================================

Dataset:
    data/processed/yoochoose_dl_subset.npz

Existing dataset keys:
    train_sequences
    train_time_deltas
    train_labels
    train_session_ids
    val_sequences
    val_time_deltas
    val_labels
    val_session_ids
    test_sequences
    test_time_deltas
    test_labels
    test_session_ids

Model:
    XGBoost

Features:
    14 aggregated behavioral features

Threshold tuning:
    Validation set only
    Threshold selected using maximum validation F1

Outputs:
    models/machine_learning/xgboost_model.json

    results/metrics/xgboost_metrics.csv
    results/metrics/xgboost_metrics_threshold_comparison.csv
    results/metrics/xgboost_validation_threshold_search.csv

    results/figures/xgboost_confusion_matrix.png
    results/figures/xgboost_roc_curve.png
    results/figures/xgboost_feature_importance.png
    results/figures/shap_summary_xgboost.png
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "yoochoose_dl_subset.npz"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "machine_learning"
)

METRICS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "metrics"
)

FIGURES_DIR = os.path.join(
    BASE_DIR,
    "results",
    "figures"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

N_ESTIMATORS = 300
MAX_DEPTH = 6
LEARNING_RATE = 0.05

SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8

DEFAULT_THRESHOLD = 0.50


# =============================================================================
# FEATURE NAMES
# =============================================================================

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
# LOAD DATA
# =============================================================================

def load_data():

    print("\nLoading dataset:")
    print(DATA_PATH)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_PATH}\n\n"
            "Make sure yoochoose_dl_subset.npz exists in "
            "data/processed/"
        )

    data = np.load(
        DATA_PATH,
        allow_pickle=True
    )

    required_keys = [
        "train_sequences",
        "train_time_deltas",
        "train_labels",

        "val_sequences",
        "val_time_deltas",
        "val_labels",

        "test_sequences",
        "test_time_deltas",
        "test_labels",
    ]

    for key in required_keys:
        if key not in data:
            raise KeyError(
                f"Required dataset key is missing: {key}"
            )

    train_sequences = data["train_sequences"]
    train_time_deltas = data["train_time_deltas"]
    train_labels = data["train_labels"]

    val_sequences = data["val_sequences"]
    val_time_deltas = data["val_time_deltas"]
    val_labels = data["val_labels"]

    test_sequences = data["test_sequences"]
    test_time_deltas = data["test_time_deltas"]
    test_labels = data["test_labels"]

    print("\nDataset loaded successfully.")

    print(
        f"Train sequences : {train_sequences.shape}"
    )

    print(
        f"Validation sequences : {val_sequences.shape}"
    )

    print(
        f"Test sequences : {test_sequences.shape}"
    )

    return (
        train_sequences,
        train_time_deltas,
        train_labels,

        val_sequences,
        val_time_deltas,
        val_labels,

        test_sequences,
        test_time_deltas,
        test_labels,
    )


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_aggregated_features(
    sequences,
    time_deltas
):
    """
    Convert fixed-length sequential sessions into
    14 aggregated behavioral features.
    """

    sequences = np.asarray(
        sequences,
        dtype=np.int32
    )

    time_deltas = np.asarray(
        time_deltas,
        dtype=np.float32
    )

    n_sessions = sequences.shape[0]

    features = np.zeros(
        (n_sessions, 14),
        dtype=np.float32
    )

    for i in range(n_sessions):

        items = sequences[i]

        # Padding value is 0.
        valid_items = items[items != 0]

        if len(valid_items) == 0:

            interaction_count = 0
            unique_item_count = 0
            repeated_item_count = 0

            first_item = 0
            last_item = 0

            mean_item = 0
            std_item = 0

            min_item = 0
            max_item = 0

        else:

            interaction_count = len(valid_items)

            unique_item_count = len(
                np.unique(valid_items)
            )

            repeated_item_count = (
                interaction_count
                - unique_item_count
            )

            first_item = valid_items[0]
            last_item = valid_items[-1]

            mean_item = np.mean(
                valid_items
            )

            std_item = np.std(
                valid_items
            )

            min_item = np.min(
                valid_items
            )

            max_item = np.max(
                valid_items
            )

        # ---------------------------------------------------------
        # TIME FEATURES
        # ---------------------------------------------------------

        deltas = time_deltas[i]

        valid_deltas = deltas[
            deltas > 0
        ]

        if len(valid_deltas) == 0:

            total_time_delta = 0
            mean_time_delta = 0
            max_time_delta = 0
            std_time_delta = 0
            nonzero_time_intervals = 0

        else:

            total_time_delta = np.sum(
                valid_deltas
            )

            mean_time_delta = np.mean(
                valid_deltas
            )

            max_time_delta = np.max(
                valid_deltas
            )

            std_time_delta = np.std(
                valid_deltas
            )

            nonzero_time_intervals = len(
                valid_deltas
            )

        features[i] = [
            interaction_count,
            unique_item_count,
            repeated_item_count,
            first_item,
            last_item,
            mean_item,
            std_item,
            min_item,
            max_item,
            total_time_delta,
            mean_time_delta,
            max_time_delta,
            std_time_delta,
            nonzero_time_intervals,
        ]

    return features


# =============================================================================
# THRESHOLD TUNING
# =============================================================================

def tune_threshold_on_validation(
    model,
    X_val,
    y_val
):

    print("\n" + "=" * 80)
    print("VALIDATION-BASED THRESHOLD TUNING")
    print("=" * 80)

    validation_probability = model.predict_proba(
        X_val
    )[:, 1]

    best_threshold = DEFAULT_THRESHOLD
    best_f1 = -1

    results = []

    for threshold in np.arange(
        0.05,
        0.95,
        0.01
    ):

        prediction = (
            validation_probability
            >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_val,
            prediction,
            zero_division=0
        )

        results.append(
            {
                "threshold": round(
                    float(threshold),
                    2
                ),
                "validation_f1": current_f1
            }
        )

        if current_f1 > best_f1:

            best_f1 = current_f1
            best_threshold = float(
                threshold
            )

    print(
        f"\nBest validation threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Best validation F1: "
        f"{best_f1:.4f}"
    )

    return (
        best_threshold,
        best_f1,
        results
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("XGBOOST PURCHASE INTENT MODEL")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------

    (
        train_sequences,
        train_time_deltas,
        y_train,

        val_sequences,
        val_time_deltas,
        y_val,

        test_sequences,
        test_time_deltas,
        y_test,

    ) = load_data()

    # -------------------------------------------------------------------------
    # 2. CREATE AGGREGATED FEATURES
    # -------------------------------------------------------------------------

    print("\nCreating 14 aggregated behavioral features...")

    feature_start = time.perf_counter()

    X_train = create_aggregated_features(
        train_sequences,
        train_time_deltas
    )

    X_val = create_aggregated_features(
        val_sequences,
        val_time_deltas
    )

    X_test = create_aggregated_features(
        test_sequences,
        test_time_deltas
    )

    feature_time = (
        time.perf_counter()
        - feature_start
    )

    print(
        f"\nFeature generation completed in "
        f"{feature_time:.2f} seconds."
    )

    print(
        f"X_train shape: {X_train.shape}"
    )

    print(
        f"X_val shape: {X_val.shape}"
    )

    print(
        f"X_test shape: {X_test.shape}"
    )

    # -------------------------------------------------------------------------
    # 3. CLASS IMBALANCE
    # -------------------------------------------------------------------------

    negative_count = np.sum(
        y_train == 0
    )

    positive_count = np.sum(
        y_train == 1
    )

    if positive_count == 0:
        raise ValueError(
            "No positive purchase samples found."
        )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print(
        f"\nNegative samples: "
        f"{negative_count}"
    )

    print(
        f"Positive samples: "
        f"{positive_count}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # -------------------------------------------------------------------------
    # 4. CREATE XGBOOST MODEL
    # -------------------------------------------------------------------------

    print("\nCreating XGBoost model...")

    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,

        max_depth=MAX_DEPTH,

        learning_rate=LEARNING_RATE,

        subsample=SUBSAMPLE,

        colsample_bytree=COLSAMPLE_BYTREE,

        scale_pos_weight=scale_pos_weight,

        objective="binary:logistic",

        eval_metric="logloss",

        tree_method="hist",

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )

    # -------------------------------------------------------------------------
    # 5. TRAIN
    # -------------------------------------------------------------------------

    print("\nTraining XGBoost...")

    training_start = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    training_time = (
        time.perf_counter()
        - training_start
    )

    print(
        f"\nTraining time: "
        f"{training_time:.2f} seconds"
    )

    # -------------------------------------------------------------------------
    # 6. VALIDATION THRESHOLD TUNING
    # -------------------------------------------------------------------------

    (
        tuned_threshold,
        validation_best_f1,
        threshold_results

    ) = tune_threshold_on_validation(
        model,
        X_val,
        y_val
    )

    # -------------------------------------------------------------------------
    # 7. TEST PROBABILITIES
    # -------------------------------------------------------------------------

    print("\nGenerating test probabilities...")

    prediction_start = time.perf_counter()

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    prediction_time = (
        time.perf_counter()
        - prediction_start
    )

    # -------------------------------------------------------------------------
    # 8. DEFAULT 0.50
    # -------------------------------------------------------------------------

    default_prediction = (
        test_probability
        >= DEFAULT_THRESHOLD
    ).astype(int)

    precision_default = precision_score(
        y_test,
        default_prediction,
        zero_division=0
    )

    recall_default = recall_score(
        y_test,
        default_prediction,
        zero_division=0
    )

    f1_default = f1_score(
        y_test,
        default_prediction,
        zero_division=0
    )

    # -------------------------------------------------------------------------
    # 9. TUNED THRESHOLD
    # -------------------------------------------------------------------------

    tuned_prediction = (
        test_probability
        >= tuned_threshold
    ).astype(int)

    precision_tuned = precision_score(
        y_test,
        tuned_prediction,
        zero_division=0
    )

    recall_tuned = recall_score(
        y_test,
        tuned_prediction,
        zero_division=0
    )

    f1_tuned = f1_score(
        y_test,
        tuned_prediction,
        zero_division=0
    )

    # -------------------------------------------------------------------------
    # 10. ROC-AUC AND PR-AUC
    # -------------------------------------------------------------------------

    roc_auc = roc_auc_score(
        y_test,
        test_probability
    )

    pr_auc = average_precision_score(
        y_test,
        test_probability
    )

    # -------------------------------------------------------------------------
    # 11. PRINT RESULTS
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("XGBOOST TEST RESULTS")
    print("=" * 80)

    print("\nDEFAULT THRESHOLD = 0.50")

    print(
        f"Precision: {precision_default:.4f}"
    )

    print(
        f"Recall   : {recall_default:.4f}"
    )

    print(
        f"F1-score : {f1_default:.4f}"
    )

    print("\nTUNED VALIDATION THRESHOLD")

    print(
        f"Threshold: {tuned_threshold:.2f}"
    )

    print(
        f"Precision: {precision_tuned:.4f}"
    )

    print(
        f"Recall   : {recall_tuned:.4f}"
    )

    print(
        f"F1-score : {f1_tuned:.4f}"
    )

    print("\nTHRESHOLD-INDEPENDENT METRICS")

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC : {pr_auc:.4f}"
    )

    # -------------------------------------------------------------------------
    # 12. CONFUSION MATRIX
    # -------------------------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        tuned_prediction
    )

    print("\nTuned threshold confusion matrix:")

    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Purchase",
            "Purchase"
        ]
    )

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    display.plot(
        ax=axis,
        values_format="d"
    )

    axis.set_title(
        f"XGBoost Confusion Matrix "
        f"(Threshold = {tuned_threshold:.2f})"
    )

    figure.tight_layout()

    confusion_path = os.path.join(
        FIGURES_DIR,
        "xgboost_confusion_matrix.png"
    )

    figure.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        f"[SAVED] {confusion_path}"
    )

    # -------------------------------------------------------------------------
    # 13. CLASSIFICATION REPORT
    # -------------------------------------------------------------------------

    print(
        "\nClassification Report "
        "(Tuned Threshold)"
    )

    print(
        classification_report(
            y_test,
            tuned_prediction,
            target_names=[
                "No Purchase",
                "Purchase"
            ],
            zero_division=0
        )
    )

    # -------------------------------------------------------------------------
    # 14. SAVE MODEL
    # -------------------------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "xgboost_model.json"
    )

    model.save_model(
        model_path
    )

    print(
        f"[SAVED] {model_path}"
    )

    # -------------------------------------------------------------------------
    # 15. SAVE MAIN METRICS
    # -------------------------------------------------------------------------

    metrics = pd.DataFrame(
        [
            {
                "model": "XGBoost",

                "precision": precision_tuned,

                "recall": recall_tuned,

                "f1_score": f1_tuned,

                "roc_auc": roc_auc,

                "training_time_seconds":
                    training_time,

                "prediction_time_seconds":
                    prediction_time,

                "n_estimators":
                    N_ESTIMATORS,

                "max_depth":
                    MAX_DEPTH,

                "learning_rate":
                    LEARNING_RATE,

                "scale_pos_weight":
                    scale_pos_weight,

                "feature_count":
                    X_train.shape[1],

                "decision_threshold":
                    tuned_threshold,

                "threshold_selection":
                    "validation_F1",
            }
        ]
    )

    metrics_path = os.path.join(
        METRICS_DIR,
        "xgboost_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False
    )

    print(
        f"[SAVED] {metrics_path}"
    )

    # -------------------------------------------------------------------------
    # 16. SAVE THRESHOLD COMPARISON
    # -------------------------------------------------------------------------

    threshold_comparison = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "threshold_type": "default_0.5",
                "threshold": DEFAULT_THRESHOLD,
                "precision": precision_default,
                "recall": recall_default,
                "f1_score": f1_default,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            },

            {
                "model": "XGBoost",
                "threshold_type": "tuned_validation",
                "threshold": tuned_threshold,
                "precision": precision_tuned,
                "recall": recall_tuned,
                "f1_score": f1_tuned,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            },
        ]
    )

    threshold_path = os.path.join(
        METRICS_DIR,
        "xgboost_metrics_threshold_comparison.csv"
    )

    threshold_comparison.to_csv(
        threshold_path,
        index=False
    )

    print(
        f"[SAVED] {threshold_path}"
    )

    # -------------------------------------------------------------------------
    # 17. SAVE VALIDATION THRESHOLD SEARCH
    # -------------------------------------------------------------------------

    threshold_search = pd.DataFrame(
        threshold_results
    )

    threshold_search_path = os.path.join(
        METRICS_DIR,
        "xgboost_validation_threshold_search.csv"
    )

    threshold_search.to_csv(
        threshold_search_path,
        index=False
    )

    print(
        f"[SAVED] {threshold_search_path}"
    )

    # -------------------------------------------------------------------------
    # 18. ROC CURVE
    # -------------------------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_test,
        test_probability
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    axis.plot(
        fpr,
        tpr,
        label=f"XGBoost ROC-AUC = {roc_auc:.4f}"
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    axis.set_xlabel(
        "False Positive Rate"
    )

    axis.set_ylabel(
        "True Positive Rate"
    )

    axis.set_title(
        "XGBoost ROC Curve"
    )

    axis.legend()

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    roc_path = os.path.join(
        FIGURES_DIR,
        "xgboost_roc_curve.png"
    )

    figure.savefig(
        roc_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        f"[SAVED] {roc_path}"
    )

    # -------------------------------------------------------------------------
    # 19. FEATURE IMPORTANCE
    # -------------------------------------------------------------------------

    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance": importance
        }
    ).sort_values(
        "importance",
        ascending=False
    )

    print("\nFeature Importance:")

    print(
        importance_df.to_string(
            index=False
        )
    )

    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    axis.barh(
        importance_df["feature"],
        importance_df["importance"]
    )

    axis.invert_yaxis()

    axis.set_xlabel(
        "Importance"
    )

    axis.set_ylabel(
        "Feature"
    )

    axis.set_title(
        "XGBoost Feature Importance"
    )

    figure.tight_layout()

    importance_path = os.path.join(
        FIGURES_DIR,
        "xgboost_feature_importance.png"
    )

    figure.savefig(
        importance_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        f"[SAVED] {importance_path}"
    )

    # -------------------------------------------------------------------------
    # 20. FINAL SUMMARY
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("XGBOOST TRAINING COMPLETED")
    print("=" * 80)

    print(
        f"Default threshold : {DEFAULT_THRESHOLD:.2f}"
    )

    print(
        f"Tuned threshold   : {tuned_threshold:.2f}"
    )

    print(
        f"Default F1        : {f1_default:.4f}"
    )

    print(
        f"Tuned F1          : {f1_tuned:.4f}"
    )

    print(
        f"ROC-AUC           : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC            : {pr_auc:.4f}"
    )

    print(
        "\nThreshold selected using validation data only."
    )

    print(
        "Test data was used only for final evaluation."
    )

    print("=" * 80)


if __name__ == "__main__":
    main()