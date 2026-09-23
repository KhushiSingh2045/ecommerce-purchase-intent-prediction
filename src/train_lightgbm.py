

import os
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "machine_learning")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")

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
NUM_LEAVES = 31

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
# HELPER FUNCTIONS
# =============================================================================

def load_dataset():
    """
    Load the prepared YOOCHOOSE aggregated feature dataset.

    The script expects:
        yoochoose_aggregated_features.npz
    """

    dataset_path = os.path.join(
        PROCESSED_DIR,
        "yoochoose_aggregated_features.npz"
    )

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found:\n{dataset_path}\n\n"
            "Make sure the aggregated feature dataset has already been created."
        )

    data = np.load(dataset_path, allow_pickle=True)

    required_keys = [
        "X_train",
        "X_val",
        "X_test",
        "y_train",
        "y_val",
        "y_test",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in data
    ]

    if missing_keys:
        raise KeyError(
            "The dataset is missing the following keys:\n"
            + ", ".join(missing_keys)
        )

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


def calculate_scale_pos_weight(y_train):
    """
    Calculate class weight for the positive class.

    scale_pos_weight =
        number of negative samples / number of positive samples
    """

    negative_count = np.sum(y_train == 0)
    positive_count = np.sum(y_train == 1)

    if positive_count == 0:
        raise ValueError(
            "Training data contains no positive samples."
        )

    return float(negative_count / positive_count)


def tune_threshold_on_validation(model, X_val, y_val):
    """
    Find the decision threshold that maximizes F1 on the validation set.

    IMPORTANT:
        The test set is NOT used here.

    Threshold search:
        0.05 to 0.94
        step = 0.01
    """

    print("\n" + "=" * 80)
    print("VALIDATION-BASED THRESHOLD TUNING")
    print("=" * 80)

    print("\nGenerating validation probabilities...")

    validation_probability = model.predict_proba(
        X_val
    )[:, 1]

    best_threshold = DEFAULT_THRESHOLD
    best_f1 = -1.0

    threshold_results = []

    for threshold in np.arange(0.05, 0.95, 0.01):

        validation_prediction = (
            validation_probability >= threshold
        ).astype(int)

        current_f1 = f1_score(
            y_val,
            validation_prediction,
            zero_division=0
        )

        threshold_results.append(
            {
                "threshold": float(threshold),
                "f1_score": float(current_f1),
            }
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)

    print(f"\nBest validation threshold : {best_threshold:.2f}")
    print(f"Best validation F1        : {best_f1:.4f}")

    return best_threshold, best_f1, threshold_results


def calculate_threshold_metrics(y_true, probability, threshold):
    """
    Calculate threshold-dependent classification metrics.
    """

    prediction = (
        probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0
    )

    return (
        prediction,
        precision,
        recall,
        f1,
    )


def save_confusion_matrix(y_true, prediction, threshold):
    """
    Save confusion matrix using the tuned threshold.
    """

    cm = confusion_matrix(
        y_true,
        prediction
    )

    print("\nConfusion Matrix:")
    print(cm)

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Purchase",
            "Purchase"
        ]
    )

    display.plot(
        ax=axis,
        values_format="d"
    )

    axis.set_title(
        f"LightGBM Confusion Matrix\n"
        f"Tuned Threshold = {threshold:.2f}"
    )

    figure.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "lightgbm_confusion_matrix.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        f"[SAVED] {output_path}"
    )


def save_roc_curve(y_true, probability, roc_auc):
    """
    Save ROC curve.

    ROC curve is threshold-independent.
    """

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true,
        probability
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"LightGBM (ROC-AUC = {roc_auc:.4f})"
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
        "LightGBM ROC Curve"
    )

    axis.legend()

    axis.grid(
        alpha=0.3
    )

    figure.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "lightgbm_roc_curve.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        f"[SAVED] {output_path}"
    )


def save_feature_importance(model):
    """
    Save LightGBM feature importance plot.
    """

    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance": importance,
        }
    )

    importance_df = importance_df.sort_values(
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
        "LightGBM Feature Importance"
    )

    figure.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "lightgbm_feature_importance.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(figure)

    print(
        f"[SAVED] {output_path}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("LIGHTGBM TRAINING - YOOCHOOSE PURCHASE INTENT")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------

    print("\n[1] Loading dataset...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_dataset()

    print(
        f"X_train shape : {X_train.shape}"
    )

    print(
        f"X_val shape   : {X_val.shape}"
    )

    print(
        f"X_test shape  : {X_test.shape}"
    )

    print(
        f"Training positive samples : {np.sum(y_train == 1)}"
    )

    print(
        f"Training negative samples : {np.sum(y_train == 0)}"
    )

    print(
        f"Validation positive samples : {np.sum(y_val == 1)}"
    )

    print(
        f"Test positive samples : {np.sum(y_test == 1)}"
    )

    # -------------------------------------------------------------------------
    # 2. CLASS IMBALANCE
    # -------------------------------------------------------------------------

    scale_pos_weight = calculate_scale_pos_weight(
        y_train
    )

    print(
        f"\nCalculated scale_pos_weight : "
        f"{scale_pos_weight:.4f}"
    )

    # -------------------------------------------------------------------------
    # 3. CREATE MODEL
    # -------------------------------------------------------------------------

    print("\n[2] Creating LightGBM model...")

    model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,

        scale_pos_weight=scale_pos_weight,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbosity=-1,
    )

    print("\nModel configuration:")
    print(
        f"n_estimators       = {N_ESTIMATORS}"
    )
    print(
        f"max_depth          = {MAX_DEPTH}"
    )
    print(
        f"learning_rate      = {LEARNING_RATE}"
    )
    print(
        f"num_leaves         = {NUM_LEAVES}"
    )
    print(
        f"subsample          = {SUBSAMPLE}"
    )
    print(
        f"colsample_bytree   = {COLSAMPLE_BYTREE}"
    )
    print(
        f"scale_pos_weight   = {scale_pos_weight:.4f}"
    )
    print(
        f"random_state       = {RANDOM_STATE}"
    )

    # -------------------------------------------------------------------------
    # 4. TRAIN MODEL
    # -------------------------------------------------------------------------

    print("\n[3] Training LightGBM...")

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
        f"\nTraining completed in "
        f"{training_time:.2f} seconds."
    )

    # -------------------------------------------------------------------------
    # 5. VALIDATION-BASED THRESHOLD TUNING
    # -------------------------------------------------------------------------

    (
        tuned_threshold,
        validation_best_f1,
        threshold_results,
    ) = tune_threshold_on_validation(
        model,
        X_val,
        y_val
    )

    # -------------------------------------------------------------------------
    # 6. TEST PREDICTION
    # -------------------------------------------------------------------------

    print("\n[4] Generating test predictions...")

    prediction_start = time.perf_counter()

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    prediction_time = (
        time.perf_counter()
        - prediction_start
    )

    print(
        f"Prediction completed in "
        f"{prediction_time:.4f} seconds."
    )

    # -------------------------------------------------------------------------
    # 7. DEFAULT THRESHOLD = 0.50
    # -------------------------------------------------------------------------

    (
        default_prediction,
        precision_default,
        recall_default,
        f1_default,
    ) = calculate_threshold_metrics(
        y_test,
        test_probability,
        DEFAULT_THRESHOLD
    )

    # -------------------------------------------------------------------------
    # 8. TUNED THRESHOLD
    # -------------------------------------------------------------------------

    (
        tuned_prediction,
        precision_tuned,
        recall_tuned,
        f1_tuned,
    ) = calculate_threshold_metrics(
        y_test,
        test_probability,
        tuned_threshold
    )

    # -------------------------------------------------------------------------
    # 9. THRESHOLD-INDEPENDENT METRICS
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
    # 10. PRINT RESULTS
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    print("\nDEFAULT THRESHOLD = 0.50")

    print(
        f"Precision : {precision_default:.4f}"
    )

    print(
        f"Recall    : {recall_default:.4f}"
    )

    print(
        f"F1-score  : {f1_default:.4f}"
    )

    print("\nTUNED THRESHOLD")

    print(
        f"Threshold : {tuned_threshold:.2f}"
    )

    print(
        f"Precision : {precision_tuned:.4f}"
    )

    print(
        f"Recall    : {recall_tuned:.4f}"
    )

    print(
        f"F1-score  : {f1_tuned:.4f}"
    )

    print("\nTHRESHOLD-INDEPENDENT METRICS")

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print(
        f"\nValidation F1 used for threshold selection : "
        f"{validation_best_f1:.4f}"
    )

    # -------------------------------------------------------------------------
    # 11. CONFUSION MATRICES
    # -------------------------------------------------------------------------

    print("\nDefault threshold confusion matrix:")

    print(
        confusion_matrix(
            y_test,
            default_prediction
        )
    )

    print("\nTuned threshold confusion matrix:")

    print(
        confusion_matrix(
            y_test,
            tuned_prediction
        )
    )

    # -------------------------------------------------------------------------
    # 12. CLASSIFICATION REPORT
    # -------------------------------------------------------------------------

    print("\nClassification Report - Tuned Threshold")

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
    # 13. SAVE MODEL
    # -------------------------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "lightgbm_model.txt"
    )

    model.booster_.save_model(
        model_path
    )

    print(
        f"[SAVED] {model_path}"
    )

    # -------------------------------------------------------------------------
    # 14. SAVE MAIN METRICS
    # -------------------------------------------------------------------------

    # IMPORTANT:
    # The main LightGBM metrics file contains the tuned test result.
    # This keeps compatibility with the existing final_results.py pipeline.

    metrics_df = pd.DataFrame(
        [
            {
                "model": "LightGBM",

                "precision": precision_tuned,

                "recall": recall_tuned,

                "f1_score": f1_tuned,

                "roc_auc": roc_auc,

                "training_time_seconds": training_time,

                "prediction_time_seconds": prediction_time,

                "n_estimators": N_ESTIMATORS,

                "max_depth": MAX_DEPTH,

                "learning_rate": LEARNING_RATE,

                "num_leaves": NUM_LEAVES,

                "scale_pos_weight": scale_pos_weight,

                "feature_count": X_train.shape[1],

                "decision_threshold": tuned_threshold,

                "threshold_selection": "validation_F1",
            }
        ]
    )

    metrics_path = os.path.join(
        METRICS_DIR,
        "lightgbm_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print(
        f"[SAVED] {metrics_path}"
    )

    # -------------------------------------------------------------------------
    # 15. SAVE THRESHOLD COMPARISON
    # -------------------------------------------------------------------------

    threshold_comparison_df = pd.DataFrame(
        [
            {
                "model": "LightGBM",

                "threshold_type": "default_0.5",

                "threshold": DEFAULT_THRESHOLD,

                "precision": precision_default,

                "recall": recall_default,

                "f1_score": f1_default,

                "roc_auc": roc_auc,

                "pr_auc": pr_auc,
            },

            {
                "model": "LightGBM",

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

    threshold_comparison_path = os.path.join(
        METRICS_DIR,
        "lightgbm_metrics_threshold_comparison.csv"
    )

    threshold_comparison_df.to_csv(
        threshold_comparison_path,
        index=False
    )

    print(
        f"[SAVED] {threshold_comparison_path}"
    )

    # -------------------------------------------------------------------------
    # 16. SAVE THRESHOLD SEARCH RESULTS
    # -------------------------------------------------------------------------

    threshold_search_df = pd.DataFrame(
        threshold_results
    )

    threshold_search_path = os.path.join(
        METRICS_DIR,
        "lightgbm_validation_threshold_search.csv"
    )

    threshold_search_df.to_csv(
        threshold_search_path,
        index=False
    )

    print(
        f"[SAVED] {threshold_search_path}"
    )

    # -------------------------------------------------------------------------
    # 17. SAVE FIGURES
    # -------------------------------------------------------------------------

    save_confusion_matrix(
        y_test,
        tuned_prediction,
        tuned_threshold
    )

    save_roc_curve(
        y_test,
        test_probability,
        roc_auc
    )

    save_feature_importance(
        model
    )

    # -------------------------------------------------------------------------
    # 18. FINAL SUMMARY
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("LIGHTGBM TRAINING COMPLETED")
    print("=" * 80)

    print(
        f"\nDefault threshold : {DEFAULT_THRESHOLD:.2f}"
    )

    print(
        f"Tuned threshold   : {tuned_threshold:.2f}"
    )

    print(
        f"\nDefault F1        : {f1_default:.4f}"
    )

    print(
        f"Tuned F1          : {f1_tuned:.4f}"
    )

    print(
        f"\nROC-AUC           : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC            : {pr_auc:.4f}"
    )

    print(
        "\nThreshold was selected using the validation set only."
    )

    print(
        "The test set was used only for final evaluation."
    )

    print(
        "\nNo retraining was performed during threshold tuning."
    )

    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()