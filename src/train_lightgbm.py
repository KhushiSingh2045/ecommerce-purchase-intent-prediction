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

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "yoochoose_balanced_train.npz"
)

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
SCALE_POS_WEIGHT = 1.0  # Training data is balanced 50:50


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
# LOAD DATA & FEATURE EXTRACTION
# =============================================================================

def load_data():
    print("\nLoading dataset:")
    print(DATA_PATH)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_PATH}\n\n"
            "Run balance_dataset first:\n"
            "python -m src.balance_dataset"
        )

    data = np.load(DATA_PATH, allow_pickle=True)

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
            raise KeyError(f"Required dataset key is missing: {key}")

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
    print(f"Train sequences       : {train_sequences.shape}")
    print(f"Validation sequences  : {val_sequences.shape}")
    print(f"Test sequences        : {test_sequences.shape}")

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


def create_aggregated_features(sequences, time_deltas):
    """Convert fixed-length sequential sessions into 14 aggregated behavioral features."""
    sequences = np.asarray(sequences, dtype=np.int32)
    time_deltas = np.asarray(time_deltas, dtype=np.float32)

    n_sessions = sequences.shape[0]
    features = np.zeros((n_sessions, 14), dtype=np.float32)

    for i in range(n_sessions):
        items = sequences[i]
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
            unique_item_count = len(np.unique(valid_items))
            repeated_item_count = interaction_count - unique_item_count
            first_item = valid_items[0]
            last_item = valid_items[-1]
            mean_item = np.mean(valid_items)
            std_item = np.std(valid_items)
            min_item = np.min(valid_items)
            max_item = np.max(valid_items)

        deltas = time_deltas[i]
        valid_deltas = deltas[deltas > 0]

        if len(valid_deltas) == 0:
            total_time_delta = 0
            mean_time_delta = 0
            max_time_delta = 0
            std_time_delta = 0
            nonzero_time_intervals = 0
        else:
            total_time_delta = np.sum(valid_deltas)
            mean_time_delta = np.mean(valid_deltas)
            max_time_delta = np.max(valid_deltas)
            std_time_delta = np.std(valid_deltas)
            nonzero_time_intervals = len(valid_deltas)

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
# THRESHOLD TUNING & METRICS HELPERS
# =============================================================================

def tune_threshold_on_validation(model, X_val, y_val):
    print("\n" + "=" * 80)
    print("VALIDATION-BASED THRESHOLD TUNING")
    print("=" * 80)

    print("\nGenerating validation probabilities...")
    validation_probability = model.predict_proba(X_val)[:, 1]

    best_threshold = DEFAULT_THRESHOLD
    best_f1 = -1.0
    threshold_results = []

    for threshold in np.arange(0.05, 0.95, 0.01):
        validation_prediction = (validation_probability >= threshold).astype(int)
        current_f1 = f1_score(y_val, validation_prediction, zero_division=0)

        threshold_results.append(
            {
                "threshold": round(float(threshold), 2),
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
    prediction = (probability >= threshold).astype(int)
    precision = precision_score(y_true, prediction, zero_division=0)
    recall = recall_score(y_true, prediction, zero_division=0)
    f1 = f1_score(y_true, prediction, zero_division=0)
    return prediction, precision, recall, f1


def save_confusion_matrix(y_true, prediction, threshold):
    cm = confusion_matrix(y_true, prediction)
    print("\nConfusion Matrix:")
    print(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No Purchase", "Purchase"]
    )
    display.plot(ax=ax, values_format="d")
    ax.set_title(
        f"LightGBM Confusion Matrix\nTuned Threshold = {threshold:.2f}"
    )
    fig.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "lightgbm_confusion_matrix.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path}")


def save_roc_curve(y_true, probability, roc_auc):
    fpr, tpr, _ = roc_curve(y_true, probability)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"LightGBM (ROC-AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("LightGBM ROC Curve (Balanced Training)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "lightgbm_roc_curve.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path}")


def save_feature_importance(model):
    importance = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": FEATURE_NAMES, "importance": importance}
    ).sort_values("importance", ascending=False)

    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(importance_df["feature"], importance_df["importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("LightGBM Feature Importance (Balanced Training)")
    fig.tight_layout()

    output_path = os.path.join(FIGURES_DIR, "lightgbm_feature_importance.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("LIGHTGBM TRAINING - YOOCHOOSE PURCHASE INTENT (BALANCED DATASET)")
    print("=" * 80)

    # 1. Load Data
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

    # 2. Create Aggregated Features
    print("\nCreating 14 aggregated behavioral features...")
    feature_start = time.perf_counter()

    X_train = create_aggregated_features(train_sequences, train_time_deltas)
    X_val = create_aggregated_features(val_sequences, val_time_deltas)
    X_test = create_aggregated_features(test_sequences, test_time_deltas)

    feature_time = time.perf_counter() - feature_start
    print(f"Feature generation completed in {feature_time:.2f} seconds.")

    print(f"X_train shape : {X_train.shape}")
    print(f"X_val shape   : {X_val.shape}")
    print(f"X_test shape  : {X_test.shape}")

    print(f"\nTraining positive samples   : {np.sum(y_train == 1):,}")
    print(f"Training negative samples   : {np.sum(y_train == 0):,}")
    print(f"Validation positive samples : {np.sum(y_val == 1):,}")
    print(f"Test positive samples       : {np.sum(y_test == 1):,}")
    print(f"Using scale_pos_weight      : {SCALE_POS_WEIGHT:.1f}")

    # 3. Create Model
    print("\n[2] Initializing LightGBM model...")
    model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        scale_pos_weight=SCALE_POS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    # 4. Train Model
    print("\n[3] Training LightGBM...")
    training_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - training_start
    print(f"\nTraining completed in {training_time:.2f} seconds.")

    # 5. Validation-based Threshold Tuning
    (
        tuned_threshold,
        validation_best_f1,
        threshold_results,
    ) = tune_threshold_on_validation(model, X_val, y_val)

    # 6. Test Prediction
    print("\n[4] Generating test predictions...")
    prediction_start = time.perf_counter()
    test_probability = model.predict_proba(X_test)[:, 1]
    prediction_time = time.perf_counter() - prediction_start
    print(f"Prediction completed in {prediction_time:.4f} seconds.")

    # 7. Default Threshold = 0.50 Metrics
    (
        default_prediction,
        precision_default,
        recall_default,
        f1_default,
    ) = calculate_threshold_metrics(
        y_test, test_probability, DEFAULT_THRESHOLD
    )

    # 8. Tuned Threshold Metrics
    (
        tuned_prediction,
        precision_tuned,
        recall_tuned,
        f1_tuned,
    ) = calculate_threshold_metrics(
        y_test, test_probability, tuned_threshold
    )

    # 9. Threshold-Independent Metrics
    roc_auc = roc_auc_score(y_test, test_probability)
    pr_auc = average_precision_score(y_test, test_probability)

    # 10. Print Results
    print("\n" + "=" * 80)
    print("TEST RESULTS (NATURAL TEST DISTRIBUTION)")
    print("=" * 80)
    print("\nDEFAULT THRESHOLD = 0.50")
    print(f"Precision : {precision_default:.4f}")
    print(f"Recall    : {recall_default:.4f}")
    print(f"F1-score  : {f1_default:.4f}")

    print("\nTUNED THRESHOLD")
    print(f"Threshold : {tuned_threshold:.2f}")
    print(f"Precision : {precision_tuned:.4f}")
    print(f"Recall    : {recall_tuned:.4f}")
    print(f"F1-score  : {f1_tuned:.4f}")

    print("\nTHRESHOLD-INDEPENDENT METRICS")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")
    print(f"\nValidation F1 used for threshold selection : {validation_best_f1:.4f}")

    # 11. Confusion Matrices
    print("\nDefault threshold confusion matrix:")
    print(confusion_matrix(y_test, default_prediction))

    print("\nTuned threshold confusion matrix:")
    print(confusion_matrix(y_test, tuned_prediction))

    # 12. Classification Report
    print("\nClassification Report - Tuned Threshold:")
    print(
        classification_report(
            y_test,
            tuned_prediction,
            target_names=["No Purchase", "Purchase"],
            zero_division=0,
        )
    )

    # 13. Save Model
    model_path = os.path.join(MODEL_DIR, "lightgbm_model.txt")
    model.booster_.save_model(model_path)
    print(f"[SAVED] {model_path}")

    # 14. Save Main Metrics
    metrics_df = pd.DataFrame(
        [
            {
                "model": "LightGBM",
                "precision": precision_tuned,
                "recall": recall_tuned,
                "f1_score": f1_tuned,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "training_time_seconds": training_time,
                "prediction_time_seconds": prediction_time,
                "n_estimators": N_ESTIMATORS,
                "max_depth": MAX_DEPTH,
                "learning_rate": LEARNING_RATE,
                "num_leaves": NUM_LEAVES,
                "scale_pos_weight": SCALE_POS_WEIGHT,
                "feature_count": X_train.shape[1],
                "decision_threshold": tuned_threshold,
                "threshold_selection": "validation_F1",
            }
        ]
    )
    metrics_path = os.path.join(METRICS_DIR, "lightgbm_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[SAVED] {metrics_path}")

    # 15. Save Threshold Comparison
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
        METRICS_DIR, "lightgbm_metrics_threshold_comparison.csv"
    )
    threshold_comparison_df.to_csv(threshold_comparison_path, index=False)
    print(f"[SAVED] {threshold_comparison_path}")

    # 16. Save Threshold Search Results
    threshold_search_df = pd.DataFrame(threshold_results)
    threshold_search_path = os.path.join(
        METRICS_DIR, "lightgbm_validation_threshold_search.csv"
    )
    threshold_search_df.to_csv(threshold_search_path, index=False)
    print(f"[SAVED] {threshold_search_path}")

    # 17. Save Figures
    save_confusion_matrix(y_test, tuned_prediction, tuned_threshold)
    save_roc_curve(y_test, test_probability, roc_auc)
    save_feature_importance(model)

    print("\n" + "=" * 80)
    print("LIGHTGBM TRAINING COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()