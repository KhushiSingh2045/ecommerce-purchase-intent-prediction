"""
Train LightGBM Model with Robust Binary Persistence (joblib)
Trains on tabular features from yoochoose_ml_temporal.npz.
Tunes decision threshold on validation set; evaluates once on test set.
Outputs artifacts to models/machine_learning/lightgbm.joblib and lightgbm_model.txt.
"""

from pathlib import Path
import time
import joblib
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

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "yoochoose_ml_temporal.npz"
MODELS_DIR = BASE_DIR / "models" / "machine_learning"
RESULTS_DIR = BASE_DIR / "results" / "metrics"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_JOBLIB_PATH = MODELS_DIR / "lightgbm.joblib"
MODEL_TXT_PATH = MODELS_DIR / "lightgbm_model.txt"
METRICS_OUTPUT_PATH = RESULTS_DIR / "lightgbm_metrics.csv"
SEARCH_OUTPUT_PATH = RESULTS_DIR / "lightgbm_validation_threshold_search.csv"
THRESH_COMP_PATH = RESULTS_DIR / "lightgbm_metrics_threshold_comparison.csv"


def find_optimal_threshold(y_true, probabilities):
    best_f1 = 0.0
    best_threshold = 0.50
    search_records = []

    for t in np.linspace(0.05, 0.95, 91):
        t = round(float(t), 2)
        preds = (probabilities >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        search_records.append({"threshold": t, "f1_score": float(score)})
        if score > best_f1:
            best_f1 = score
            best_threshold = t

    return best_threshold, best_f1, pd.DataFrame(search_records)


def main():
    print("=" * 70)
    print("LIGHTGBM ROBUST TRAINING PIPELINE (JOBLIB SERIALIZATION)")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"ML dataset missing: {DATA_PATH}. Run prepare_ml_dataset.py first.")

    data = np.load(DATA_PATH, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    feature_names = list(data["feature_names"])

    print(f"Loaded tabular features: {len(feature_names)} features")
    print(f"  Train : {X_train.shape[0]:,} sessions")
    print(f"  Val   : {X_val.shape[0]:,} sessions")
    print(f"  Test  : {X_test.shape[0]:,} sessions")

    trn_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, free_raw_data=False)
    val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data, feature_name=feature_names, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "scale_pos_weight": 1.0,
        "random_state": 42,
        "verbose": -1,
    }

    print("\nTraining LightGBM model...")
    t0_train = time.perf_counter()
    bst = lgb.train(
        params,
        trn_data,
        num_boost_round=300,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )
    train_time = time.perf_counter() - t0_train

    # Validation Threshold Search
    val_probs = bst.predict(X_val)
    opt_thresh, opt_val_f1, search_df = find_optimal_threshold(y_val, val_probs)
    search_df.to_csv(SEARCH_OUTPUT_PATH, index=False)
    print(f"\nOptimal threshold tuned on validation: tau={opt_thresh:.2f} (Val F1={opt_val_f1:.4f})")

    # Final Evaluation on Test Set
    t0_pred = time.perf_counter()
    test_probs = bst.predict(X_test)
    pred_time = time.perf_counter() - t0_pred

    test_preds_opt = (test_probs >= opt_thresh).astype(int)
    test_preds_default = (test_probs >= 0.50).astype(int)

    test_acc = accuracy_score(y_test, test_preds_opt)
    test_prec = precision_score(y_test, test_preds_opt, zero_division=0)
    test_rec = recall_score(y_test, test_preds_opt, zero_division=0)
    test_f1 = f1_score(y_test, test_preds_opt, zero_division=0)
    test_auc = roc_auc_score(y_test, test_probs)
    test_prauc = average_precision_score(y_test, test_probs)

    print("\n" + "=" * 50)
    print(f"TEST RESULTS (tau={opt_thresh:.2f})")
    print("=" * 50)
    print(f"ROC-AUC    : {test_auc:.4f}")
    print(f"PR-AUC     : {test_prauc:.4f}")
    print(f"Precision  : {test_prec:.4f}")
    print(f"Recall     : {test_rec:.4f}")
    print(f"F1-Score   : {test_f1:.4f}")
    print(f"Accuracy   : {test_acc:.4f}")
    print(f"Inference  : {pred_time*1000:.2f} ms total")

    # Save via Binary Joblib (Immune to text corruption)
    joblib.dump(bst, MODEL_JOBLIB_PATH)
    print(f"\n[SAVED] Binary model artifact: {MODEL_JOBLIB_PATH}")

    # Also save text version with explicit unix linebreaks
    if MODEL_TXT_PATH.exists():
        MODEL_TXT_PATH.unlink()
    bst.save_model(str(MODEL_TXT_PATH))

    # Metrics CSV
    metrics_record = pd.DataFrame([{
        "model": "LightGBM",
        "precision": float(test_prec),
        "recall": float(test_rec),
        "f1_score": float(test_f1),
        "roc_auc": float(test_auc),
        "pr_auc": float(test_prauc),
        "training_time_seconds": float(train_time),
        "prediction_time_seconds": float(pred_time),
        "n_estimators": bst.best_iteration if bst.best_iteration else 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "scale_pos_weight": 1.0,
        "feature_count": len(feature_names),
        "decision_threshold": float(opt_thresh),
        "threshold_selection": "validation_grid_search",
    }])
    metrics_record.to_csv(METRICS_OUTPUT_PATH, index=False)

    # Threshold Comparison CSV
    thresh_comp = pd.DataFrame([
        {
            "model": "LightGBM",
            "threshold_type": "default_0.50",
            "threshold": 0.50,
            "precision": float(precision_score(y_test, test_preds_default, zero_division=0)),
            "recall": float(recall_score(y_test, test_preds_default, zero_division=0)),
            "f1_score": float(f1_score(y_test, test_preds_default, zero_division=0)),
            "roc_auc": float(test_auc),
            "pr_auc": float(test_prauc),
        },
        {
            "model": "LightGBM",
            "threshold_type": "tuned_validation",
            "threshold": float(opt_thresh),
            "precision": float(test_prec),
            "recall": float(test_rec),
            "f1_score": float(test_f1),
            "roc_auc": float(test_auc),
            "pr_auc": float(test_prauc),
        },
    ])
    thresh_comp.to_csv(THRESH_COMP_PATH, index=False)
    print("LIGHTGBM TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()