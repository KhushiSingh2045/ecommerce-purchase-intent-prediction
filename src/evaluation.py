"""
Evaluates the trained models (Logistic Regression, XGBoost, LightGBM) on
the held-out test set, using Precision, Recall, F1, and ROC-AUC - the
metrics specified in the methodology, since accuracy alone is misleading
under class imbalance.
"""
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

TEST_SET_PATH = "data/processed/test_set.csv"
TARGET_COLUMN = "Revenue"

MODEL_PATHS = {
    "logistic_regression": "models/baseline/logistic_regression.joblib",
    "xgboost": "models/machine_learning/xgboost.joblib",
    "lightgbm": "models/machine_learning/lightgbm.joblib",
}


def evaluate_model(model, X_test, y_test) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


if __name__ == "__main__":
    test_df = pd.read_csv(TEST_SET_PATH)
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    print(f"Loaded {len(test_df)} test rows from {TEST_SET_PATH}")

    results = {}
    for name, path in MODEL_PATHS.items():
        model = joblib.load(path)
        results[name] = evaluate_model(model, X_test, y_test)

    comparison = pd.DataFrame(results).T
    print("\nComparative Evaluation")
    print(comparison.to_string())

    comparison.to_csv("results/metrics/comparative_evaluation.csv")
    print("\nSaved -> results/metrics/comparative_evaluation.csv")