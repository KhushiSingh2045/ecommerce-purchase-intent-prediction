"""
Trains three models on the cleaned, aggregated UCI Online Shoppers data:
Logistic Regression (baseline), XGBoost, and LightGBM. Saves each trained
model, plus the held-out test set for evaluation.py to use later.
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

PROCESSED_PATH = "data/processed/shoppers_clean.csv"
TEST_SET_PATH = "data/processed/test_set.csv"
TARGET_COLUMN = "Revenue"


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # First split off the test set (20%) - this stays completely unseen
    # until evaluation.py runs.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # From the remaining 80%, split off a validation set (10% of the original)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_class_weights(y_train: pd.Series) -> dict:
    counts = y_train.value_counts()
    total = len(y_train)
    weights = {cls: total / (len(counts) * n) for cls, n in counts.items()}
    print(f"Class weights: {weights}")
    return weights


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_PATH)
    print(f"Loaded {len(df)} cleaned rows from {PROCESSED_PATH}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    class_weights = compute_class_weights(y_train)
    sample_weights = y_train.map(class_weights).values

    os.makedirs("models/baseline", exist_ok=True)
    os.makedirs("models/machine_learning", exist_ok=True)

    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    joblib.dump(lr, "models/baseline/logistic_regression.joblib")
    print("Saved -> models/baseline/logistic_regression.joblib")

    print("\nTraining XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="auc", random_state=42
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    joblib.dump(xgb, "models/machine_learning/xgboost.joblib")
    print("Saved -> models/machine_learning/xgboost.joblib")

    print("\nTraining LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=300, max_depth=-1, learning_rate=0.05, num_leaves=31,
        random_state=42, verbose=-1
    )
    lgbm.fit(X_train, y_train, sample_weight=sample_weights)
    joblib.dump(lgbm, "models/machine_learning/lightgbm.joblib")
    print("Saved -> models/machine_learning/lightgbm.joblib")

    # Save the test set so evaluation.py always evaluates on the exact
    # same unseen data these models never saw during training.
    X_test.assign(**{TARGET_COLUMN: y_test}).to_csv(TEST_SET_PATH, index=False)
    print(f"\nSaved test set -> {TEST_SET_PATH}")