"""
Explains WHY the model predicts what it predicts, using SHAP - not just
whether it's accurate. Uses XGBoost (your best-performing model) and
saves a summary plot showing which features push predictions toward
"Purchase" vs. "No Purchase".
"""
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

TEST_SET_PATH = "data/processed/test_set.csv"
MODEL_PATH = "models/machine_learning/xgboost.joblib"
OUTPUT_PATH = "results/figures/shap_summary_xgboost.png"
SAMPLE_SIZE = 200


if __name__ == "__main__":
    test_df = pd.read_csv(TEST_SET_PATH)
    X_test = test_df.drop(columns=["Revenue"])
    print(f"Loaded {len(test_df)} test rows from {TEST_SET_PATH}")

    model = joblib.load(MODEL_PATH)
    sample = X_test.sample(min(SAMPLE_SIZE, len(X_test)), random_state=42)
    print(f"Explaining predictions on a sample of {len(sample)} sessions...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {OUTPUT_PATH}")