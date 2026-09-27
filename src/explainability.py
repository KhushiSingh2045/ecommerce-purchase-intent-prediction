"""
SHAP Tree Explainability
Computes and visualizes feature attributions for XGBoost on real test sessions.
Outputs figure to results/figures/shap_summary_xgboost.png
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost as xgb
from src.train_xgboost import FEATURE_NAMES, create_aggregated_features

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "yoochoose_balanced_train.npz"
MODEL_PATH = BASE_DIR / "models" / "machine_learning" / "xgboost_model.json"
OUTPUT_FIG = BASE_DIR / "results" / "figures" / "shap_summary_xgboost.png"
OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading dataset for SHAP explainability...")
    data = np.load(DATA_PATH, allow_pickle=True)
    test_seq = data["test_sequences"]
    test_deltas = data["test_time_deltas"]

    X_test = create_aggregated_features(test_seq, test_deltas)

    # Sample 300 sessions for clear, rapid explanation calculation
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_test), size=300, replace=False)
    X_sample = X_test[sample_indices]

    print(f"Loading XGBoost model from: {MODEL_PATH}")
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=FEATURE_NAMES,
        show=False,
    )
    plt.title("XGBoost SHAP Feature Importance Summary (Test Sample)", pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {OUTPUT_FIG}")


if __name__ == "__main__":
    main()