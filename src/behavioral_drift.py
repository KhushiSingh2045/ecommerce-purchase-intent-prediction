from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, norm

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DATASET PATHS
# ============================================================

UCI_FILE = PROCESSED_DIR / "cross_dataset_uci.csv"
YOOCHOOSE_FILE = PROCESSED_DIR / "cross_dataset_yoochoose.csv"


# ============================================================
# 3. FEATURES
# ============================================================

FEATURES = [
    "session_click_count",
    "session_duration_seconds",
    "unique_items_or_pages",
]

TARGET = "purchase"


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_dataset(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found:\n{file_path}")

    df = pd.read_csv(file_path)
    required_columns = FEATURES + [TARGET]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_path.name}: {missing}")

    df = df[required_columns].copy()
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required_columns)
    df[TARGET] = df[TARGET].astype(int)
    return df


# ============================================================
# 5. STATISTICAL HELPERS
# ============================================================

def standardized_mean_difference(group1, group2):
    mean1 = group1.mean()
    mean2 = group2.mean()
    std1 = group1.std()
    std2 = group2.std()
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    return 0.0 if pooled_std == 0 else float((mean1 - mean2) / pooled_std)


def interpret_effect(effect):
    abs_effect = abs(effect)
    if abs_effect < 0.2:
        return "Small drift"
    elif abs_effect < 0.5:
        return "Moderate drift"
    elif abs_effect < 0.8:
        return "Large drift"
    return "Very large drift"


# ============================================================
# 6. FEATURE DRIFT ANALYSIS
# ============================================================

def analyze_feature_drift(uci_df, yoochoose_df):
    results = []
    for feature in FEATURES:
        uci_v = uci_df[feature]
        yoo_v = yoochoose_df[feature]

        diff_pct = (
            ((yoo_v.mean() - uci_v.mean()) / abs(uci_v.mean())) * 100
            if uci_v.mean() != 0
            else 0.0
        )
        effect = standardized_mean_difference(uci_v, yoo_v)
        ks_stat, p_val = ks_2samp(uci_v, yoo_v)

        results.append({
            "Feature": feature,
            "UCI_Mean": float(uci_v.mean()),
            "YOOCHOOSE_Mean": float(yoo_v.mean()),
            "UCI_Median": float(uci_v.median()),
            "YOOCHOOSE_Median": float(yoo_v.median()),
            "UCI_Std": float(uci_v.std()),
            "YOOCHOOSE_Std": float(yoo_v.std()),
            "Mean_Difference_Percent": float(diff_pct),
            "Standardized_Mean_Difference": float(effect),
            "Drift_Interpretation": interpret_effect(effect),
            "KS_Statistic": float(ks_stat),
            "KS_P_Value": float(p_val),
            "Statistical_Conclusion": (
                "Significant distribution difference"
                if p_val < 0.05
                else "No significant difference"
            ),
        })
    return pd.DataFrame(results)


# ============================================================
# 7. PURCHASE RATE ANALYSIS
# ============================================================

def analyze_purchase_rate(uci_df, yoochoose_df):
    p1 = float(uci_df[TARGET].mean())
    n1 = len(uci_df)
    p2 = float(yoochoose_df[TARGET].mean())
    n2 = len(yoochoose_df)

    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z_stat = (p1 - p2) / se if se > 0 else 0.0
    p_val = 2 * (1 - norm.cdf(abs(z_stat)))

    cohens_h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

    return {
        "Feature": "purchase_rate",
        "UCI_Mean": p1,
        "YOOCHOOSE_Mean": p2,
        "UCI_Median": float(uci_df[TARGET].median()),
        "YOOCHOOSE_Median": float(yoochoose_df[TARGET].median()),
        "KS_Statistic": float(abs(p1 - p2)),  # Absolute rate difference (non-null metric)
        "KS_P_Value": float(p_val),
        "Standardized_Mean_Difference": float(cohens_h),
        "Drift_Interpretation": interpret_effect(cohens_h),
    }


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("Running Behavioral Drift Analysis...")
    uci_df = load_dataset(UCI_FILE)
    yoochoose_df = load_dataset(YOOCHOOSE_FILE)

    drift_df = analyze_feature_drift(uci_df, yoochoose_df)
    drift_file = METRICS_DIR / "behavioral_drift_tests.csv"
    drift_df.to_csv(drift_file, index=False)

    rate_dict = analyze_purchase_rate(uci_df, yoochoose_df)

    # Combined Summary: Clean and completely free of NaNs
    summary_rows = []
    for feature in FEATURES:
        row = drift_df[drift_df["Feature"] == feature].iloc[0]
        summary_rows.append({
            "Feature": feature,
            "UCI_Mean": row["UCI_Mean"],
            "YOOCHOOSE_Mean": row["YOOCHOOSE_Mean"],
            "UCI_Median": row["UCI_Median"],
            "YOOCHOOSE_Median": row["YOOCHOOSE_Median"],
            "KS_Statistic": row["KS_Statistic"],
            "KS_P_Value": row["KS_P_Value"],
            "Standardized_Mean_Difference": row["Standardized_Mean_Difference"],
            "Drift_Interpretation": row["Drift_Interpretation"],
        })
    summary_rows.append(rate_dict)

    summary_df = pd.DataFrame(summary_rows)
    summary_file = METRICS_DIR / "behavioral_drift_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"[SAVED] {summary_file}")


if __name__ == "__main__":
    main()