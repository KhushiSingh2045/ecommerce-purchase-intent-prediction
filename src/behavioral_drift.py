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
# 5. EFFECT SIZE
# ============================================================

def standardized_mean_difference(group1, group2):
    mean1 = group1.mean()
    mean2 = group2.mean()

    std1 = group1.std()
    std2 = group2.std()

    pooled_std = np.sqrt((std1**2 + std2**2) / 2)

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


# ============================================================
# 6. DRIFT INTERPRETATION
# ============================================================

def interpret_effect(effect):
    absolute_effect = abs(effect)

    if absolute_effect < 0.2:
        return "Small drift"
    elif absolute_effect < 0.5:
        return "Moderate drift"
    elif absolute_effect < 0.8:
        return "Large drift"
    else:
        return "Very large drift"


# ============================================================
# 7. FEATURE DRIFT ANALYSIS
# ============================================================

def analyze_feature_drift(uci_df, yoochoose_df):
    results = []

    for feature in FEATURES:
        uci_values = uci_df[feature]
        yoochoose_values = yoochoose_df[feature]

        # Descriptive statistics
        uci_mean = uci_values.mean()
        yoochoose_mean = yoochoose_values.mean()

        uci_median = uci_values.median()
        yoochoose_median = yoochoose_values.median()

        uci_std = uci_values.std()
        yoochoose_std = yoochoose_values.std()

        # Percentage difference
        if uci_mean != 0:
            mean_difference_percent = (
                (yoochoose_mean - uci_mean) / abs(uci_mean)
            ) * 100
        else:
            mean_difference_percent = np.nan

        # Standardized mean difference
        effect = standardized_mean_difference(uci_values, yoochoose_values)

        # KS statistical test
        ks_statistic, p_value = ks_2samp(uci_values, yoochoose_values)

        if p_value < 0.05:
            statistical_result = "Significant distribution difference"
        else:
            statistical_result = "No statistically significant difference"

        results.append({
            "Feature": feature,
            "UCI_Mean": uci_mean,
            "YOOCHOOSE_Mean": yoochoose_mean,
            "UCI_Median": uci_median,
            "YOOCHOOSE_Median": yoochoose_median,
            "UCI_Std": uci_std,
            "YOOCHOOSE_Std": yoochoose_std,
            "Mean_Difference_Percent": mean_difference_percent,
            "Standardized_Mean_Difference": effect,
            "Drift_Interpretation": interpret_effect(effect),
            "KS_Statistic": ks_statistic,
            "KS_P_Value": p_value,
            "Statistical_Conclusion": statistical_result,
        })

    return pd.DataFrame(results)


# ============================================================
# 8. PURCHASE RATE ANALYSIS
# ============================================================

def compute_rate_drift(
    feature_name,
    p1,
    n1,
    p2,
    n2,
    dataset1_name="UCI",
    dataset2_name="YOOCHOOSE",
):
    """Two-proportion z-test and Cohen's h effect size for comparing rates."""
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z_statistic = (p1 - p2) / se if se > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(abs(z_statistic)))

    cohens_h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    abs_h = abs(cohens_h)
    if abs_h < 0.2:
        interpretation = "Small difference"
    elif abs_h < 0.5:
        interpretation = "Medium difference"
    else:
        interpretation = "Large difference"

    return {
        "Feature": feature_name,
        f"{dataset1_name}_Mean": p1,
        f"{dataset2_name}_Mean": p2,
        f"{dataset1_name}_Median": "N/A (rate, not a distribution)",
        f"{dataset2_name}_Median": "N/A (rate, not a distribution)",
        "KS_Statistic": "N/A (see Z_Statistic)",
        "KS_P_Value": round(float(p_value), 10),
        "Z_Statistic": round(float(z_statistic), 4),
        "Standardized_Mean_Difference": round(float(cohens_h), 4),
        "Drift_Interpretation": interpretation,
    }


def analyze_purchase_rate(uci_df, yoochoose_df):
    """Wrapper function to compute purchase rate drift and format comparison metrics."""
    p1 = float(uci_df[TARGET].mean())
    n1 = len(uci_df)
    p2 = float(yoochoose_df[TARGET].mean())
    n2 = len(yoochoose_df)

    rate_dict = compute_rate_drift("purchase_rate", p1, n1, p2, n2)

    # Add explicit rate keys expected downstream
    rate_dict["UCI_Purchase_Rate"] = p1
    rate_dict["YOOCHOOSE_Purchase_Rate"] = p2

    return pd.DataFrame([rate_dict])


# ============================================================
# 9. CREATE DISTRIBUTION GRAPHS
# ============================================================

def create_distribution_plot(
    uci_df, yoochoose_df, feature, title, filename
):
    plt.figure(figsize=(10, 6))

    sample_size = min(50000, len(yoochoose_df))
    yoochoose_sample = yoochoose_df[feature].sample(
        n=sample_size, random_state=42
    )

    plt.hist(uci_df[feature], bins=50, alpha=0.5, density=True, label="UCI")
    plt.hist(
        yoochoose_sample, bins=50, alpha=0.5, density=True, label="YOOCHOOSE"
    )

    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)

    output_path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure -> {output_path}")


# ============================================================
# 10. PURCHASE RATE GRAPH
# ============================================================

def create_purchase_rate_plot(uci_df, yoochoose_df):
    datasets = ["UCI", "YOOCHOOSE"]
    rates = [uci_df[TARGET].mean() * 100, yoochoose_df[TARGET].mean() * 100]

    plt.figure(figsize=(8, 6))
    plt.bar(datasets, rates, color=["#1f77b4", "#ff7f0e"])
    plt.ylabel("Purchase Rate (%)")
    plt.xlabel("Dataset")
    plt.title("Purchase Rate Comparison")
    plt.grid(axis="y", alpha=0.25)

    for index, value in enumerate(rates):
        plt.text(index, value, f"{value:.2f}%", ha="center", va="bottom")

    output_path = FIGURES_DIR / "behavioral_drift_purchase_rate.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure -> {output_path}")


# ============================================================
# 11. MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("BEHAVIORAL DRIFT ANALYSIS")
    print("=" * 70)

    print("\nLoading datasets...")
    uci_df = load_dataset(UCI_FILE)
    yoochoose_df = load_dataset(YOOCHOOSE_FILE)

    print(f"\nUCI sessions        : {len(uci_df):,}")
    print(f"YOOCHOOSE sessions  : {len(yoochoose_df):,}")

    # Feature distribution drift
    print("\n" + "=" * 70)
    print("FEATURE DISTRIBUTION DRIFT")
    print("=" * 70)

    drift_df = analyze_feature_drift(uci_df, yoochoose_df)
    print(drift_df.round(6).to_string(index=False))

    drift_file = METRICS_DIR / "behavioral_drift_tests.csv"
    drift_df.to_csv(drift_file, index=False)
    print(f"\nSaved -> {drift_file}")

    # Purchase rate drift
    print("\n" + "=" * 70)
    print("PURCHASE RATE DRIFT")
    print("=" * 70)

    purchase_df = analyze_purchase_rate(uci_df, yoochoose_df)
    print(purchase_df.to_string(index=False))

    # Combined summary table
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

    summary_rows.append({
        "Feature": "purchase_rate",
        "UCI_Mean": purchase_df["UCI_Purchase_Rate"].iloc[0],
        "YOOCHOOSE_Mean": purchase_df["YOOCHOOSE_Purchase_Rate"].iloc[0],
        "UCI_Median": np.nan,
        "YOOCHOOSE_Median": np.nan,
        "KS_Statistic": np.nan,
        "KS_P_Value": purchase_df["KS_P_Value"].iloc[0],
        "Standardized_Mean_Difference": purchase_df[
            "Standardized_Mean_Difference"
        ].iloc[0],
        "Drift_Interpretation": purchase_df["Drift_Interpretation"].iloc[0],
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = METRICS_DIR / "behavioral_drift_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved -> {summary_file}")

    # Generate figures
    print("\n" + "=" * 70)
    print("CREATING DRIFT GRAPHS")
    print("=" * 70)

    create_distribution_plot(
        uci_df,
        yoochoose_df,
        "session_click_count",
        "Session Click Count Distribution: UCI vs YOOCHOOSE",
        "behavioral_drift_clicks.png",
    )

    create_distribution_plot(
        uci_df,
        yoochoose_df,
        "session_duration_seconds",
        "Session Duration Distribution: UCI vs YOOCHOOSE",
        "behavioral_drift_duration.png",
    )

    create_distribution_plot(
        uci_df,
        yoochoose_df,
        "unique_items_or_pages",
        "Unique Items/Pages Distribution: UCI vs YOOCHOOSE",
        "behavioral_drift_unique_items.png",
    )

    create_purchase_rate_plot(uci_df, yoochoose_df)

    print("\n" + "=" * 70)
    print("BEHAVIORAL DRIFT ANALYSIS COMPLETED")
    print("=" * 70)
    print("\nGenerated files:")
    print(f" - {drift_file}")
    print(f" - {summary_file}")
    print(f" - {FIGURES_DIR / 'behavioral_drift_clicks.png'}")
    print(f" - {FIGURES_DIR / 'behavioral_drift_duration.png'}")
    print(f" - {FIGURES_DIR / 'behavioral_drift_unique_items.png'}")
    print(f" - {FIGURES_DIR / 'behavioral_drift_purchase_rate.png'}")


# ============================================================
# 12. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()