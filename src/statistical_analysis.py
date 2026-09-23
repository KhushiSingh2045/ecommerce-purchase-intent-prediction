from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    mannwhitneyu,
    chi2_contingency,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed"
)

RESULTS_DIR = (
    PROJECT_ROOT /
    "results" /
    "metrics"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. DATASET FILES
# ============================================================

UCI_FILE = (
    PROCESSED_DIR /
    "cross_dataset_uci.csv"
)

YOOCHOOSE_FILE = (
    PROCESSED_DIR /
    "cross_dataset_yoochoose.csv"
)


# ============================================================
# 3. FEATURES
# ============================================================

FEATURES = [
    "session_click_count",
    "session_duration_seconds",
    "unique_items_or_pages",
]

TARGET = "purchase"

ALPHA = 0.05


# ============================================================
# 4. LOAD DATASET
# ============================================================

def load_dataset(file_path):

    if not file_path.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = (
        FEATURES +
        [TARGET]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns in "
            f"{file_path.name}: "
            f"{missing_columns}"
        )

    df = df[
        required_columns
    ].copy()

    for column in required_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_columns
    )

    df[TARGET] = df[
        TARGET
    ].astype(int)

    return df


# ============================================================
# 5. MANN-WHITNEY EFFECT SIZE
# ============================================================

def rank_biserial_effect_size(
    u_statistic,
    n1,
    n2
):
    """
    Rank-biserial correlation based on the
    Mann-Whitney U statistic.

    Range:
        -1 to +1

    Values closer to 0 indicate a smaller
    distributional difference.
    """

    if n1 == 0 or n2 == 0:

        return np.nan

    return (
        (2 * u_statistic)
        / (n1 * n2)
    ) - 1


# ============================================================
# 6. INTERPRET EFFECT SIZE
# ============================================================

def interpret_rank_biserial(effect):

    absolute_effect = abs(effect)

    if absolute_effect < 0.10:
        return "Negligible"

    elif absolute_effect < 0.30:
        return "Small"

    elif absolute_effect < 0.50:
        return "Moderate"

    else:
        return "Large"


# ============================================================
# 7. MANN-WHITNEY TEST
# ============================================================

def perform_mann_whitney_test(
    uci_df,
    yoochoose_df,
    feature
):

    uci_values = (
        uci_df[feature]
        .to_numpy()
    )

    yoochoose_values = (
        yoochoose_df[feature]
        .to_numpy()
    )

    # Two-sided test
    u_statistic, p_value = mannwhitneyu(
        uci_values,
        yoochoose_values,
        alternative="two-sided"
    )

    effect_size = rank_biserial_effect_size(
        u_statistic,
        len(uci_values),
        len(yoochoose_values)
    )

    if p_value < ALPHA:

        conclusion = (
            "Reject H0: "
            "significant distribution difference"
        )

    else:

        conclusion = (
            "Fail to reject H0: "
            "no statistically significant difference"
        )

    return {

        "Hypothesis":
            f"H_{FEATURES.index(feature) + 1}",

        "Test":
            "Mann-Whitney U",

        "Feature":
            feature,

        "UCI_N":
            len(uci_values),

        "YOOCHOOSE_N":
            len(yoochoose_values),

        "U_Statistic":
            u_statistic,

        "P_Value":
            p_value,

        "Alpha":
            ALPHA,

        "Significant":
            p_value < ALPHA,

        "Rank_Biserial_Effect":
            effect_size,

        "Effect_Interpretation":
            interpret_rank_biserial(
                effect_size
            ),

        "Conclusion":
            conclusion,
    }


# ============================================================
# 8. PURCHASE RATE CHI-SQUARE TEST
# ============================================================

def perform_purchase_chi_square(
    uci_df,
    yoochoose_df
):

    uci_purchase = (
        uci_df[TARGET]
        .value_counts()
        .reindex(
            [0, 1],
            fill_value=0
        )
    )

    yoochoose_purchase = (
        yoochoose_df[TARGET]
        .value_counts()
        .reindex(
            [0, 1],
            fill_value=0
        )
    )

    contingency_table = np.array([
        [
            uci_purchase[0],
            uci_purchase[1]
        ],
        [
            yoochoose_purchase[0],
            yoochoose_purchase[1]
        ],
    ])

    chi2, p_value, degrees_of_freedom, expected = (
        chi2_contingency(
            contingency_table
        )
    )

    total = contingency_table.sum()

    phi = np.sqrt(
        chi2 / total
    )

    if p_value < ALPHA:

        conclusion = (
            "Reject H0: "
            "purchase rate differs significantly"
        )

    else:

        conclusion = (
            "Fail to reject H0: "
            "no statistically significant purchase-rate difference"
        )

    return {

        "Hypothesis":
            "H4",

        "Test":
            "Chi-square test of independence",

        "Feature":
            "purchase",

        "UCI_No_Purchase":
            int(uci_purchase[0]),

        "UCI_Purchase":
            int(uci_purchase[1]),

        "YOOCHOOSE_No_Purchase":
            int(yoochoose_purchase[0]),

        "YOOCHOOSE_Purchase":
            int(yoochoose_purchase[1]),

        "Chi2_Statistic":
            chi2,

        "Degrees_of_Freedom":
            degrees_of_freedom,

        "P_Value":
            p_value,

        "Alpha":
            ALPHA,

        "Significant":
            p_value < ALPHA,

        "Phi_Effect_Size":
            phi,

        "Conclusion":
            conclusion,
    }


# ============================================================
# 9. DESCRIPTIVE STATISTICS
# ============================================================

def create_descriptive_summary(
    uci_df,
    yoochoose_df
):

    rows = []

    for feature in FEATURES:

        uci_values = uci_df[feature]
        yoochoose_values = yoochoose_df[feature]

        rows.append({

            "Feature":
                feature,

            "UCI_Mean":
                uci_values.mean(),

            "YOOCHOOSE_Mean":
                yoochoose_values.mean(),

            "UCI_Median":
                uci_values.median(),

            "YOOCHOOSE_Median":
                yoochoose_values.median(),

            "UCI_Std":
                uci_values.std(),

            "YOOCHOOSE_Std":
                yoochoose_values.std(),

            "UCI_Min":
                uci_values.min(),

            "YOOCHOOSE_Min":
                yoochoose_values.min(),

            "UCI_Max":
                uci_values.max(),

            "YOOCHOOSE_Max":
                yoochoose_values.max(),
        })

    return pd.DataFrame(rows)


# ============================================================
# 10. MAIN ANALYSIS
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("STATISTICAL / HYPOTHESIS ANALYSIS")
    print("=" * 70)

    print("\nSignificance level:")
    print(f"alpha = {ALPHA}")

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    uci_df = load_dataset(
        UCI_FILE
    )

    yoochoose_df = load_dataset(
        YOOCHOOSE_FILE
    )

    print(
        f"UCI sessions       : "
        f"{len(uci_df)}"
    )

    print(
        f"YOOCHOOSE sessions : "
        f"{len(yoochoose_df)}"
    )

    # --------------------------------------------------------
    # Descriptive statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)

    descriptive_df = create_descriptive_summary(
        uci_df,
        yoochoose_df
    )

    print(
        descriptive_df.round(4).to_string(
            index=False
        )
    )

    descriptive_file = (
        RESULTS_DIR /
        "statistical_descriptive_summary.csv"
    )

    descriptive_df.to_csv(
        descriptive_file,
        index=False
    )

    # --------------------------------------------------------
    # Mann-Whitney tests
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MANN-WHITNEY U TESTS")
    print("=" * 70)

    mann_whitney_results = []

    for feature in FEATURES:

        print(
            f"\nTesting feature: "
            f"{feature}"
        )

        result = perform_mann_whitney_test(
            uci_df,
            yoochoose_df,
            feature
        )

        mann_whitney_results.append(
            result
        )

        print(
            f"U statistic : "
            f"{result['U_Statistic']:.4f}"
        )

        print(
            f"P-value     : "
            f"{result['P_Value']:.6e}"
        )

        print(
            f"Effect size : "
            f"{result['Rank_Biserial_Effect']:.4f}"
        )

        print(
            f"Interpretation: "
            f"{result['Effect_Interpretation']}"
        )

        print(
            f"Conclusion   : "
            f"{result['Conclusion']}"
        )

    mann_whitney_df = pd.DataFrame(
        mann_whitney_results
    )

    # --------------------------------------------------------
    # Purchase hypothesis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PURCHASE-RATE HYPOTHESIS")
    print("=" * 70)

    purchase_result = (
        perform_purchase_chi_square(
            uci_df,
            yoochoose_df
        )
    )

    print(
        f"\nChi-square statistic : "
        f"{purchase_result['Chi2_Statistic']:.4f}"
    )

    print(
        f"Degrees of freedom   : "
        f"{purchase_result['Degrees_of_Freedom']}"
    )

    print(
        f"P-value              : "
        f"{purchase_result['P_Value']:.6e}"
    )

    print(
        f"Phi effect size      : "
        f"{purchase_result['Phi_Effect_Size']:.4f}"
    )

    print(
        f"Conclusion            : "
        f"{purchase_result['Conclusion']}"
    )

    # --------------------------------------------------------
    # Save hypothesis results
    # --------------------------------------------------------

    purchase_df = pd.DataFrame(
        [purchase_result]
    )

    # Keep common columns separately
    mann_whitney_output = mann_whitney_df.copy()

    mann_whitney_file = (
        RESULTS_DIR /
        "hypothesis_mann_whitney_results.csv"
    )

    mann_whitney_output.to_csv(
        mann_whitney_file,
        index=False
    )

    purchase_file = (
        RESULTS_DIR /
        "hypothesis_purchase_chi_square.csv"
    )

    purchase_df.to_csv(
        purchase_file,
        index=False
    )

    # --------------------------------------------------------
    # Create combined hypothesis summary
    # --------------------------------------------------------

    summary_rows = []

    for _, row in mann_whitney_df.iterrows():

        summary_rows.append({

            "Hypothesis":
                row["Hypothesis"],

            "Feature":
                row["Feature"],

            "Statistical_Test":
                row["Test"],

            "Test_Statistic":
                row["U_Statistic"],

            "P_Value":
                row["P_Value"],

            "Effect_Size":
                row[
                    "Rank_Biserial_Effect"
                ],

            "Effect_Interpretation":
                row[
                    "Effect_Interpretation"
                ],

            "Significant":
                row["Significant"],

            "Conclusion":
                row["Conclusion"],
        })

    summary_rows.append({

        "Hypothesis":
            purchase_result["Hypothesis"],

        "Feature":
            "purchase rate",

        "Statistical_Test":
            purchase_result["Test"],

        "Test_Statistic":
            purchase_result["Chi2_Statistic"],

        "P_Value":
            purchase_result["P_Value"],

        "Effect_Size":
            purchase_result[
                "Phi_Effect_Size"
            ],

        "Effect_Interpretation":
            "Phi coefficient",

        "Significant":
            purchase_result["Significant"],

        "Conclusion":
            purchase_result["Conclusion"],
    })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_file = (
        RESULTS_DIR /
        "statistical_hypothesis_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS COMPLETED")
    print("=" * 70)

    print("\nHypothesis Summary:")

    print(
        summary_df.round(6).to_string(
            index=False
        )
    )

    print("\nGenerated files:")

    print(
        f" - {descriptive_file}"
    )

    print(
        f" - {mann_whitney_file}"
    )

    print(
        f" - {purchase_file}"
    )

    print(
        f" - {summary_file}"
    )

    print("\nInterpretation rule:")

    print(
        "If p-value < 0.05, reject the null hypothesis."
    )

    print(
        "If p-value >= 0.05, fail to reject the null hypothesis."
    )


# ============================================================
# 11. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()