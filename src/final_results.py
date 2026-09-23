from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
OUTPUT_DIR = PROJECT_ROOT / "results" / "final_tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(filename):
    path = METRICS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path)


def format_p_value(value):
    try:
        value = float(value)

        if value < 0.001:
            return "<0.001"

        return f"{value:.4f}"

    except Exception:
        return value


def save_table(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}")


# ============================================================
# TABLE 1 - MODEL COMPARISON
# ============================================================

def create_model_comparison():

    files = [
        ("LSTM", "lstm_metrics.csv"),
        ("GRU", "gru_metrics.csv"),
        ("GRU + Attention", "gru_attention_metrics.csv"),
        ("XGBoost", "xgboost_metrics.csv"),
        ("LightGBM", "lightgbm_metrics.csv"),
    ]

    rows = []

    for model_name, filename in files:

        df = load_csv(filename)

        row = df.iloc[0]

        rows.append({
            "Model": model_name,
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1_Score": row["f1_score"],
            "ROC_AUC": row["roc_auc"],
        })

    result = pd.DataFrame(rows)

    save_table(
        result,
        "Table_1_Model_Comparison.csv"
    )

    return result


# ============================================================
# TABLE 2 - EARLY PREDICTION
# ============================================================

def create_early_prediction():

    df = load_csv("early_prediction_metrics.csv")

    result = df.copy()

    result["precision"] = result["precision"].round(4)
    result["recall"] = result["recall"].round(4)
    result["f1_score"] = result["f1_score"].round(4)
    result["roc_auc"] = result["roc_auc"].round(4)

    result = result.rename(
        columns={
            "model": "Model",
            "session_percentage": "Observed_Session_Percentage",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1_Score",
            "roc_auc": "ROC_AUC",
            "prediction_time_seconds": "Prediction_Time_Seconds",
        }
    )

    save_table(
        result,
        "Table_2_Early_Prediction.csv"
    )

    return result


# ============================================================
# TABLE 3 - ANONYMOUS USER PREDICTION
# ============================================================

def create_anonymous_prediction():

    df = load_csv("anonymous_user_prediction.csv")

    result = df.copy()

    result = result.rename(
        columns={
            "model": "Model",
            "anonymous_sessions": "Anonymous_Sessions",
            "predicted_purchase": "Predicted_Purchase",
            "predicted_no_purchase": "Predicted_No_Purchase",
            "mean_purchase_probability": "Mean_Purchase_Probability",
            "predicted_purchase_rate": "Predicted_Purchase_Rate",
            "prediction_time_seconds": "Prediction_Time_Seconds",
        }
    )

    result["Mean_Purchase_Probability"] = (
        result["Mean_Purchase_Probability"].round(4)
    )

    result["Predicted_Purchase_Rate"] = (
        result["Predicted_Purchase_Rate"].round(4)
    )

    result["Prediction_Time_Seconds"] = (
        result["Prediction_Time_Seconds"].round(4)
    )

    save_table(
        result,
        "Table_3_Anonymous_User_Prediction.csv"
    )

    return result


# ============================================================
# TABLE 4 - CROSS DATASET VALIDATION
# ============================================================

def create_cross_dataset():

    df = load_csv("cross_dataset_results.csv")

    result = df.copy()

    result["Precision"] = result["Precision"].round(4)
    result["Recall"] = result["Recall"].round(4)
    result["F1_Score"] = result["F1_Score"].round(4)
    result["ROC_AUC"] = result["ROC_AUC"].round(4)
    result["Accuracy"] = result["Accuracy"].round(4)

    save_table(
        result,
        "Table_4_Cross_Dataset_Validation.csv"
    )

    return result


# ============================================================
# TABLE 5 - BEHAVIORAL DRIFT
# ============================================================

def create_behavioral_drift():

    df = load_csv("behavioral_drift_tests.csv")

    result = df[
        [
            "Feature",
            "UCI_Mean",
            "YOOCHOOSE_Mean",
            "UCI_Median",
            "YOOCHOOSE_Median",
            "Mean_Difference_Percent",
            "Standardized_Mean_Difference",
            "Drift_Interpretation",
            "KS_Statistic",
            "KS_P_Value",
        ]
    ].copy()

    result["UCI_Mean"] = result["UCI_Mean"].round(4)
    result["YOOCHOOSE_Mean"] = result["YOOCHOOSE_Mean"].round(4)
    result["UCI_Median"] = result["UCI_Median"].round(4)
    result["YOOCHOOSE_Median"] = result["YOOCHOOSE_Median"].round(4)

    result["Mean_Difference_Percent"] = (
        result["Mean_Difference_Percent"].round(2)
    )

    result["Standardized_Mean_Difference"] = (
        result["Standardized_Mean_Difference"].round(4)
    )

    result["KS_Statistic"] = (
        result["KS_Statistic"].round(4)
    )

    result["KS_P_Value"] = (
        result["KS_P_Value"].apply(format_p_value)
    )

    save_table(
        result,
        "Table_5_Behavioral_Drift.csv"
    )

    return result


# ============================================================
# TABLE 6 - DATASET CHARACTERISTICS / PURCHASE RATE
# ============================================================

def create_dataset_statistics():

    df = load_csv("cross_dataset_statistics.csv")

    result = df.copy()

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:

        if column != "Rows":
            result[column] = result[column].round(4)

    result = result.rename(
        columns={
            "Dataset": "Dataset",
            "Rows": "Sessions",
            "Purchase_Rate": "Purchase_Rate",
        }
    )

    save_table(
        result,
        "Table_6_Dataset_Statistics.csv"
    )

    return result


# ============================================================
# TABLE 7 - HYPOTHESIS TESTING
# ============================================================

def create_hypothesis_testing():

    df = load_csv("statistical_hypothesis_summary.csv")

    result = df.copy()

    result["Test_Statistic"] = (
        result["Test_Statistic"].round(4)
    )

    result["Effect_Size"] = (
        result["Effect_Size"].round(4)
    )

    result["P_Value"] = (
        result["P_Value"].apply(format_p_value)
    )

    save_table(
        result,
        "Table_7_Hypothesis_Testing.csv"
    )

    return result


# ============================================================
# TABLE 8 - REAL-TIME LATENCY
# ============================================================

def create_latency():

    df = load_csv(
        "refined_realtime_latency_metrics.csv"
    )

    result = df[
        [
            "model",
            "model_only_mean_ms_per_session",
            "model_only_median_ms_per_session",
            "model_only_p95_ms_per_session",
            "end_to_end_mean_ms_per_session",
            "end_to_end_median_ms_per_session",
            "end_to_end_p95_ms_per_session",
            "latency_unit",
            "p95_interpretation",
        ]
    ].copy()

    result = result.rename(
        columns={
            "model": "Model",
            "model_only_mean_ms_per_session": "Model_Only_Mean_ms",
            "model_only_median_ms_per_session": "Model_Only_Median_ms",
            "model_only_p95_ms_per_session": "Model_Only_P95_ms",
            "end_to_end_mean_ms_per_session": "End_to_End_Mean_ms",
            "end_to_end_median_ms_per_session": "End_to_End_Median_ms",
            "end_to_end_p95_ms_per_session": "End_to_End_P95_ms",
            "latency_unit": "Latency_Unit",
            "p95_interpretation": "P95_Interpretation",
        }
    )

    numeric_columns = [
        "Model_Only_Mean_ms",
        "Model_Only_Median_ms",
        "Model_Only_P95_ms",
        "End_to_End_Mean_ms",
        "End_to_End_Median_ms",
        "End_to_End_P95_ms",
    ]

    for column in numeric_columns:
        result[column] = result[column].round(4)

    save_table(
        result,
        "Table_8_Real_Time_Latency.csv"
    )

    return result


# ============================================================
# TABLE 9 - OVERALL EXPERIMENT SUMMARY
# ============================================================

def create_overall_summary(
    model_comparison,
    early_prediction,
    anonymous_prediction,
    cross_dataset,
    drift,
    latency
):

    summary = pd.DataFrame([
        {
            "Experiment": "Model Comparison",
            "Result_File": "Table_1_Model_Comparison.csv",
            "Status": "Completed",
        },
        {
            "Experiment": "Early Prediction",
            "Result_File": "Table_2_Early_Prediction.csv",
            "Status": "Completed",
        },
        {
            "Experiment": "Anonymous User Prediction",
            "Result_File": "Table_3_Anonymous_User_Prediction.csv",
            "Status": "Completed",
        },
        {
            "Experiment": "Cross-Dataset Validation",
            "Result_File": "Table_4_Cross_Dataset_Validation.csv",
            "Status": "Completed",
        },
        {
            "Experiment": "Behavioral Drift",
            "Result_File": "Table_5_Behavioral_Drift.csv",
            "Status": "Completed",
        },
        {
            "Experiment": "Dataset Statistics",
            "Result_File": "Table_6_Dataset_Statistics.csv",
            "Status": "Completed",
        },
        {
            "Experiment": "Hypothesis Testing",
            "Result_File": "Table_7_Hypothesis_Testing.csv",
            "Status": "Completed",
        },
        {
            "Experiment": "Real-Time Latency",
            "Result_File": "Table_8_Real_Time_Latency.csv",
            "Status": "Completed",
        },
    ])

    save_table(
        summary,
        "Table_9_Overall_Experimental_Summary.csv"
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("STEP 17 - PUBLICATION-READY RESULT TABLE GENERATION")
    print("=" * 80)

    print("\nNo models will be trained.")
    print("No predictions will be regenerated.")
    print("Existing CSV outputs will be used only.")

    model_comparison = create_model_comparison()

    early_prediction = create_early_prediction()

    anonymous_prediction = create_anonymous_prediction()

    cross_dataset = create_cross_dataset()

    drift = create_behavioral_drift()

    dataset_statistics = create_dataset_statistics()

    hypothesis = create_hypothesis_testing()

    latency = create_latency()

    summary = create_overall_summary(
        model_comparison,
        early_prediction,
        anonymous_prediction,
        cross_dataset,
        drift,
        latency,
    )

    print("\n" + "=" * 80)
    print("STEP 17 COMPLETED")
    print("=" * 80)

    print("\nPublication-ready tables saved in:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")

    for path in sorted(OUTPUT_DIR.glob("Table_*.csv")):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()