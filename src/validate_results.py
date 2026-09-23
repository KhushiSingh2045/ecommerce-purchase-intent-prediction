from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


# These are the ACTUAL files found in your project
EXPECTED_FILES = [
    "lstm_metrics.csv",
    "gru_metrics.csv",
    "gru_attention_metrics.csv",
    "xgboost_metrics.csv",
    "lightgbm_metrics.csv",
    "early_prediction_metrics.csv",
    "anonymous_user_prediction.csv",
    "cross_dataset_results.csv",
    "cross_dataset_statistics.csv",
    "behavioral_drift_tests.csv",
    "behavioral_drift_summary.csv",
    "statistical_descriptive_summary.csv",
    "hypothesis_mann_whitney_results.csv",
    "hypothesis_purchase_chi_square.csv",
    "statistical_hypothesis_summary.csv",
    "refined_realtime_event_latency.csv",
    "refined_realtime_latency_metrics.csv",
    "refined_realtime_session_predictions.csv",
]


def check_csv(file_path):
    print("\n" + "=" * 80)
    print(f"CHECKING: {file_path.name}")
    print("=" * 80)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"ERROR READING FILE: {e}")
        return {
            "file": file_path.name,
            "rows": None,
            "columns": None,
            "missing_values": None,
            "infinite_values": None,
            "duplicate_rows": None,
            "status": "READ_ERROR",
        }

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    missing = int(df.isna().sum().sum())

    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) > 0:
        infinite_values = int(
            np.isinf(numeric_df.to_numpy()).sum()
        )
    else:
        infinite_values = 0

    duplicate_rows = int(df.duplicated().sum())

    constant_columns = []

    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            constant_columns.append(col)

    print("\nMissing values:")

    missing_columns = df.isna().sum()
    missing_columns = missing_columns[missing_columns > 0]

    if len(missing_columns) == 0:
        print("  None")
    else:
        for col, count in missing_columns.items():
            percentage = count / len(df) * 100
            print(f"  {col}: {count} ({percentage:.4f}%)")

    print(f"\nInfinite numeric values: {infinite_values}")
    print(f"Duplicate rows: {duplicate_rows}")

    print("\nConstant columns:")

    if len(constant_columns) == 0:
        print("  None")
    else:
        for col in constant_columns:
            print(f"  WARNING: {col}")

    status = "OK"

    if missing > 0:
        status = "CHECK"

    if infinite_values > 0:
        status = "CHECK"

    if duplicate_rows > 0:
        status = "CHECK"

    # Constant columns are NOT automatically an error.
    # Many metric files contain one row per model and therefore
    # naturally have constant hyperparameter columns.
    print("\nStatus:", status)

    return {
        "file": file_path.name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": missing,
        "infinite_values": infinite_values,
        "duplicate_rows": duplicate_rows,
        "status": status,
    }


def main():

    print("=" * 80)
    print("STEP 17 RESULT FILE VALIDATION")
    print("=" * 80)

    print("\nProject:")
    print(PROJECT_ROOT)

    print("\nMetrics directory:")
    print(METRICS_DIR)

    print("\n" + "=" * 80)
    print("EXPECTED RESULT FILES")
    print("=" * 80)

    missing_files = []

    for filename in EXPECTED_FILES:

        path = METRICS_DIR / filename

        if path.exists():
            print(f"[OK]      {filename}")
        else:
            print(f"[MISSING] {filename}")
            missing_files.append(filename)

    print("\n" + "=" * 80)
    print("CSV QUALITY CHECK")
    print("=" * 80)

    reports = []

    for filename in EXPECTED_FILES:

        path = METRICS_DIR / filename

        if path.exists():
            reports.append(check_csv(path))

    # Also inspect any other CSVs that exist in the metrics folder.
    actual_csv_files = sorted(METRICS_DIR.glob("*.csv"))

    checked_names = {x["file"] for x in reports}

    for path in actual_csv_files:

        if path.name not in checked_names:
            reports.append(check_csv(path))

    report_df = pd.DataFrame(reports)

    output_file = METRICS_DIR / "step17_csv_validation_report.csv"

    report_df.to_csv(output_file, index=False)

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if missing_files:
        print("\nMissing expected files:")
        for filename in missing_files:
            print(f"  - {filename}")
    else:
        print("\nAll required result files are present.")

    attention = report_df[report_df["status"] != "OK"]

    if len(attention) > 0:
        print("\nFiles requiring attention:")
        print(attention.to_string(index=False))
    else:
        print("\nNo CSV quality problems detected.")

    print("\nValidation report:")
    print(output_file)

    print("\n" + "=" * 80)
    print("STEP 17 VALIDATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()