"""
Cross-Dataset Validation
========================

Purpose:
    Compare purchase-intent prediction performance across available
    e-commerce datasets.

Important:
    This script does NOT assume that different datasets have identical
    columns.

    It first discovers CSV files, identifies possible target columns,
    finds common numeric features, and only performs a comparison when
    there is enough compatible information.

Output:
    results/metrics/cross_dataset_results.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIRS = [
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "data" / "external",
    PROJECT_ROOT / "data" / "processed",
]

RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. POSSIBLE TARGET COLUMN NAMES
# ============================================================

TARGET_CANDIDATES = [
    "Revenue",
    "revenue",
    "purchase",
    "Purchase",
    "purchased",
    "Purchased",
    "target",
    "Target",
    "label",
    "Label",
]


# ============================================================
# 3. DATASET DISCOVERY
# ============================================================

def find_csv_files():
    """
    Find CSV files in the project's data directories.
    """

    csv_files = []

    for directory in DATA_DIRS:

        if not directory.exists():
            continue

        for file in directory.rglob("*.csv"):
            csv_files.append(file)

    # Remove duplicates
    csv_files = sorted(set(csv_files))

    return csv_files


# ============================================================
# 4. TARGET COLUMN DETECTION
# ============================================================

def find_target_column(df):
    """
    Detect a likely purchase-intent target column.
    """

    for column in TARGET_CANDIDATES:

        if column in df.columns:
            return column

    return None


# ============================================================
# 5. TARGET NORMALIZATION
# ============================================================

def normalize_target(series):
    """
    Convert common binary target representations into 0/1.

    Returns:
        pandas Series or None
    """

    # Already numeric
    if pd.api.types.is_numeric_dtype(series):

        unique_values = set(series.dropna().unique())

        if unique_values.issubset({0, 1}):
            return series.astype(int)

    # Convert text representations
    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "purchase": 1,
        "purchased": 1,
        "buy": 1,
        "buyer": 1,
        "revenue": 1,
        "positive": 1,
        "1": 1,
        "0": 0,
        "not purchased": 0,
        "no purchase": 0,
        "negative": 0,
        "false": 0,
    }

    values = series.astype(str).str.strip().str.lower()

    converted = values.map(mapping)

    if converted.notna().sum() == 0:
        return None

    return converted


# ============================================================
# 6. LOAD DATASET INFORMATION
# ============================================================

def inspect_dataset(file_path):
    """
    Load a dataset and collect basic information.
    """

    try:

        df = pd.read_csv(file_path)

    except Exception as error:

        print(f"\nCould not read: {file_path}")
        print(f"Reason: {error}")

        return None

    target_column = find_target_column(df)

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    return {
        "path": file_path,
        "dataframe": df,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "target_column": target_column,
        "numeric_columns": numeric_columns,
    }


# ============================================================
# 7. PRINT DATASET INFORMATION
# ============================================================

def print_dataset_information(info):

    print("\n" + "=" * 70)
    print("DATASET")
    print("=" * 70)

    print(f"File       : {info['path']}")
    print(f"Rows       : {info['rows']}")
    print(f"Columns    : {len(info['columns'])}")

    print("\nTarget column:")
    print(info["target_column"])

    print("\nNumeric columns:")
    print(info["numeric_columns"])


# ============================================================
# 8. FIND COMPATIBLE NUMERIC FEATURES
# ============================================================

def find_common_features(dataset_infos):

    if len(dataset_infos) < 2:
        return []

    common_features = set(
        dataset_infos[0]["numeric_columns"]
    )

    for info in dataset_infos[1:]:
        common_features.intersection_update(
            info["numeric_columns"]
        )

    # Remove possible target columns
    target_columns = set()

    for info in dataset_infos:
        if info["target_column"] is not None:
            target_columns.add(info["target_column"])

    common_features -= target_columns

    return sorted(common_features)


# ============================================================
# 9. PREPARE DATA
# ============================================================

def prepare_dataset(info, feature_columns):

    df = info["dataframe"].copy()

    target_column = info["target_column"]

    if target_column is None:
        return None

    target = normalize_target(df[target_column])

    if target is None:
        print(
            f"Target column '{target_column}' "
            f"could not be converted to binary 0/1."
        )
        return None

    # Keep only compatible features
    available_features = [
        feature
        for feature in feature_columns
        if feature in df.columns
    ]

    if len(available_features) == 0:
        return None

    X = df[available_features].copy()
    y = target

    # Remove rows with invalid target
    valid_rows = y.notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    # Convert numeric columns
    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    return X, y.astype(int), available_features


# ============================================================
# 10. TRAIN AND EVALUATE MODEL
# ============================================================

def evaluate_dataset(X, y, dataset_name):

    # Need both classes
    if y.nunique() < 2:

        print(
            f"\nSkipping {dataset_name}: "
            "target contains fewer than two classes."
        )

        return None

    # Need enough samples
    if len(X) < 20:

        print(
            f"\nSkipping {dataset_name}: "
            "too few rows for a meaningful split."
        )

        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    try:

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )

    except ValueError:

        roc_auc = np.nan

    return {
        "Dataset": dataset_name,
        "Model": "Logistic Regression",
        "Feature_Representation": "Common Numeric Features",
        "Rows": len(X),
        "Features": X.shape[1],
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "Accuracy": accuracy,
    }


# ============================================================
# 11. MAIN CROSS-DATASET PROCEDURE
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CROSS-DATASET VALIDATION")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    # --------------------------------------------------------
    # Find datasets
    # --------------------------------------------------------

    csv_files = find_csv_files()

    if len(csv_files) == 0:

        print("\nNo CSV files were found.")

        print("\nChecked directories:")

        for directory in DATA_DIRS:
            print(f" - {directory}")

        print(
            "\nPlace the required datasets in "
            "data/raw/ or data/external/ and run again."
        )

        return

    print("\nCSV files discovered:")

    for file in csv_files:
        print(f" - {file}")

    # --------------------------------------------------------
    # Inspect datasets
    # --------------------------------------------------------

    dataset_infos = []

    for file in csv_files:

        info = inspect_dataset(file)

        if info is None:
            continue

        print_dataset_information(info)

        if info["target_column"] is not None:

            dataset_infos.append(info)

    # --------------------------------------------------------
    # Need at least two datasets
    # --------------------------------------------------------

    if len(dataset_infos) < 2:

        print("\n" + "=" * 70)
        print("CROSS-DATASET VALIDATION CANNOT YET BE PERFORMED")
        print("=" * 70)

        print(
            "\nAt least two datasets with identifiable purchase-intent "
            "target columns are required."
        )

        print(
            "\nThis is not a code failure. "
            "The required datasets are not currently available "
            "or their target columns could not be identified."
        )

        return

    # --------------------------------------------------------
    # Find common features
    # --------------------------------------------------------

    common_features = find_common_features(
        dataset_infos
    )

    print("\n" + "=" * 70)
    print("COMMON NUMERIC FEATURES")
    print("=" * 70)

    if common_features:

        for feature in common_features:
            print(f" - {feature}")

    else:

        print(
            "\nNo common numeric features were found."
        )

        print(
            "\nDo not force the datasets together."
        )

        print(
            "A proper feature-mapping step is required "
            "because the schemas are different."
        )

        return

    # --------------------------------------------------------
    # Evaluate datasets
    # --------------------------------------------------------

    results = []

    for info in dataset_infos:

        prepared = prepare_dataset(
            info,
            common_features
        )

        if prepared is None:
            continue

        X, y, features = prepared

        print("\n" + "=" * 70)
        print(
            f"EVALUATING: "
            f"{info['path'].name}"
        )
        print("=" * 70)

        print(f"Rows used      : {len(X)}")
        print(f"Features used  : {features}")

        result = evaluate_dataset(
            X,
            y,
            info["path"].name
        )

        if result is not None:
            results.append(result)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    if not results:

        print(
            "\nNo valid cross-dataset results were produced."
        )

        return

    results_df = pd.DataFrame(results)

    output_file = (
        RESULTS_DIR /
        "cross_dataset_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("CROSS-DATASET VALIDATION COMPLETED")
    print("=" * 70)

    print("\nResults:")

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved -> {output_file}"
    )


# ============================================================
# 12. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()