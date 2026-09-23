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
    confusion_matrix,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CROSS-DATASET FILES
# ============================================================

UCI_FILE = PROCESSED_DIR / "cross_dataset_uci.csv"
YOOCHOOSE_FILE = PROCESSED_DIR / "cross_dataset_yoochoose.csv"


# ============================================================
# 3. HARMONIZED FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "session_click_count",
    "session_duration_seconds",
    "unique_items_or_pages",
]

TARGET_COLUMN = "purchase"


# ============================================================
# 4. LOAD HARMONIZED DATASET
# ============================================================

def load_harmonized_dataset(file_path):

    if not file_path.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"\nMissing required columns in {file_path.name}: "
            f"{missing_columns}"
        )

    df = df[required_columns].copy()

    # Convert features to numeric
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Convert target to numeric
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    # Keep binary target only
    df = df[
        df[TARGET_COLUMN].isin([0, 1])
    ].copy()

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    return df


# ============================================================
# 5. PRINT DATASET INFORMATION
# ============================================================

def print_dataset_information(df, dataset_name):

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)

    print(f"Rows              : {len(df)}")
    print(f"Features           : {FEATURE_COLUMNS}")
    print(f"Target             : {TARGET_COLUMN}")

    purchase_rate = df[TARGET_COLUMN].mean()

    print(
        f"Purchase rate      : "
        f"{purchase_rate:.4f} "
        f"({purchase_rate * 100:.2f}%)"
    )

    print("\nFeature statistics:")

    print(
        df[FEATURE_COLUMNS].describe().round(4).to_string()
    )


# ============================================================
# 6. CREATE MODEL
# ============================================================

def create_model():

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

    return model


# ============================================================
# 7. WITHIN-DATASET BASELINE
# ============================================================

def evaluate_within_dataset(
    df,
    dataset_name
):

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    if y.nunique() < 2:

        print(
            f"\nSkipping {dataset_name}: "
            "only one target class is present."
        )

        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    return {
        "Evaluation_Type": "Within-Dataset",
        "Train_Dataset": dataset_name,
        "Test_Dataset": dataset_name,
        "Train_Rows": len(X_train),
        "Test_Rows": len(X_test),
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "Accuracy": accuracy,
    }


# ============================================================
# 8. CROSS-DATASET EVALUATION
# ============================================================

def evaluate_cross_dataset(
    train_df,
    test_df,
    train_name,
    test_name
):

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    print("\n" + "=" * 70)
    print("CROSS-DATASET EVALUATION")
    print("=" * 70)

    print(f"Training dataset : {train_name}")
    print(f"Testing dataset  : {test_name}")

    print(f"Training rows    : {len(X_train)}")
    print(f"Testing rows     : {len(X_test)}")

    # Create model
    model = create_model()

    # Train ONLY on source dataset
    model.fit(
        X_train,
        y_train
    )

    # Test ONLY on external dataset
    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nResults:")

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")

    print("\nConfusion Matrix:")

    print(cm)

    return {
        "Evaluation_Type": "Cross-Dataset",
        "Train_Dataset": train_name,
        "Test_Dataset": test_name,
        "Train_Rows": len(X_train),
        "Test_Rows": len(X_test),
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "Accuracy": accuracy,
    }


# ============================================================
# 9. MAIN PROCEDURE
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("CROSS-DATASET VALIDATION")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    print("\nChecking harmonized datasets...")

    print(
        f"\nUCI      : {UCI_FILE}"
    )

    print(
        f"YOOCHOOSE: {YOOCHOOSE_FILE}"
    )

    if not UCI_FILE.exists():

        print(
            "\nERROR: Harmonized UCI dataset was not found."
        )

        print(
            "Run:"
        )

        print(
            "python -m src.harmonize_cross_dataset_features"
        )

        return

    if not YOOCHOOSE_FILE.exists():

        print(
            "\nERROR: Harmonized YOOCHOOSE dataset was not found."
        )

        print(
            "Run:"
        )

        print(
            "python -m src.harmonize_cross_dataset_features"
        )

        return

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    uci_df = load_harmonized_dataset(
        UCI_FILE
    )

    yoochoose_df = load_harmonized_dataset(
        YOOCHOOSE_FILE
    )

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print_dataset_information(
        uci_df,
        "UCI Online Shoppers Intention"
    )

    print_dataset_information(
        yoochoose_df,
        "YOOCHOOSE"
    )

    # --------------------------------------------------------
    # Confirm compatible schema
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HARMONIZED FEATURE SPACE")
    print("=" * 70)

    for feature in FEATURE_COLUMNS:
        print(f" - {feature}")

    print(f" - Target: {TARGET_COLUMN}")

    print(
        "\nThe two datasets now use the same feature representation."
    )

    # --------------------------------------------------------
    # Within-dataset baselines
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("WITHIN-DATASET BASELINES")
    print("=" * 70)

    results = []

    uci_baseline = evaluate_within_dataset(
        uci_df,
        "UCI"
    )

    if uci_baseline is not None:
        results.append(
            uci_baseline
        )

    yoochoose_baseline = evaluate_within_dataset(
        yoochoose_df,
        "YOOCHOOSE"
    )

    if yoochoose_baseline is not None:
        results.append(
            yoochoose_baseline
        )

    # --------------------------------------------------------
    # UCI -> YOOCHOOSE
    # --------------------------------------------------------

    uci_to_yoochoose = evaluate_cross_dataset(
        train_df=uci_df,
        test_df=yoochoose_df,
        train_name="UCI",
        test_name="YOOCHOOSE"
    )

    if uci_to_yoochoose is not None:
        results.append(
            uci_to_yoochoose
        )

    # --------------------------------------------------------
    # YOOCHOOSE -> UCI
    # --------------------------------------------------------

    yoochoose_to_uci = evaluate_cross_dataset(
        train_df=yoochoose_df,
        test_df=uci_df,
        train_name="YOOCHOOSE",
        test_name="UCI"
    )

    if yoochoose_to_uci is not None:
        results.append(
            yoochoose_to_uci
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    if not results:

        print(
            "\nNo valid results were produced."
        )

        return

    results_df = pd.DataFrame(
        results
    )

    output_file = (
        RESULTS_DIR /
        "cross_dataset_results.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Save dataset statistics
    # --------------------------------------------------------

    statistics = []

    for name, df in [
        ("UCI", uci_df),
        ("YOOCHOOSE", yoochoose_df),
    ]:

        row = {
            "Dataset": name,
            "Rows": len(df),
            "Purchase_Rate": df[TARGET_COLUMN].mean(),
        }

        for feature in FEATURE_COLUMNS:

            row[
                f"{feature}_Mean"
            ] = df[feature].mean()

            row[
                f"{feature}_Median"
            ] = df[feature].median()

            row[
                f"{feature}_Std"
            ] = df[feature].std()

        statistics.append(row)

    statistics_df = pd.DataFrame(
        statistics
    )

    statistics_file = (
        RESULTS_DIR /
        "cross_dataset_statistics.csv"
    )

    statistics_df.to_csv(
        statistics_file,
        index=False
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CROSS-DATASET VALIDATION COMPLETED")
    print("=" * 70)

    print("\nResults:")

    print(
        results_df.round(4).to_string(
            index=False
        )
    )

    print(
        f"\nSaved results -> {output_file}"
    )

    print(
        f"Saved statistics -> {statistics_file}"
    )

    print("\nImportant:")
    print(
        "The existing trained deep-learning/XGBoost/LightGBM "
        "models were NOT retrained."
    )

    print(
        "This experiment uses a common three-feature representation "
        "and Logistic Regression to test cross-dataset generalization."
    )


# ============================================================
# 10. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()