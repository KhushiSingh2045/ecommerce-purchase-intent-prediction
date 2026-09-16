"""
Cleans the raw UCI Online Shoppers Purchasing Intention dataset:
removes duplicates, fixes data types, encodes categorical columns,
and scales numeric columns. Saves a model-ready CSV to data/processed/.
"""
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_PATH = "data/raw/uci_shoppers/online_shoppers_intention.csv"
PROCESSED_PATH = "data/processed/shoppers_clean.csv"

CATEGORICAL_COLUMNS = ["Month", "VisitorType"]
NUMERIC_COLUMNS = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]


def clean_shoppers_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Remove duplicate rows
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {n_before - len(df)} duplicate rows.")

    # 2. Fix data types
    df["Revenue"] = df["Revenue"].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)

    # 3. Encode categorical columns (Month, VisitorType) as numbers
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 4. Scale numeric columns
    scaler = StandardScaler()
    df[NUMERIC_COLUMNS] = scaler.fit_transform(df[NUMERIC_COLUMNS])

    return df


if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} rows from {RAW_PATH}")

    cleaned = clean_shoppers_dataset(df)

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    cleaned.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved {len(cleaned)} cleaned rows to {PROCESSED_PATH}")