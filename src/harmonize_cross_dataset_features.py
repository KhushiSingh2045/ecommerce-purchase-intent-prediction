
import os
import numpy as np
import pandas as pd

UCI_RAW_PATH = "data/raw/uci_shoppers/online_shoppers_intention.csv"
YOOCHOOSE_SEQ_PATH = "data/processed/yoochoose_sequences.npz"
YOOCHOOSE_DL_SUBSET_PATH = "data/processed/yoochoose_dl_subset.npz"

UCI_OUT_PATH = "data/processed/cross_dataset_uci.csv"
YOOCHOOSE_OUT_PATH = "data/processed/cross_dataset_yoochoose.csv"


def harmonize_uci(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    harmonized = pd.DataFrame({
        "session_click_count": df["Administrative"] + df["Informational"] + df["ProductRelated"],
        "session_duration_seconds": df["Administrative_Duration"] + df["Informational_Duration"] + df["ProductRelated_Duration"],
        "unique_items_or_pages": df["ProductRelated"],  # closest available proxy - UCI has no raw item IDs
        "purchase": df["Revenue"].astype(int),
    })
    return harmonized


def load_yoochoose_sequences():
    """Prefers the full sequence set; falls back to combining the DL subset's splits."""
    if os.path.exists(YOOCHOOSE_SEQ_PATH):
        data = np.load(YOOCHOOSE_SEQ_PATH)
        return data["sequences"], data["time_deltas"], data["labels"]

    if os.path.exists(YOOCHOOSE_DL_SUBSET_PATH):
        data = np.load(YOOCHOOSE_DL_SUBSET_PATH)
        sequences = np.concatenate([data["train_sequences"], data["val_sequences"], data["test_sequences"]])
        time_deltas = np.concatenate([data["train_time_deltas"], data["val_time_deltas"], data["test_time_deltas"]])
        labels = np.concatenate([data["train_labels"], data["val_labels"], data["test_labels"]])
        return sequences, time_deltas, labels

    raise FileNotFoundError(
        f"Neither {YOOCHOOSE_SEQ_PATH} nor {YOOCHOOSE_DL_SUBSET_PATH} found. "
        "Run src/feature_engineering.py or src/prepare_dl_dataset.py first."
    )


def harmonize_yoochoose() -> pd.DataFrame:
    sequences, time_deltas, labels = load_yoochoose_sequences()

    click_count = (sequences != 0).sum(axis=1)
    duration = time_deltas.sum(axis=1)
    unique_items = np.array([len(np.unique(seq[seq != 0])) for seq in sequences])

    harmonized = pd.DataFrame({
        "session_click_count": click_count,
        "session_duration_seconds": duration,
        "unique_items_or_pages": unique_items,
        "purchase": labels.astype(int),
    })
    return harmonized


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    if os.path.exists(UCI_RAW_PATH):
        uci_df = harmonize_uci(UCI_RAW_PATH)
        uci_df.to_csv(UCI_OUT_PATH, index=False)
        print(f"UCI harmonized: {len(uci_df)} sessions -> {UCI_OUT_PATH}")
        print(uci_df.describe().to_string())
    else:
        print(f"Skipping UCI: {UCI_RAW_PATH} not found.")

    print()
    yoochoose_df = harmonize_yoochoose()
    yoochoose_df.to_csv(YOOCHOOSE_OUT_PATH, index=False)
    print(f"YOOCHOOSE harmonized: {len(yoochoose_df)} sessions -> {YOOCHOOSE_OUT_PATH}")
    print(yoochoose_df.describe().to_string())

    print("\nDone.")