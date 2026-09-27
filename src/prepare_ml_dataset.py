"""
Prepare ML Dataset from Temporal Split
Extracts the 14 tabular session features for tree-based models (XGBoost/LightGBM).
Saves to data/processed/yoochoose_ml_temporal.npz
"""

from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Priority to temporal split, fallback to balanced train
if (PROCESSED_DIR / "yoochoose_temporal_split.npz").exists():
    INPUT_DATA_PATH = PROCESSED_DIR / "yoochoose_temporal_split.npz"
else:
    INPUT_DATA_PATH = PROCESSED_DIR / "yoochoose_balanced_train.npz"

OUTPUT_DATA_PATH = PROCESSED_DIR / "yoochoose_ml_temporal.npz"

FEATURE_NAMES = [
    "interaction_count",
    "unique_item_count",
    "repeated_item_count",
    "first_item_id",
    "last_item_id",
    "mean_item_id",
    "std_item_id",
    "min_item_id",
    "max_item_id",
    "total_time_delta",
    "mean_time_delta",
    "max_time_delta",
    "std_time_delta",
    "nonzero_time_intervals",
]


def extract_features(sequences, time_deltas):
    n_sessions = sequences.shape[0]
    features = np.zeros((n_sessions, 14), dtype=np.float32)

    for i in range(n_sessions):
        items = sequences[i]
        valid_items = items[items != 0]

        if len(valid_items) == 0:
            continue

        deltas = time_deltas[i]
        valid_deltas = deltas[deltas > 0]

        total_time = np.sum(valid_deltas) if len(valid_deltas) > 0 else 0.0
        mean_time = np.mean(valid_deltas) if len(valid_deltas) > 0 else 0.0
        max_time = np.max(valid_deltas) if len(valid_deltas) > 0 else 0.0
        std_time = np.std(valid_deltas) if len(valid_deltas) > 0 else 0.0

        features[i] = [
            len(valid_items),
            len(np.unique(valid_items)),
            len(valid_items) - len(np.unique(valid_items)),
            valid_items[0],
            valid_items[-1],
            np.mean(valid_items),
            np.std(valid_items),
            np.min(valid_items),
            np.max(valid_items),
            total_time,
            mean_time,
            max_time,
            std_time,
            len(valid_deltas),
        ]

    return features


def main():
    print("=" * 70)
    print("PREPARING TABULAR ML DATASET FROM TEMPORAL SEQUENCES")
    print("=" * 70)
    print(f"Loading input file: {INPUT_DATA_PATH}")

    data = np.load(INPUT_DATA_PATH, allow_pickle=True)

    train_seq = data["train_sequences"]
    train_deltas = data["train_time_deltas"]
    y_train = data["train_labels"]

    val_seq = data["val_sequences"]
    val_deltas = data["val_time_deltas"]
    y_val = data["val_labels"]

    test_seq = data["test_sequences"]
    test_deltas = data["test_time_deltas"]
    y_test = data["test_labels"]

    # Balance train set if imbalanced so XGBoost/LightGBM train on 50:50
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_pos = len(pos_idx)

    if len(neg_idx) > n_pos:
        print(f"Subsampling training negatives: {len(neg_idx):,} -> {n_pos:,} to balance 50:50...")
        np.random.seed(42)
        sampled_neg = np.random.choice(neg_idx, size=n_pos, replace=False)
        balanced_idx = np.concatenate([pos_idx, sampled_neg])
        np.random.shuffle(balanced_idx)

        train_seq = train_seq[balanced_idx]
        train_deltas = train_deltas[balanced_idx]
        y_train = y_train[balanced_idx]

    print(f"Extracting features for Train ({len(y_train):,} sessions)...")
    X_train = extract_features(train_seq, train_deltas)

    # For validation and test, if > 50,000 sessions, sample 50,000 for fast ML evaluation
    # while preserving natural distribution
    if len(y_val) > 50000:
        print("Sampling 50,000 validation sessions for tree evaluation...")
        np.random.seed(42)
        v_idx = np.random.choice(len(y_val), size=50000, replace=False)
        X_val = extract_features(val_seq[v_idx], val_deltas[v_idx])
        y_val = y_val[v_idx]
    else:
        X_val = extract_features(val_seq, val_deltas)

    if len(y_test) > 50000:
        print("Sampling 50,000 test sessions for tree evaluation...")
        np.random.seed(42)
        t_idx = np.random.choice(len(y_test), size=50000, replace=False)
        X_test = extract_features(test_seq[t_idx], test_deltas[t_idx])
        y_test = y_test[t_idx]
    else:
        X_test = extract_features(test_seq, test_deltas)

    np.savez_compressed(
        OUTPUT_DATA_PATH,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=np.array(FEATURE_NAMES),
    )

    print(f"\n[SAVED] {OUTPUT_DATA_PATH} ({OUTPUT_DATA_PATH.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    main()