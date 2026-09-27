"""
YOOCHOOSE SEQUENTIAL DATA PIPELINE

Builds a leakage-controlled sequential dataset directly from the
raw YOOCHOOSE clickstream files.

Pipeline:
    1. Load raw clicks.
    2. Load purchase-session IDs from raw buys.
    3. Identify session start timestamps.
    4. Sort sessions chronologically.
    5. Create temporal train / validation / test splits.
    6. Learn the item vocabulary from TRAINING sessions only.
    7. Map unseen validation/test items to UNK.
    8. Build fixed-length padded item sequences.
    9. Build per-event time-delta sequences.
   10. Save the complete temporal split dataset.

Important:
    - Raw data is never modified.
    - Validation/test sessions never enter training.
    - Validation/test items do not influence the training vocabulary.
    - Purchase labels come only from the raw buy-session file.
    - No balancing is performed here.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CLICKS_PATH = Path(
    "data/raw/yoochoose/yoochoose-clicks.dat"
)

BUYS_PATH = Path(
    "data/raw/yoochoose/yoochoose-buys.dat"
)

OUTPUT_PATH = Path(
    "data/processed/yoochoose_temporal_split.npz"
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_SEQ_LEN = 20

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# Special token:
# 0 = padding
# 1 = unknown item
# 2+ = training vocabulary items
PAD_CODE = 0
UNK_CODE = 1


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

if not np.isclose(
    TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO,
    1.0
):
    raise ValueError(
        "TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO must equal 1.0."
    )


# ============================================================
# LOAD CLICKS
# ============================================================

def load_clicks(path: Path) -> pd.DataFrame:
    """
    Load the raw YOOCHOOSE click file.

    Expected raw schema:
        session_id,timestamp,item_id,category

    The category field is retained for schema validation but is not
    currently used as a model input.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"\nRaw click file was not found:\n{path}\n"
            "\nExpected location:\n"
            "data/raw/yoochoose/yoochoose-clicks.dat"
        )

    print("\nLoading raw clicks:")
    print(path)

    clicks = pd.read_csv(
        path,
        header=None,
        names=[
            "session_id",
            "timestamp",
            "item_id",
            "category",
        ],
        dtype={
            "session_id": "int64",
            "item_id": "int64",
            "category": "string",
        },
    )

    clicks["timestamp"] = pd.to_datetime(
        clicks["timestamp"],
        errors="coerce"
    )

    invalid_timestamp_count = int(
        clicks["timestamp"].isna().sum()
    )

    if invalid_timestamp_count > 0:
        raise ValueError(
            f"Found {invalid_timestamp_count:,} click rows "
            "with invalid timestamps."
        )

    if clicks["session_id"].isna().any():
        raise ValueError("Null session IDs found.")

    if clicks["item_id"].isna().any():
        raise ValueError("Null item IDs found.")

    clicks = clicks.sort_values(
        ["session_id", "timestamp"],
        kind="mergesort"
    ).reset_index(drop=True)

    print(f"Clicks loaded   : {len(clicks):,}")
    print(
        f"Unique sessions : "
        f"{clicks['session_id'].nunique():,}"
    )
    print(
        f"Unique items    : "
        f"{clicks['item_id'].nunique():,}"
    )

    return clicks


# ============================================================
# LOAD PURCHASE SESSION IDS
# ============================================================

def load_purchase_session_ids(path: Path) -> set[int]:
    """
    Load purchase sessions from the raw YOOCHOOSE buy file.

    A session is labelled:
        1 = purchase
        0 = no purchase
    """

    if not path.exists():
        raise FileNotFoundError(
            f"\nRaw purchase file was not found:\n{path}\n"
            "\nExpected location:\n"
            "data/raw/yoochoose/yoochoose-buys.dat"
        )

    print("\nLoading raw purchases:")
    print(path)

    buys = pd.read_csv(
        path,
        header=None,
        names=[
            "session_id",
            "timestamp",
            "item_id",
            "price",
            "quantity",
        ],
        dtype={
            "session_id": "int64",
            "item_id": "int64",
        },
        usecols=[0, 1, 2, 3, 4],
    )

    purchase_sessions = set(
        buys["session_id"].dropna().astype(np.int64).unique()
    )

    print(
        f"Purchase sessions: "
        f"{len(purchase_sessions):,}"
    )

    return purchase_sessions


# ============================================================
# TEMPORAL SESSION SPLIT
# ============================================================

def create_temporal_session_split(
    clicks: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Split sessions chronologically using each session's first event.

    This avoids randomly mixing future sessions into training.
    """

    print("\n" + "=" * 70)
    print("CREATING TEMPORAL SESSION SPLIT")
    print("=" * 70)

    session_start = (
        clicks.groupby("session_id", sort=False)["timestamp"]
        .min()
        .rename("session_start")
        .reset_index()
    )

    session_start = session_start.sort_values(
        "session_start",
        kind="mergesort"
    ).reset_index(drop=True)

    n_sessions = len(session_start)

    if n_sessions < 10:
        raise ValueError(
            "Too few sessions for a reliable temporal split."
        )

    train_end = int(
        np.floor(n_sessions * train_ratio)
    )

    validation_end = train_end + int(
        np.floor(n_sessions * validation_ratio)
    )

    # Ensure every session belongs to exactly one split.
    train_end = max(1, min(train_end, n_sessions - 2))
    validation_end = max(
        train_end + 1,
        min(validation_end, n_sessions - 1)
    )

    train_ids = session_start.iloc[
        :train_end
    ]["session_id"].to_numpy(dtype=np.int64)

    validation_ids = session_start.iloc[
        train_end:validation_end
    ]["session_id"].to_numpy(dtype=np.int64)

    test_ids = session_start.iloc[
        validation_end:
    ]["session_id"].to_numpy(dtype=np.int64)

    if len(train_ids) == 0:
        raise ValueError("Training split is empty.")

    if len(validation_ids) == 0:
        raise ValueError("Validation split is empty.")

    if len(test_ids) == 0:
        raise ValueError("Test split is empty.")

    train_set = set(train_ids.tolist())
    validation_set = set(validation_ids.tolist())
    test_set = set(test_ids.tolist())

    if train_set & validation_set:
        raise ValueError(
            "Temporal split leakage: train/validation overlap."
        )

    if train_set & test_set:
        raise ValueError(
            "Temporal split leakage: train/test overlap."
        )

    if validation_set & test_set:
        raise ValueError(
            "Temporal split leakage: validation/test overlap."
        )

    print(
        f"\nTotal sessions       : {n_sessions:,}"
    )
    print(
        f"Training sessions    : {len(train_ids):,}"
    )
    print(
        f"Validation sessions  : {len(validation_ids):,}"
    )
    print(
        f"Test sessions        : {len(test_ids):,}"
    )

    print("\nTemporal boundaries:")

    print(
        "Train start          :",
        session_start.iloc[0]["session_start"]
    )

    print(
        "Train end            :",
        session_start.iloc[train_end - 1]["session_start"]
    )

    print(
        "Validation start     :",
        session_start.iloc[train_end]["session_start"]
    )

    print(
        "Validation end       :",
        session_start.iloc[validation_end - 1]["session_start"]
    )

    print(
        "Test start           :",
        session_start.iloc[validation_end]["session_start"]
    )

    print(
        "Test end             :",
        session_start.iloc[-1]["session_start"]
    )

    return (
        train_ids,
        validation_ids,
        test_ids,
        session_start,
    )


# ============================================================
# BUILD TRAINING VOCABULARY
# ============================================================

def build_training_vocabulary(
    clicks: pd.DataFrame,
    train_session_ids: np.ndarray,
) -> dict[int, int]:
    """
    Learn the item vocabulary using TRAINING clicks only.

    Mapping:
        0 = padding
        1 = unknown item
        2+ = known training item
    """

    print("\n" + "=" * 70)
    print("BUILDING TRAINING-ONLY ITEM VOCABULARY")
    print("=" * 70)

    train_mask = clicks["session_id"].isin(
        set(train_session_ids.tolist())
    )

    train_items = (
        clicks.loc[train_mask, "item_id"]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )

    vocabulary = {
        int(item_id): int(index + 2)
        for index, item_id in enumerate(train_items)
    }

    print(
        f"Training vocabulary size: "
        f"{len(vocabulary):,}"
    )

    print(f"Padding code            : {PAD_CODE}")
    print(f"Unknown-item code       : {UNK_CODE}")

    return vocabulary


# ============================================================
# BUILD SEQUENCES FOR ONE SPLIT
# ============================================================

def build_sequences_for_split(
    clicks: pd.DataFrame,
    session_ids: np.ndarray,
    vocabulary: dict[int, int],
    max_seq_len: int = MAX_SEQ_LEN,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Build fixed-length item sequences and time-delta sequences.

    Only the first max_seq_len events are retained.

    Padding:
        item code = 0
        time delta = 0

    Unknown validation/test items:
        item code = 1
    """

    session_set = set(
        session_ids.tolist()
    )

    split_clicks = clicks[
        clicks["session_id"].isin(session_set)
    ].copy()

    split_clicks = split_clicks.sort_values(
        ["session_id", "timestamp"],
        kind="mergesort"
    )

    ordered_session_ids = (
        pd.Index(session_ids)
    )

    session_to_row = {
        int(session_id): int(index)
        for index, session_id in enumerate(
            ordered_session_ids
        )
    }

    sequences = np.full(
        (
            len(ordered_session_ids),
            max_seq_len,
        ),
        PAD_CODE,
        dtype=np.int32,
    )

    time_deltas = np.zeros(
        (
            len(ordered_session_ids),
            max_seq_len,
        ),
        dtype=np.float32,
    )

    # Work on only the first max_seq_len events per session.
    split_clicks["position"] = (
        split_clicks.groupby(
            "session_id",
            sort=False
        ).cumcount()
    )

    split_clicks = split_clicks[
        split_clicks["position"] < max_seq_len
    ].copy()

    # Convert timestamps to seconds.
    timestamp_seconds = (
        split_clicks["timestamp"].astype("int64")
        // 1_000_000_000
    )

    split_clicks["time_delta"] = (
        timestamp_seconds
        .groupby(split_clicks["session_id"])
        .diff()
        .fillna(0)
        .clip(lower=0)
        .astype(np.float32)
    )

    split_clicks["item_code"] = (
        split_clicks["item_id"]
        .map(vocabulary)
        .fillna(UNK_CODE)
        .astype(np.int32)
    )

    row_indices = (
        split_clicks["session_id"]
        .map(session_to_row)
        .to_numpy()
    )

    positions = (
        split_clicks["position"]
        .to_numpy(dtype=np.int64)
    )

    sequences[
        row_indices,
        positions
    ] = split_clicks["item_code"].to_numpy(
        dtype=np.int32
    )

    time_deltas[
        row_indices,
        positions
    ] = split_clicks["time_delta"].to_numpy(
        dtype=np.float32
    )

    return (
        sequences,
        time_deltas,
        ordered_session_ids.to_numpy(dtype=np.int64),
    )


# ============================================================
# LABEL SESSIONS
# ============================================================

def create_labels(
    session_ids: np.ndarray,
    purchase_session_ids: set[int],
) -> np.ndarray:
    """
    Assign binary purchase labels.
    """

    return np.asarray(
        [
            1 if int(session_id) in purchase_session_ids else 0
            for session_id in session_ids
        ],
        dtype=np.int8,
    )


# ============================================================
# PRINT SPLIT STATISTICS
# ============================================================

def print_split_statistics(
    name: str,
    labels: np.ndarray,
    sequences: np.ndarray,
) -> None:

    positive = int(
        np.sum(labels == 1)
    )

    negative = int(
        np.sum(labels == 0)
    )

    total = len(labels)

    purchase_rate = (
        positive / total
        if total > 0
        else 0.0
    )

    nonzero_events = np.sum(
        sequences != PAD_CODE,
        axis=1
    )

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(
        f"Sessions             : {total:,}"
    )

    print(
        f"Purchase sessions    : {positive:,}"
    )

    print(
        f"No-purchase sessions : {negative:,}"
    )

    print(
        f"Purchase rate        : "
        f"{purchase_rate * 100:.2f}%"
    )

    print(
        f"Mean sequence length : "
        f"{nonzero_events.mean():.2f}"
    )

    print(
        f"Median sequence len  : "
        f"{np.median(nonzero_events):.2f}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:

    print("=" * 80)
    print("YOOCHOOSE TEMPORAL SEQUENTIAL DATA PIPELINE")
    print("=" * 80)

    np.random.seed(RANDOM_SEED)

    # --------------------------------------------------------
    # 1. Load raw files
    # --------------------------------------------------------

    clicks = load_clicks(CLICKS_PATH)

    purchase_session_ids = (
        load_purchase_session_ids(BUYS_PATH)
    )

    # --------------------------------------------------------
    # 2. Temporal session split
    # --------------------------------------------------------

    (
        train_session_ids,
        validation_session_ids,
        test_session_ids,
        session_start,
    ) = create_temporal_session_split(
        clicks
    )

    # --------------------------------------------------------
    # 3. Training-only vocabulary
    # --------------------------------------------------------

    vocabulary = build_training_vocabulary(
        clicks,
        train_session_ids,
    )

    # --------------------------------------------------------
    # 4. Build sequences
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILDING TRAINING SEQUENCES")
    print("=" * 70)

    (
        train_sequences,
        train_time_deltas,
        train_session_ids,
    ) = build_sequences_for_split(
        clicks,
        train_session_ids,
        vocabulary,
        MAX_SEQ_LEN,
    )

    print("\n" + "=" * 70)
    print("BUILDING VALIDATION SEQUENCES")
    print("=" * 70)

    (
        validation_sequences,
        validation_time_deltas,
        validation_session_ids,
    ) = build_sequences_for_split(
        clicks,
        validation_session_ids,
        vocabulary,
        MAX_SEQ_LEN,
    )

    print("\n" + "=" * 70)
    print("BUILDING TEST SEQUENCES")
    print("=" * 70)

    (
        test_sequences,
        test_time_deltas,
        test_session_ids,
    ) = build_sequences_for_split(
        clicks,
        test_session_ids,
        vocabulary,
        MAX_SEQ_LEN,
    )

    # --------------------------------------------------------
    # 5. Create labels
    # --------------------------------------------------------

    train_labels = create_labels(
        train_session_ids,
        purchase_session_ids,
    )

    validation_labels = create_labels(
        validation_session_ids,
        purchase_session_ids,
    )

    test_labels = create_labels(
        test_session_ids,
        purchase_session_ids,
    )

    # --------------------------------------------------------
    # 6. Strong leakage checks
    # --------------------------------------------------------

    train_set = set(
        train_session_ids.tolist()
    )

    validation_set = set(
        validation_session_ids.tolist()
    )

    test_set = set(
        test_session_ids.tolist()
    )

    if train_set & validation_set:
        raise RuntimeError(
            "TRAIN/VALIDATION SESSION LEAKAGE DETECTED."
        )

    if train_set & test_set:
        raise RuntimeError(
            "TRAIN/TEST SESSION LEAKAGE DETECTED."
        )

    if validation_set & test_set:
        raise RuntimeError(
            "VALIDATION/TEST SESSION LEAKAGE DETECTED."
        )

    # --------------------------------------------------------
    # 7. Verify array alignment
    # --------------------------------------------------------

    checks = [
        (
            len(train_sequences),
            len(train_time_deltas),
            "training sequence/time-delta length"
        ),
        (
            len(train_sequences),
            len(train_labels),
            "training sequence/label length"
        ),
        (
            len(validation_sequences),
            len(validation_time_deltas),
            "validation sequence/time-delta length"
        ),
        (
            len(validation_sequences),
            len(validation_labels),
            "validation sequence/label length"
        ),
        (
            len(test_sequences),
            len(test_time_deltas),
            "test sequence/time-delta length"
        ),
        (
            len(test_sequences),
            len(test_labels),
            "test sequence/label length"
        ),
    ]

    for left, right, description in checks:
        if left != right:
            raise RuntimeError(
                f"Alignment failure: {description}."
            )

    # --------------------------------------------------------
    # 8. Statistics
    # --------------------------------------------------------

    print_split_statistics(
        "TRAINING",
        train_labels,
        train_sequences,
    )

    print_split_statistics(
        "VALIDATION",
        validation_labels,
        validation_sequences,
    )

    print_split_statistics(
        "TEST",
        test_labels,
        test_sequences,
    )

    # --------------------------------------------------------
    # 9. Unknown item statistics
    # --------------------------------------------------------

    validation_unknown_rate = float(
        np.mean(
            validation_sequences == UNK_CODE
        )
    )

    test_unknown_rate = float(
        np.mean(
            test_sequences == UNK_CODE
        )
    )

    print("\nUnknown-item rates:")
    print(
        f"Validation array rate : "
        f"{validation_unknown_rate * 100:.4f}%"
    )
    print(
        f"Test array rate       : "
        f"{test_unknown_rate * 100:.4f}%"
    )

    # --------------------------------------------------------
    # 10. Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vocabulary_items = np.asarray(
        sorted(vocabulary.keys()),
        dtype=np.int64,
    )

    vocabulary_codes = np.asarray(
        [
            vocabulary[item]
            for item in sorted(vocabulary.keys())
        ],
        dtype=np.int32,
    )

    # Save session-start timestamps as strings because NumPy NPZ
    # is more portable than storing pandas Timestamp objects.
    train_start_lookup = (
        session_start
        .set_index("session_id")
        .loc[train_session_ids]["session_start"]
        .astype(str)
        .to_numpy()
    )

    validation_start_lookup = (
        session_start
        .set_index("session_id")
        .loc[validation_session_ids]["session_start"]
        .astype(str)
        .to_numpy()
    )

    test_start_lookup = (
        session_start
        .set_index("session_id")
        .loc[test_session_ids]["session_start"]
        .astype(str)
        .to_numpy()
    )

    np.savez_compressed(
        OUTPUT_PATH,

        # Training
        train_sequences=train_sequences,
        train_time_deltas=train_time_deltas,
        train_labels=train_labels,
        train_session_ids=train_session_ids,
        train_session_start_times=train_start_lookup,

        # Validation
        val_sequences=validation_sequences,
        val_time_deltas=validation_time_deltas,
        val_labels=validation_labels,
        val_session_ids=validation_session_ids,
        val_session_start_times=validation_start_lookup,

        # Test
        test_sequences=test_sequences,
        test_time_deltas=test_time_deltas,
        test_labels=test_labels,
        test_session_ids=test_session_ids,
        test_session_start_times=test_start_lookup,

        # Vocabulary
        vocabulary_items=vocabulary_items,
        vocabulary_codes=vocabulary_codes,

        # Metadata
        max_sequence_length=np.asarray(
            MAX_SEQ_LEN,
            dtype=np.int32,
        ),
        pad_code=np.asarray(
            PAD_CODE,
            dtype=np.int32,
        ),
        unknown_code=np.asarray(
            UNK_CODE,
            dtype=np.int32,
        ),
        random_seed=np.asarray(
            RANDOM_SEED,
            dtype=np.int32,
        ),
        train_ratio=np.asarray(
            TRAIN_RATIO,
            dtype=np.float32,
        ),
        validation_ratio=np.asarray(
            VALIDATION_RATIO,
            dtype=np.float32,
        ),
        test_ratio=np.asarray(
            TEST_RATIO,
            dtype=np.float32,
        ),
        split_method=np.asarray(
            "chronological_session_start",
        ),
        vocabulary_method=np.asarray(
            "training_only",
        ),
    )

    file_size_mb = (
        os.path.getsize(OUTPUT_PATH)
        / (1024 * 1024)
    )

    print("\n" + "=" * 80)
    print("TEMPORAL DATA PIPELINE COMPLETED")
    print("=" * 80)

    print(
        f"\nSaved dataset:\n{OUTPUT_PATH}"
    )

    print(
        f"File size: {file_size_mb:.2f} MB"
    )

    print("\nLeakage checks:")
    print("Train ∩ Validation : 0")
    print("Train ∩ Test       : 0")
    print("Validation ∩ Test  : 0")

    print("\nImportant:")
    print("- Split is chronological.")
    print("- Vocabulary was learned from TRAIN only.")
    print("- Validation/test unseen items use UNK.")
    print("- Validation and test remain naturally distributed.")
    print("- No balancing has been performed.")


if __name__ == "__main__":
    main()