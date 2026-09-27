"""
Create a reproducible, computationally bounded training dataset.

Input:
    data/processed/yoochoose_temporal_split.npz

Output:
    data/processed/yoochoose_balanced_train.npz

Method:
    1. Keep the temporal validation and test sets completely unchanged.
    2. Sample an equal number of positive and negative sessions from the
       temporal training split.
    3. Sample WITHOUT replacement.
    4. Use a fixed random seed for reproducibility.
    5. Do not use SMOTE or synthetic sequence generation.

Default:
    200,000 training sessions
        100,000 positive
        100,000 negative

This script intentionally does NOT modify validation or test data.
"""

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_PATH = Path("data/processed/yoochoose_temporal_split.npz")
OUTPUT_PATH = Path("data/processed/yoochoose_balanced_train.npz")

RANDOM_SEED = 42

# Total number of training sessions.
# Must be even because we create a 50:50 balanced dataset.
TARGET_TRAINING_SESSIONS = 200_000

# 50:50 positive / negative training balance.
POSITIVE_RATIO = 0.50


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def require_keys(data, required_keys):
    """Ensure all expected arrays exist in the input NPZ."""

    missing = [key for key in required_keys if key not in data]

    if missing:
        raise KeyError(
            "The temporal split file is missing required arrays:\n"
            + "\n".join(f"  - {key}" for key in missing)
        )


def class_counts(labels):
    """Return negative and positive class counts."""

    labels = np.asarray(labels)

    negative = int(np.sum(labels == 0))
    positive = int(np.sum(labels == 1))

    invalid = int(np.sum((labels != 0) & (labels != 1)))

    if invalid:
        raise ValueError(
            f"Found {invalid} labels that are neither 0 nor 1."
        )

    return negative, positive


def verify_session_uniqueness(session_ids, split_name):
    """Verify that session IDs are unique within a split."""

    session_ids = np.asarray(session_ids)

    unique_count = np.unique(session_ids).size

    if unique_count != len(session_ids):
        duplicates = len(session_ids) - unique_count

        raise ValueError(
            f"{split_name} contains {duplicates} duplicate session IDs."
        )


def verify_no_overlap(train_ids, val_ids, test_ids):
    """Verify that no session appears in multiple splits."""

    train_ids = set(np.asarray(train_ids).tolist())
    val_ids = set(np.asarray(val_ids).tolist())
    test_ids = set(np.asarray(test_ids).tolist())

    train_val = train_ids.intersection(val_ids)
    train_test = train_ids.intersection(test_ids)
    val_test = val_ids.intersection(test_ids)

    if train_val:
        raise ValueError(
            f"Train/validation leakage detected: {len(train_val)} sessions."
        )

    if train_test:
        raise ValueError(
            f"Train/test leakage detected: {len(train_test)} sessions."
        )

    if val_test:
        raise ValueError(
            f"Validation/test leakage detected: {len(val_test)} sessions."
        )


def verify_alignment(sequences, time_deltas, labels, session_ids, split_name):
    """Verify that all session-level arrays have the same number of rows."""

    lengths = {
        "sequences": len(sequences),
        "time_deltas": len(time_deltas),
        "labels": len(labels),
        "session_ids": len(session_ids),
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"{split_name} arrays are misaligned:\n"
            + "\n".join(
                f"  {name}: {length}"
                for name, length in lengths.items()
            )
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("YOOCHOOSE — REPRODUCIBLE BALANCED TRAINING SAMPLE")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"\nInput file not found:\n"
            f"  {INPUT_PATH}\n\n"
            "Run the temporal feature-engineering/split step first."
        )

    if TARGET_TRAINING_SESSIONS <= 0:
        raise ValueError("TARGET_TRAINING_SESSIONS must be positive.")

    if TARGET_TRAINING_SESSIONS % 2 != 0:
        raise ValueError(
            "TARGET_TRAINING_SESSIONS must be even for a 50:50 balance."
        )

    positive_target = int(
        TARGET_TRAINING_SESSIONS * POSITIVE_RATIO
    )

    negative_target = TARGET_TRAINING_SESSIONS - positive_target

    print(f"\nInput : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Target training sessions: {TARGET_TRAINING_SESSIONS:,}")
    print(f"Target positive sessions : {positive_target:,}")
    print(f"Target negative sessions : {negative_target:,}")

    # -----------------------------------------------------------------
    # Load temporal split
    # -----------------------------------------------------------------

    print("\nLoading temporal split...")

    data = np.load(INPUT_PATH, allow_pickle=True)

    required_keys = [
        "train_sequences",
        "train_time_deltas",
        "train_labels",
        "train_session_ids",
        "train_session_start_times",
        "val_sequences",
        "val_time_deltas",
        "val_labels",
        "val_session_ids",
        "val_session_start_times",
        "test_sequences",
        "test_time_deltas",
        "test_labels",
        "test_session_ids",
        "test_session_start_times",
        "vocabulary_codes",
        "vocabulary_items",
    ]

    require_keys(data, required_keys)

    # -----------------------------------------------------------------
    # Extract arrays
    # -----------------------------------------------------------------

    train_sequences = data["train_sequences"]
    train_time_deltas = data["train_time_deltas"]
    train_labels = data["train_labels"]
    train_session_ids = data["train_session_ids"]
    train_session_start_times = data["train_session_start_times"]

    val_sequences = data["val_sequences"]
    val_time_deltas = data["val_time_deltas"]
    val_labels = data["val_labels"]
    val_session_ids = data["val_session_ids"]
    val_session_start_times = data["val_session_start_times"]

    test_sequences = data["test_sequences"]
    test_time_deltas = data["test_time_deltas"]
    test_labels = data["test_labels"]
    test_session_ids = data["test_session_ids"]
    test_session_start_times = data["test_session_start_times"]

    vocabulary_codes = data["vocabulary_codes"]
    vocabulary_items = data["vocabulary_items"]

    # -----------------------------------------------------------------
    # Validate array alignment
    # -----------------------------------------------------------------

    verify_alignment(
        train_sequences,
        train_time_deltas,
        train_labels,
        train_session_ids,
        "Training",
    )

    verify_alignment(
        val_sequences,
        val_time_deltas,
        val_labels,
        val_session_ids,
        "Validation",
    )

    verify_alignment(
        test_sequences,
        test_time_deltas,
        test_labels,
        test_session_ids,
        "Test",
    )

    # -----------------------------------------------------------------
    # Validate session uniqueness
    # -----------------------------------------------------------------

    print("\nChecking session uniqueness...")

    verify_session_uniqueness(train_session_ids, "Training")
    verify_session_uniqueness(val_session_ids, "Validation")
    verify_session_uniqueness(test_session_ids, "Test")

    verify_no_overlap(
        train_session_ids,
        val_session_ids,
        test_session_ids,
    )

    print("No duplicate sessions or cross-split overlap detected.")

    # -----------------------------------------------------------------
    # Class distribution before sampling
    # -----------------------------------------------------------------

    train_negative, train_positive = class_counts(train_labels)
    val_negative, val_positive = class_counts(val_labels)
    test_negative, test_positive = class_counts(test_labels)

    print("\nOriginal temporal split:")
    print(
        f"Training   : {len(train_labels):,} "
        f"({train_positive / len(train_labels) * 100:.2f}% positive)"
    )
    print(
        f"Validation : {len(val_labels):,} "
        f"({val_positive / len(val_labels) * 100:.2f}% positive)"
    )
    print(
        f"Test       : {len(test_labels):,} "
        f"({test_positive / len(test_labels) * 100:.2f}% positive)"
    )

    # -----------------------------------------------------------------
    # Check that enough samples exist
    # -----------------------------------------------------------------

    if train_positive < positive_target:
        raise ValueError(
            f"Not enough positive training sessions.\n"
            f"Available: {train_positive:,}\n"
            f"Required : {positive_target:,}"
        )

    if train_negative < negative_target:
        raise ValueError(
            f"Not enough negative training sessions.\n"
            f"Available: {train_negative:,}\n"
            f"Required : {negative_target:,}"
        )

    # -----------------------------------------------------------------
    # Find class indices
    # -----------------------------------------------------------------

    positive_indices = np.flatnonzero(train_labels == 1)
    negative_indices = np.flatnonzero(train_labels == 0)

    # -----------------------------------------------------------------
    # Reproducible sampling WITHOUT replacement
    # -----------------------------------------------------------------

    print("\nSampling training sessions...")

    rng = np.random.default_rng(RANDOM_SEED)

    selected_positive = rng.choice(
        positive_indices,
        size=positive_target,
        replace=False,
    )

    selected_negative = rng.choice(
        negative_indices,
        size=negative_target,
        replace=False,
    )

    selected_indices = np.concatenate(
        [
            selected_positive,
            selected_negative,
        ]
    )

    # Shuffle the selected training rows once.
    shuffle_order = rng.permutation(len(selected_indices))

    selected_indices = selected_indices[shuffle_order]

    # -----------------------------------------------------------------
    # Create bounded training dataset
    # -----------------------------------------------------------------

    balanced_train_sequences = train_sequences[selected_indices]
    balanced_train_time_deltas = train_time_deltas[selected_indices]
    balanced_train_labels = train_labels[selected_indices]
    balanced_train_session_ids = train_session_ids[selected_indices]
    balanced_train_session_start_times = (
        train_session_start_times[selected_indices]
    )

    # -----------------------------------------------------------------
    # Verify final training distribution
    # -----------------------------------------------------------------

    final_negative, final_positive = class_counts(
        balanced_train_labels
    )

    if final_negative != negative_target:
        raise RuntimeError(
            "Final negative class count is incorrect."
        )

    if final_positive != positive_target:
        raise RuntimeError(
            "Final positive class count is incorrect."
        )

    final_ratio = final_positive / len(balanced_train_labels)

    if not np.isclose(final_ratio, 0.50):
        raise RuntimeError(
            f"Final training class ratio is not 50:50: "
            f"{final_ratio:.6f}"
        )

    # -----------------------------------------------------------------
    # Verify sampled training sessions don't overlap with validation/test
    # -----------------------------------------------------------------

    verify_no_overlap(
        balanced_train_session_ids,
        val_session_ids,
        test_session_ids,
    )

    # -----------------------------------------------------------------
    # Verify validation/test are unchanged
    # -----------------------------------------------------------------

    original_val_ids = val_session_ids.copy()
    original_test_ids = test_session_ids.copy()

    # Since we never modify these arrays, this explicitly verifies
    # that their contents are unchanged before saving.
    if not np.array_equal(val_session_ids, original_val_ids):
        raise RuntimeError("Validation session IDs were modified.")

    if not np.array_equal(test_session_ids, original_test_ids):
        raise RuntimeError("Test session IDs were modified.")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        OUTPUT_PATH,

        # -------------------------------------------------------------
        # Balanced training data
        # -------------------------------------------------------------
        train_sequences=balanced_train_sequences,
        train_time_deltas=balanced_train_time_deltas,
        train_labels=balanced_train_labels,
        train_session_ids=balanced_train_session_ids,
        train_session_start_times=balanced_train_session_start_times,

        # -------------------------------------------------------------
        # ORIGINAL temporal validation data — unchanged
        # -------------------------------------------------------------
        val_sequences=val_sequences,
        val_time_deltas=val_time_deltas,
        val_labels=val_labels,
        val_session_ids=val_session_ids,
        val_session_start_times=val_session_start_times,

        # -------------------------------------------------------------
        # ORIGINAL temporal test data — unchanged
        # -------------------------------------------------------------
        test_sequences=test_sequences,
        test_time_deltas=test_time_deltas,
        test_labels=test_labels,
        test_session_ids=test_session_ids,
        test_session_start_times=test_session_start_times,

        # -------------------------------------------------------------
        # Vocabulary
        # -------------------------------------------------------------
        vocabulary_codes=vocabulary_codes,
        vocabulary_items=vocabulary_items,

        # -------------------------------------------------------------
        # Experiment metadata
        # -------------------------------------------------------------
        random_seed=np.array(RANDOM_SEED),
        target_training_sessions=np.array(TARGET_TRAINING_SESSIONS),
        target_positive_sessions=np.array(positive_target),
        target_negative_sessions=np.array(negative_target),
        balance_method=np.array(
            "Random class-balanced sampling without replacement"
        ),
        training_balanced=np.array(True),
        validation_balanced=np.array(False),
        test_balanced=np.array(False),
        validation_test_unchanged=np.array(True),
        temporal_split_preserved=np.array(True),
    )

    # -----------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("BALANCING COMPLETE")
    print("=" * 70)

    print("\nTraining:")
    print(
        f"  Sessions : {len(balanced_train_labels):,}"
    )
    print(
        f"  Positive : {final_positive:,}"
    )
    print(
        f"  Negative : {final_negative:,}"
    )
    print(
        f"  Positive rate: {final_ratio * 100:.2f}%"
    )

    print("\nValidation — UNCHANGED:")
    print(
        f"  Sessions : {len(val_labels):,}"
    )
    print(
        f"  Positive : {val_positive:,}"
    )
    print(
        f"  Negative : {val_negative:,}"
    )
    print(
        f"  Positive rate: "
        f"{val_positive / len(val_labels) * 100:.2f}%"
    )

    print("\nTest — UNCHANGED:")
    print(
        f"  Sessions : {len(test_labels):,}"
    )
    print(
        f"  Positive : {test_positive:,}"
    )
    print(
        f"  Negative : {test_negative:,}"
    )
    print(
        f"  Positive rate: "
        f"{test_positive / len(test_labels) * 100:.2f}%"
    )

    print("\nOutput:")
    print(f"  {OUTPUT_PATH}")

    print("\nReproducibility:")
    print(f"  Random seed: {RANDOM_SEED}")
    print("  Sampling: WITHOUT replacement")
    print("  Training balance: 50:50")
    print("  Validation modified: NO")
    print("  Test modified: NO")
    print("  Temporal split preserved: YES")

    print("=" * 70)


if __name__ == "__main__":
    main()