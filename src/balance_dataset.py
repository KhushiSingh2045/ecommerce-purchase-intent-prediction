"""
STEP - BALANCE YOOCHOOSE TRAINING DATASET

Purpose:
    Create a balanced training dataset for the e-commerce
    purchase-intent prediction project.

Important methodology:
    - Only the TRAINING set is balanced.
    - Validation set remains unchanged.
    - Test set remains unchanged.
    - No SMOTE is used because item IDs are categorical
      sequence values.
    - Positive purchase sessions are randomly oversampled
      with replacement.
    - Final training distribution = 50% Purchase / 50% No Purchase.
    - Validation and test distributions remain natural.


"""

import os
import random
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

INPUT_FILE = os.path.join(
    "data",
    "processed",
    "yoochoose_dl_subset.npz"
)

OUTPUT_FILE = os.path.join(
    "data",
    "processed",
    "yoochoose_balanced_train.npz"
)

TARGET_RATIO = 0.50


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# PRINT CLASS DISTRIBUTION
# ============================================================

def print_distribution(name, labels):

    labels = np.asarray(labels)

    total = len(labels)

    purchase_count = int(
        np.sum(labels == 1)
    )

    no_purchase_count = int(
        np.sum(labels == 0)
    )

    purchase_percentage = (
        purchase_count / total * 100
        if total > 0 else 0
    )

    no_purchase_percentage = (
        no_purchase_count / total * 100
        if total > 0 else 0
    )

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(
        f"Total samples       : {total:,}"
    )

    print(
        f"Purchase            : "
        f"{purchase_count:,} "
        f"({purchase_percentage:.2f}%)"
    )

    print(
        f"No Purchase         : "
        f"{no_purchase_count:,} "
        f"({no_purchase_percentage:.2f}%)"
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("=" * 80)
    print("BALANCING YOOCHOOSE TRAINING DATASET")
    print("=" * 80)

    print("\nInput file:")
    print(INPUT_FILE)

    if not os.path.exists(INPUT_FILE):

        raise FileNotFoundError(
            f"""
Dataset not found:

{INPUT_FILE}

Make sure you have already generated:

yoochoose_dl_subset.npz
"""
        )

    data = np.load(
        INPUT_FILE,
        allow_pickle=True
    )

    required_keys = [
        "train_sequences",
        "train_time_deltas",
        "train_labels",
        "train_session_ids",

        "val_sequences",
        "val_time_deltas",
        "val_labels",
        "val_session_ids",

        "test_sequences",
        "test_time_deltas",
        "test_labels",
        "test_session_ids",
    ]

    for key in required_keys:

        if key not in data:

            raise KeyError(
                f"Required dataset key missing: {key}"
            )

    dataset = {

        "train_sequences":
            data["train_sequences"],

        "train_time_deltas":
            data["train_time_deltas"],

        "train_labels":
            data["train_labels"],

        "train_session_ids":
            data["train_session_ids"],

        "val_sequences":
            data["val_sequences"],

        "val_time_deltas":
            data["val_time_deltas"],

        "val_labels":
            data["val_labels"],

        "val_session_ids":
            data["val_session_ids"],

        "test_sequences":
            data["test_sequences"],

        "test_time_deltas":
            data["test_time_deltas"],

        "test_labels":
            data["test_labels"],

        "test_session_ids":
            data["test_session_ids"],
    }

    print("\nDataset loaded successfully.")

    return dataset


# ============================================================
# BALANCE TRAINING DATA
# ============================================================

def balance_training_data(dataset):

    X_train = np.asarray(
        dataset["train_sequences"]
    )

    time_train = np.asarray(
        dataset["train_time_deltas"]
    )

    y_train = np.asarray(
        dataset["train_labels"]
    )

    session_train = np.asarray(
        dataset["train_session_ids"]
    )

    print_distribution(
        "ORIGINAL TRAINING DATA",
        y_train
    )

    # --------------------------------------------------------
    # Find classes
    # --------------------------------------------------------

    negative_indices = np.where(
        y_train == 0
    )[0]

    positive_indices = np.where(
        y_train == 1
    )[0]

    if len(negative_indices) == 0:
        raise ValueError(
            "No negative samples found."
        )

    if len(positive_indices) == 0:
        raise ValueError(
            "No positive samples found."
        )

    print("\nOriginal class counts:")

    print(
        f"No Purchase : "
        f"{len(negative_indices):,}"
    )

    print(
        f"Purchase    : "
        f"{len(positive_indices):,}"
    )

    # --------------------------------------------------------
    # Target size
    #
    # To obtain 50:50:
    #
    # number of positive samples =
    # number of negative samples
    #
    # We keep ALL negative samples and
    # oversample positive samples.
    # --------------------------------------------------------

    target_per_class = max(
        len(negative_indices),
        len(positive_indices)
    )

    print(
        f"\nTarget samples per class: "
        f"{target_per_class:,}"
    )

    # --------------------------------------------------------
    # Keep all negative samples
    # --------------------------------------------------------

    balanced_negative_indices = (
        negative_indices.copy()
    )

    # --------------------------------------------------------
    # Oversample positive samples
    #
    # replace=True allows purchase sessions
    # to appear more than once.
    # --------------------------------------------------------

    balanced_positive_indices = np.random.choice(
        positive_indices,
        size=target_per_class,
        replace=True
    )

    # --------------------------------------------------------
    # Combine indices
    # --------------------------------------------------------

    balanced_indices = np.concatenate(
        [
            balanced_negative_indices,
            balanced_positive_indices
        ]
    )

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    np.random.shuffle(
        balanced_indices
    )

    # --------------------------------------------------------
    # Create balanced arrays
    # --------------------------------------------------------

    X_balanced = X_train[
        balanced_indices
    ]

    time_balanced = time_train[
        balanced_indices
    ]

    y_balanced = y_train[
        balanced_indices
    ]

    session_balanced = session_train[
        balanced_indices
    ]

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    print_distribution(
        "BALANCED TRAINING DATA",
        y_balanced
    )

    return (
        X_balanced,
        time_balanced,
        y_balanced,
        session_balanced
    )


# ============================================================
# CHECK DUPLICATES CREATED BY OVERSAMPLING
# ============================================================

def report_oversampling(
    original_labels,
    balanced_labels
):

    original_positive = int(
        np.sum(original_labels == 1)
    )

    balanced_positive = int(
        np.sum(balanced_labels == 1)
    )

    additional_positive = (
        balanced_positive -
        original_positive
    )

    print("\n" + "-" * 70)
    print("OVERSAMPLING INFORMATION")
    print("-" * 70)

    print(
        f"Original purchase sessions : "
        f"{original_positive:,}"
    )

    print(
        f"Balanced purchase samples  : "
        f"{balanced_positive:,}"
    )

    print(
        f"Additional purchase copies : "
        f"{additional_positive:,}"
    )

    if original_positive > 0:

        factor = (
            balanced_positive /
            original_positive
        )

        print(
            f"Oversampling factor       : "
            f"{factor:.2f}x"
        )


# ============================================================
# SAVE BALANCED DATASET
# ============================================================

def save_dataset(
    dataset,
    X_balanced,
    time_balanced,
    y_balanced,
    session_balanced
):

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    np.savez_compressed(

        OUTPUT_FILE,

        # ----------------------------------------------------
        # BALANCED TRAINING DATA
        # ----------------------------------------------------

        train_sequences=X_balanced,

        train_time_deltas=time_balanced,

        train_labels=y_balanced,

        train_session_ids=session_balanced,

        # ----------------------------------------------------
        # ORIGINAL VALIDATION DATA
        # ----------------------------------------------------

        val_sequences=dataset[
            "val_sequences"
        ],

        val_time_deltas=dataset[
            "val_time_deltas"
        ],

        val_labels=dataset[
            "val_labels"
        ],

        val_session_ids=dataset[
            "val_session_ids"
        ],

        # ----------------------------------------------------
        # ORIGINAL TEST DATA
        # ----------------------------------------------------

        test_sequences=dataset[
            "test_sequences"
        ],

        test_time_deltas=dataset[
            "test_time_deltas"
        ],

        test_labels=dataset[
            "test_labels"
        ],

        test_session_ids=dataset[
            "test_session_ids"
        ],

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        random_seed=RANDOM_SEED,

        balance_method="Random Oversampling",

        target_training_ratio=TARGET_RATIO,

        training_balanced=True,

        validation_balanced=False,

        test_balanced=False,
    )

    print("\n" + "=" * 80)
    print("BALANCED DATASET SAVED")
    print("=" * 80)

    print(
        f"\nOutput file:\n{OUTPUT_FILE}"
    )


# ============================================================
# VERIFY NO SESSION LEAKAGE
# ============================================================

def check_session_overlap(
    train_ids,
    val_ids,
    test_ids
):

    train_set = set(
        train_ids.tolist()
    )

    val_set = set(
        val_ids.tolist()
    )

    test_set = set(
        test_ids.tolist()
    )

    train_val = train_set.intersection(
        val_set
    )

    train_test = train_set.intersection(
        test_set
    )

    val_test = val_set.intersection(
        test_set
    )

    print("\n" + "-" * 70)
    print("SESSION OVERLAP CHECK")
    print("-" * 70)

    print(
        f"Train ∩ Validation : "
        f"{len(train_val)}"
    )

    print(
        f"Train ∩ Test       : "
        f"{len(train_test)}"
    )

    print(
        f"Validation ∩ Test  : "
        f"{len(val_test)}"
    )

    if (
        len(train_val) == 0
        and
        len(train_test) == 0
        and
        len(val_test) == 0
    ):

        print(
            "\n[PASS] No session leakage "
            "between splits."
        )

    else:

        raise ValueError(
            "\nSession leakage detected."
        )


# ============================================================
# VERIFY FINAL DATASET
# ============================================================

def verify_saved_dataset():

    print("\n" + "=" * 80)
    print("VERIFYING SAVED DATASET")
    print("=" * 80)

    data = np.load(
        OUTPUT_FILE,
        allow_pickle=True
    )

    train_labels = data[
        "train_labels"
    ]

    val_labels = data[
        "val_labels"
    ]

    test_labels = data[
        "test_labels"
    ]

    # --------------------------------------------------------
    # Training balance
    # --------------------------------------------------------

    train_purchase = np.sum(
        train_labels == 1
    )

    train_no_purchase = np.sum(
        train_labels == 0
    )

    print(
        f"\nTraining purchase     : "
        f"{train_purchase:,}"
    )

    print(
        f"Training no-purchase  : "
        f"{train_no_purchase:,}"
    )

    if train_purchase == train_no_purchase:

        print(
            "[PASS] Training dataset is "
            "exactly 50:50."
        )

    else:

        raise ValueError(
            "Training dataset is not balanced."
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print_distribution(
        "VALIDATION DATA",
        val_labels
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print_distribution(
        "TEST DATA",
        test_labels
    )

    # --------------------------------------------------------
    # Session leakage
    # --------------------------------------------------------

    check_session_overlap(
        data["train_session_ids"],
        data["val_session_ids"],
        data["test_session_ids"]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    dataset = load_dataset()

    # --------------------------------------------------------
    # Original distribution
    # --------------------------------------------------------

    original_train_labels = np.asarray(
        dataset["train_labels"]
    )

    # --------------------------------------------------------
    # Balance
    # --------------------------------------------------------

    (
        X_balanced,
        time_balanced,
        y_balanced,
        session_balanced
    ) = balance_training_data(
        dataset
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_oversampling(
        original_train_labels,
        y_balanced
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_dataset(
        dataset,
        X_balanced,
        time_balanced,
        y_balanced,
        session_balanced
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_saved_dataset()

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print(
        """
Your new dataset is ready.

Training:
    50% Purchase
    50% No Purchase

Validation:
    Original distribution retained

Test:
    Original distribution retained


"""
    )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()