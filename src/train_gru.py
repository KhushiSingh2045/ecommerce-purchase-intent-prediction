"""
Train GRU on the YOOCHOOSE temporal split.

The script deliberately contains NO random train/validation/test
fallback. The temporal split must already exist in the NPZ file.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

DATA_FILE = Path(
    "data/processed/yoochoose_balanced_train.npz"
)

MODEL_FILE = Path(
    "models/deep_learning/gru_model.keras"
)

METRICS_FILE = Path(
    "results/metrics/gru_metrics.csv"
)

HISTORY_FILE = Path(
    "results/figures/gru_training_history.png"
)

BATCH_SIZE = 64
EPOCHS = 10
PATIENCE = 2

GRU_UNITS = 32
EMBEDDING_DIM = 32
DROPOUT = 0.30


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(
    RANDOM_SEED
)

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
    RANDOM_SEED
)

try:
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass


# ============================================================
# DIRECTORIES
# ============================================================

MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

HISTORY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD
# ============================================================

def load_dataset():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"""
Dataset not found:

{DATA_FILE}

Run:

python -m src.feature_engineering
python -m src.balance_dataset
"""
        )

    return np.load(
        DATA_FILE,
        allow_pickle=True,
    )


# ============================================================
# VALIDATE
# ============================================================

def validate_dataset(data):

    required = [
        "train_sequences",
        "train_labels",
        "val_sequences",
        "val_labels",
        "test_sequences",
        "test_labels",
        "vocabulary_codes",
    ]

    missing = [
        key
        for key in required
        if key not in data.files
    ]

    if missing:
        raise KeyError(
            "Missing dataset fields: "
            + ", ".join(missing)
        )

    X_train = data["train_sequences"]
    y_train = data["train_labels"]

    X_val = data["val_sequences"]
    y_val = data["val_labels"]

    X_test = data["test_sequences"]
    y_test = data["test_labels"]

    if X_train.ndim != 2:
        raise ValueError(
            "Training sequences must be 2-D."
        )

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            "Train/validation sequence length mismatch."
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Train/test sequence length mismatch."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "Training sequence/label mismatch."
        )

    if len(X_val) != len(y_val):
        raise ValueError(
            "Validation sequence/label mismatch."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "Test sequence/label mismatch."
        )

    positive = np.sum(y_train == 1)
    negative = np.sum(y_train == 0)

    if positive != negative:
        raise ValueError(
            "Training set is not balanced: "
            f"{positive:,} positive / "
            f"{negative:,} negative."
        )


# ============================================================
# VOCABULARY
# ============================================================

def get_vocabulary_size(data):

    codes = np.asarray(
        data["vocabulary_codes"]
    )

    if len(codes) == 0:
        raise ValueError(
            "Training vocabulary is empty."
        )

    return int(
        codes.max()
    ) + 1


# ============================================================
# MODEL
# ============================================================

def build_model(
    vocabulary_size,
    sequence_length,
):

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(sequence_length,)
            ),

            tf.keras.layers.Embedding(
                input_dim=vocabulary_size,
                output_dim=EMBEDDING_DIM,
                mask_zero=True,
                name="item_embedding",
            ),

            tf.keras.layers.GRU(
                GRU_UNITS,
                name="gru",
            ),

            tf.keras.layers.Dropout(
                DROPOUT
            ),

            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                name="purchase_probability",
            ),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            ),
        ],
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("YOOCHOOSE GRU — TEMPORAL SPLIT")
    print("=" * 70)

    data = load_dataset()

    validate_dataset(
        data
    )

    X_train = np.asarray(
        data["train_sequences"],
        dtype=np.int32,
    )

    y_train = np.asarray(
        data["train_labels"],
        dtype=np.int8,
    )

    X_val = np.asarray(
        data["val_sequences"],
        dtype=np.int32,
    )

    y_val = np.asarray(
        data["val_labels"],
        dtype=np.int8,
    )

    X_test = np.asarray(
        data["test_sequences"],
        dtype=np.int32,
    )

    y_test = np.asarray(
        data["test_labels"],
        dtype=np.int8,
    )

    vocabulary_size = get_vocabulary_size(
        data
    )

    sequence_length = X_train.shape[1]

    print(
        f"\nTrain      : {X_train.shape}"
    )

    print(
        f"Validation : {X_val.shape}"
    )

    print(
        f"Test       : {X_test.shape}"
    )

    print(
        f"\nVocabulary : {vocabulary_size:,}"
    )

    print(
        f"Sequence length : {sequence_length}"
    )

    print(
        f"Train purchase rate : "
        f"{np.mean(y_train) * 100:.2f}%"
    )

    print(
        f"Validation purchase rate : "
        f"{np.mean(y_val) * 100:.2f}%"
    )

    print(
        f"Test purchase rate : "
        f"{np.mean(y_test) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model(
        vocabulary_size,
        sequence_length,
    )

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting GRU training...")

    start = time.perf_counter()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
    )

    training_time = (
        time.perf_counter() - start
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    start = time.perf_counter()

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    ).ravel()

    prediction_time = (
        time.perf_counter() - start
    )

    predictions = (
        probabilities >= 0.50
    ).astype(np.int8)

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

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print("\n" + "=" * 70)
    print("GRU TEMPORAL TEST RESULTS")
    print("=" * 70)

    print(
        f"\nPrecision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No Purchase",
                "Purchase",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    model.save(
        MODEL_FILE
    )

    metrics = pd.DataFrame(
        [
            {
                "model": "GRU",
                "evaluation_split": "temporal_test",
                "threshold": 0.50,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "training_time_seconds":
                    training_time,
                "prediction_time_seconds":
                    prediction_time,
                "epochs":
                    len(
                        history.history["loss"]
                    ),
                "batch_size":
                    BATCH_SIZE,
                "gru_units":
                    GRU_UNITS,
                "embedding_dim":
                    EMBEDDING_DIM,
            }
        ]
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history.history["loss"],
        label="Training Loss",
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy")

    plt.title(
        "GRU Training and Validation Loss"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        HISTORY_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nSaved:")
    print(MODEL_FILE)
    print(METRICS_FILE)
    print(HISTORY_FILE)

    print("\nGRU experiment completed.")


if __name__ == "__main__":
    main()