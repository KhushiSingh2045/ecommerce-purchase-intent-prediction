from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import tensorflow as tf

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

BATCH_SIZE = 64
EPOCHS = 10
GRU_UNITS = 32
EMBEDDING_DIM = 32
PATIENCE = 2

DATA_FILE = Path("data/processed/yoochoose_balanced_train.npz")

MODEL_PATH = Path("models/deep_learning/gru_model.keras")
METRICS_PATH = Path("results/metrics/gru_metrics.csv")
FIGURE_PATH = Path("results/figures/gru_training_history.png")


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Limit TensorFlow threads
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("GRU PURCHASE-INTENT MODEL (BALANCED TRAINING SET)")
print("=" * 60)

print(f"\nLoading dataset from: {DATA_FILE}")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}\n\n"
        "Ensure you have generated the balanced dataset first using:\n"
        "python -m src.balance_dataset"
    )

data = np.load(DATA_FILE, allow_pickle=True)

print("\nAvailable dataset arrays:")
print(data.files)


# ============================================================
# FIND TRAIN / VALIDATION / TEST ARRAYS
# ============================================================

def find_array(possible_names):
    for name in possible_names:
        if name in data.files:
            return data[name]
    return None


X_train = find_array(["train_sequences", "X_train", "sequences_train"])
X_val = find_array(
    ["val_sequences", "X_val", "X_validation", "validation_sequences"]
)
X_test = find_array(["test_sequences", "X_test", "sequences_test"])

y_train = find_array(["train_labels", "y_train", "labels_train"])
y_val = find_array(
    ["val_labels", "y_val", "y_validation", "validation_labels"]
)
y_test = find_array(["test_labels", "y_test", "labels_test"])


# ============================================================
# IF SPLITS ARE NOT STORED, RECREATE THE SAME SPLIT
# ============================================================

if X_train is None or X_val is None or X_test is None:
    print("\nSaved train/validation/test arrays were not found.")
    print("Recreating the same 70/15/15 stratified split...")

    X = find_array(["sequences", "X"])
    y = find_array(["labels", "y"])

    if X is None or y is None:
        raise ValueError(
            "Could not find sequence and label arrays in the NPZ file."
        )

    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
    )


# Convert labels and sequences to integer arrays
y_train = np.asarray(y_train).astype(np.int32)
y_val = np.asarray(y_val).astype(np.int32)
y_test = np.asarray(y_test).astype(np.int32)

X_train = np.asarray(X_train).astype(np.int32)
X_val = np.asarray(X_val).astype(np.int32)
X_test = np.asarray(X_test).astype(np.int32)


# ============================================================
# DATA INFORMATION
# ============================================================

print("\nDataset shapes:")
print(f"Training sequences   : {X_train.shape}")
print(f"Validation sequences : {X_val.shape}")
print(f"Test sequences       : {X_test.shape}")

vocab_size = int(
    max(
        X_train.max(),
        X_val.max(),
        X_test.max()
    )
) + 1

print(f"\nVocabulary size      : {vocab_size}")

print("\nClass distribution:")

train_positive = int(np.sum(y_train == 1))
train_negative = int(np.sum(y_train == 0))

val_positive = int(np.sum(y_val == 1))
val_negative = int(np.sum(y_val == 0))

test_positive = int(np.sum(y_test == 1))
test_negative = int(np.sum(y_test == 0))

print(
    f"Training   -> Purchase: {train_positive:,}, "
    f"No Purchase: {train_negative:,}"
)

print(
    f"Validation -> Purchase: {val_positive:,}, "
    f"No Purchase: {val_negative:,}"
)

print(
    f"Test       -> Purchase: {test_positive:,}, "
    f"No Purchase: {test_negative:,}"
)


# ============================================================
# BUILD GRU MODEL
# ============================================================

print("\nBuilding GRU model...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),

    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ),

    tf.keras.layers.GRU(
        GRU_UNITS
    ),

    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

print("\nModel summary:")
model.summary()


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("TRAINING GRU (BALANCED TRAINING SET)")
print("=" * 60)

start_time = time.perf_counter()

# Balanced training set: class_weight is omitted
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)
training_time = time.perf_counter() - start_time

print(f"\nTraining time: {training_time:.2f} seconds")
print(f"Epochs completed: {len(history.history['loss'])}")


# ============================================================
# TEST PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("TESTING GRU ON ORIGINAL TEST DISTRIBUTION")
print("=" * 60)

prediction_start = time.perf_counter()

y_probability = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=0
).ravel()

prediction_time = time.perf_counter() - prediction_start

y_pred = (y_probability >= 0.5).astype(int)


# ============================================================
# EVALUATION
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nGRU Test Results")
print("-" * 40)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Purchase", "Purchase"],
        zero_division=0
    )
)

print(f"Training time   : {training_time:.2f} seconds")
print(f"Prediction time : {prediction_time:.2f} seconds")


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

model.save(MODEL_PATH)

print(f"\nSaved model -> {MODEL_PATH}")


# ============================================================
# SAVE METRICS
# ============================================================

METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

metrics_df = pd.DataFrame([{
    "model": "GRU",
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "training_time_seconds": training_time,
    "prediction_time_seconds": prediction_time,
    "epochs": len(history.history["loss"]),
    "batch_size": BATCH_SIZE,
    "gru_units": GRU_UNITS
}])

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)

print(f"Saved metrics -> {METRICS_PATH}")


# ============================================================
# SAVE TRAINING HISTORY FIGURE
# ============================================================

try:
    import matplotlib.pyplot as plt

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GRU Training and Validation Loss (Balanced Training Set)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved training figure -> {FIGURE_PATH}")

except ImportError:
    print(
        "\nMatplotlib is not installed. "
        "Training figure was not created."
    )


print("\n" + "=" * 60)
print("GRU EXPERIMENT COMPLETED")
print("=" * 60)