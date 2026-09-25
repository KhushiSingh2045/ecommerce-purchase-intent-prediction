import os
import random
import time
import numpy as np
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
DATA_FILE = "data/processed/yoochoose_balanced_train.npz"

MODEL_DIR = "models/deep_learning"
RESULTS_DIR = "results/metrics"
FIGURES_DIR = "results/figures"

MODEL_FILE = os.path.join(MODEL_DIR, "lstm_model.keras")
METRICS_FILE = os.path.join(RESULTS_DIR, "lstm_metrics.csv")
HISTORY_FILE = os.path.join(FIGURES_DIR, "lstm_training_history.png")

RANDOM_SEED = 42

# Resource-conscious settings
BATCH_SIZE = 64
EPOCHS = 10
LSTM_UNITS = 32
EMBEDDING_DIM = 32

# Sequence length is determined from the dataset
MAX_SEQUENCE_LENGTH = 20

# Training controls
PATIENCE = 2


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Avoid TensorFlow using excessive CPU threads
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("YOOCHOOSE LSTM TRAINING (BALANCED TRAINING SET)")
print("=" * 60)

print("\nLoading dataset:")
print(DATA_FILE)


# ============================================================
# LOAD PREPARED DATASET
# ============================================================

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}\n\n"
        "Run this command first:\n"
        "python -m src.balance_dataset"
    )

data = np.load(DATA_FILE)

required_keys = [
    "train_sequences",
    "train_labels",
    "val_sequences",
    "val_labels",
    "test_sequences",
    "test_labels",
]

for key in required_keys:
    if key not in data:
        raise KeyError(f"Required dataset field '{key}' is missing.")


# ============================================================
# EXTRACT DATA
# ============================================================

X_train = data["train_sequences"]
y_train = data["train_labels"]

X_val = data["val_sequences"]
y_val = data["val_labels"]

X_test = data["test_sequences"]
y_test = data["test_labels"]


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nDataset information:")

print(f"Training sequences   : {X_train.shape}")
print(f"Validation sequences : {X_val.shape}")
print(f"Test sequences       : {X_test.shape}")

print(f"\nTraining samples     : {len(y_train):,}")
print(f"Validation samples   : {len(y_val):,}")
print(f"Test samples         : {len(y_test):,}")

print(f"\nSequence length      : {X_train.shape[1]}")
print(f"Vocabulary size      : {int(np.max(X_train)) + 1}")

print("\nClass distribution:")

print(
    f"Training   -> Purchase: {np.sum(y_train == 1):,}, "
    f"No Purchase: {np.sum(y_train == 0):,}"
)

print(
    f"Validation -> Purchase: {np.sum(y_val == 1):,}, "
    f"No Purchase: {np.sum(y_val == 0):,}"
)

print(
    f"Test       -> Purchase: {np.sum(y_test == 1):,}, "
    f"No Purchase: {np.sum(y_test == 0):,}"
)


# ============================================================
# DETERMINE VOCABULARY SIZE
# ============================================================

vocab_size = (
    int(max(np.max(X_train), np.max(X_val), np.max(X_test))) + 1
)

sequence_length = X_train.shape[1]

print("\nModel configuration:")
print(f"Vocabulary size : {vocab_size:,}")
print(f"Sequence length : {sequence_length}")
print(f"Embedding dim   : {EMBEDDING_DIM}")
print(f"LSTM units      : {LSTM_UNITS}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Maximum epochs  : {EPOCHS}")


# ============================================================
# BUILD LSTM MODEL
# ============================================================

print("\n" + "-" * 60)
print("Building LSTM model...")
print("-" * 60)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(sequence_length,)),
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
    ),
    tf.keras.layers.LSTM(LSTM_UNITS),
    tf.keras.layers.Dropout(0.30),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

print("\nModel summary:")
model.summary()


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=PATIENCE, restore_best_weights=True
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING LSTM TRAINING")
print("=" * 60)

start_time = time.time()

# Balanced training set: class_weight is no longer used
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1,
)

training_time = time.time() - start_time

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(f"\nTraining time: {training_time / 60:.2f} minutes")
print(f"Epochs completed: {len(history.history['loss'])}")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

prediction_start = time.time()

y_probability = model.predict(
    X_test, batch_size=BATCH_SIZE, verbose=0
).ravel()

prediction_time = time.time() - prediction_start

y_prediction = (y_probability >= 0.50).astype(np.int8)


# ============================================================
# EVALUATION
# ============================================================

precision = precision_score(y_test, y_prediction, zero_division=0)
recall = recall_score(y_test, y_prediction, zero_division=0)
f1 = f1_score(y_test, y_prediction, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)
cm = confusion_matrix(y_test, y_prediction)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LSTM TEST RESULTS")
print("=" * 60)

print(f"\nPrecision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_prediction,
        target_names=["No Purchase", "Purchase"],
        zero_division=0,
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_FILE)

print("\nSaved LSTM model:")
print(MODEL_FILE)


# ============================================================
# SAVE METRICS
# ============================================================

with open(METRICS_FILE, "w") as f:
    f.write(
        "model,precision,recall,f1_score,roc_auc,"
        "training_time_seconds,prediction_time_seconds,"
        "epochs,batch_size,lstm_units\n"
    )

    f.write(
        f"LSTM,"
        f"{precision:.6f},"
        f"{recall:.6f},"
        f"{f1:.6f},"
        f"{roc_auc:.6f},"
        f"{training_time:.2f},"
        f"{prediction_time:.2f},"
        f"{len(history.history['loss'])},"
        f"{BATCH_SIZE},"
        f"{LSTM_UNITS}\n"
    )

print("\nSaved metrics:")
print(METRICS_FILE)


# ============================================================
# SAVE TRAINING HISTORY FIGURE
# ============================================================

try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training and Validation Loss (Balanced Training Set)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(HISTORY_FILE, dpi=150)
    plt.close()

    print("\nSaved training figure:")
    print(HISTORY_FILE)

except Exception as e:
    print("\nTraining figure could not be saved:")
    print(e)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("LSTM EXPERIMENT FINISHED")
print("=" * 60)

print("\nFiles generated:")
print(f"Model   : {MODEL_FILE}")
print(f"Metrics : {METRICS_FILE}")
print(f"Figure  : {HISTORY_FILE}")

print("\nNext model:")
print("GRU")