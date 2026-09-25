from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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

MODEL_PATH = Path("models/deep_learning/gru_attention_model.keras")
METRICS_PATH = Path("results/metrics/gru_attention_metrics.csv")
HISTORY_PATH = Path("results/figures/gru_attention_training_history.png")
CM_PATH = Path("results/figures/gru_attention_confusion_matrix.png")
ROC_PATH = Path("results/figures/gru_attention_roc_curve.png")


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Limit TensorFlow CPU threads
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("GRU + ATTENTION PURCHASE-INTENT MODEL (BALANCED DATASET)")
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
# LOAD TRAIN / VALIDATION / TEST ARRAYS
# ============================================================

X_train = data["train_sequences"]
X_val = data["val_sequences"]
X_test = data["test_sequences"]

y_train = data["train_labels"]
y_val = data["val_labels"]
y_test = data["test_labels"]

# Convert to consistent datatypes
X_train = np.asarray(X_train).astype(np.int32)
X_val = np.asarray(X_val).astype(np.int32)
X_test = np.asarray(X_test).astype(np.int32)

y_train = np.asarray(y_train).astype(np.int32)
y_val = np.asarray(y_val).astype(np.int32)
y_test = np.asarray(y_test).astype(np.int32)


# ============================================================
# DATA INFORMATION
# ============================================================

print("\nDataset shapes:")
print(f"Training sequences   : {X_train.shape}")
print(f"Validation sequences : {X_val.shape}")
print(f"Test sequences       : {X_test.shape}")

sequence_length = X_train.shape[1]

vocab_size = (
    int(max(X_train.max(), X_val.max(), X_test.max())) + 1
)

print(f"\nSequence length      : {sequence_length}")
print(f"Vocabulary size      : {vocab_size}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_positive = int(np.sum(y_train == 1))
train_negative = int(np.sum(y_train == 0))

val_positive = int(np.sum(y_val == 1))
val_negative = int(np.sum(y_val == 0))

test_positive = int(np.sum(y_test == 1))
test_negative = int(np.sum(y_test == 0))

print("\nClass distribution:")
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
# GRU + ATTENTION MODEL
# ============================================================

print("\nBuilding GRU + Attention model...")

inputs = tf.keras.layers.Input(
    shape=(sequence_length,), name="sequence_input"
)

embedding = tf.keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=EMBEDDING_DIM,
    mask_zero=True,
    name="embedding",
)(inputs)

# GRU sequence output for attention mechanism
gru_output = tf.keras.layers.GRU(
    GRU_UNITS, return_sequences=True, name="gru"
)(embedding)

# Self-attention over the GRU sequence
attention_output = tf.keras.layers.Attention(name="attention")(
    [gru_output, gru_output]
)

# Residual additive connection
combined = tf.keras.layers.Add(name="attention_residual_add")(
    [gru_output, attention_output]
)

pooled = tf.keras.layers.GlobalAveragePooling1D(
    name="global_average_pooling"
)(combined)

dropout = tf.keras.layers.Dropout(0.30, name="dropout")(pooled)

outputs = tf.keras.layers.Dense(
    1, activation="sigmoid", name="purchase_probability"
)(dropout)

model = tf.keras.Model(
    inputs=inputs, outputs=outputs, name="GRU_Attention_Model"
)


# ============================================================
# COMPILE
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
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("TRAINING GRU + ATTENTION (BALANCED TRAINING SET)")
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
    verbose=1,
)

training_time = time.perf_counter() - start_time

print(f"\nTraining time: {training_time:.2f} seconds")
print(f"Epochs completed: {len(history.history['loss'])}")


# ============================================================
# TEST PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("TESTING GRU + ATTENTION ON ORIGINAL TEST DISTRIBUTION")
print("=" * 60)

prediction_start = time.perf_counter()

y_probability = model.predict(
    X_test, batch_size=BATCH_SIZE, verbose=0
).ravel()

prediction_time = time.perf_counter() - prediction_start

y_pred = (y_probability >= 0.5).astype(int)


# ============================================================
# EVALUATION METRICS
# ============================================================

precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)
cm = confusion_matrix(y_test, y_pred)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nGRU + Attention Test Results")
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
        zero_division=0,
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
    "model": "GRU_Attention",
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "training_time_seconds": training_time,
    "prediction_time_seconds": prediction_time,
    "epochs": len(history.history["loss"]),
    "batch_size": BATCH_SIZE,
    "gru_units": GRU_UNITS,
    "embedding_dim": EMBEDDING_DIM,
    "attention": "Keras_Attention",
}])

metrics_df.to_csv(METRICS_PATH, index=False)
print(f"Saved metrics -> {METRICS_PATH}")


# ============================================================
# SAVE TRAINING HISTORY FIGURE
# ============================================================

HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GRU + Attention Training and Validation Loss (Balanced Training Set)")
plt.legend()
plt.tight_layout()
plt.savefig(HISTORY_PATH, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved training figure -> {HISTORY_PATH}")


# ============================================================
# SAVE CONFUSION MATRIX FIGURE
# ============================================================

CM_PATH.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("GRU + Attention Confusion Matrix (Balanced Training)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], ["No Purchase", "Purchase"])
plt.yticks([0, 1], ["No Purchase", "Purchase"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            f"{cm[i, j]:,}",
            ha="center",
            va="center",
            color="white" if cm[i, j] > (cm.max() / 2) else "black",
        )

plt.tight_layout()
plt.savefig(CM_PATH, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved confusion matrix -> {CM_PATH}")


# ============================================================
# SAVE ROC CURVE
# ============================================================

ROC_PATH.parent.mkdir(parents=True, exist_ok=True)

fpr, tpr, _ = roc_curve(y_test, y_probability)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("GRU + Attention ROC Curve (Balanced Training)")
plt.legend()
plt.tight_layout()
plt.savefig(ROC_PATH, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved ROC curve -> {ROC_PATH}")


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("GRU + ATTENTION EXPERIMENT COMPLETED")
print("=" * 60)