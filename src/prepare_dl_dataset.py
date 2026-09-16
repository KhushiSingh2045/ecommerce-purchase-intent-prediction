import os
import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/yoochoose_sequences.npz"

OUTPUT_FILE = "data/processed/yoochoose_dl_subset.npz"

# Number of sessions to use for the deep-learning experiment
SAMPLE_SIZE = 100_000

# Reproducibility
RANDOM_SEED = 42

# Train / validation / test split
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# LOAD ORIGINAL PROCESSED SEQUENCE DATA
# ============================================================

print("=" * 60)
print("PREPARING YOOCHOOSE DEEP-LEARNING DATASET")
print("=" * 60)

print("\nLoading:")
print(INPUT_FILE)

data = np.load(INPUT_FILE)

sequences = data["sequences"]
time_deltas = data["time_deltas"]
labels = data["labels"]
session_ids = data["session_ids"]

print("\nOriginal dataset:")
print(f"Sessions      : {len(labels):,}")
print(f"Sequences     : {sequences.shape}")
print(f"Time deltas   : {time_deltas.shape}")
print(f"Labels        : {labels.shape}")
print(f"Session IDs   : {session_ids.shape}")


# ============================================================
# CHECK DATA SIZE
# ============================================================

if SAMPLE_SIZE > len(labels):
    raise ValueError(
        f"SAMPLE_SIZE ({SAMPLE_SIZE:,}) is larger than "
        f"the available sessions ({len(labels):,})."
    )


# ============================================================
# CHECK ORIGINAL CLASS DISTRIBUTION
# ============================================================

positive_count = np.sum(labels == 1)
negative_count = np.sum(labels == 0)

print("\nOriginal class distribution:")
print(f"Purchase (1)     : {positive_count:,}")
print(f"No Purchase (0)  : {negative_count:,}")
print(f"Purchase rate    : {positive_count / len(labels) * 100:.2f}%")


# ============================================================
# STRATIFIED RANDOM SAMPLING
# ============================================================

print("\n" + "-" * 60)
print(f"Selecting {SAMPLE_SIZE:,} sessions using stratified sampling...")
print("-" * 60)

# Create an index for every session
all_indices = np.arange(len(labels))

# Select a reproducible stratified subset
sample_indices, _ = train_test_split(
    all_indices,
    train_size=SAMPLE_SIZE,
    stratify=labels,
    random_state=RANDOM_SEED
)

# Sort indices for more predictable storage/access
sample_indices = np.sort(sample_indices)

# Extract selected data
sample_sequences = sequences[sample_indices]
sample_time_deltas = time_deltas[sample_indices]
sample_labels = labels[sample_indices]
sample_session_ids = session_ids[sample_indices]


# ============================================================
# CHECK SAMPLE CLASS DISTRIBUTION
# ============================================================

sample_positive = np.sum(sample_labels == 1)
sample_negative = np.sum(sample_labels == 0)

print("\nSelected dataset:")
print(f"Sessions         : {len(sample_labels):,}")
print(f"Sequences        : {sample_sequences.shape}")
print(f"Time deltas      : {sample_time_deltas.shape}")

print("\nSelected class distribution:")
print(f"Purchase (1)     : {sample_positive:,}")
print(f"No Purchase (0)  : {sample_negative:,}")
print(f"Purchase rate    : {sample_positive / len(sample_labels) * 100:.2f}%")


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "-" * 60)
print("Creating train / validation / test split...")
print("-" * 60)

# First split:
# 70% training
# 30% temporary data
train_idx, temp_idx = train_test_split(
    np.arange(len(sample_labels)),
    test_size=(VALIDATION_SIZE + TEST_SIZE),
    stratify=sample_labels,
    random_state=RANDOM_SEED
)

# Second split:
# Temporary 30% -> 15% validation + 15% test
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE),
    stratify=sample_labels[temp_idx],
    random_state=RANDOM_SEED
)


# ============================================================
# EXTRACT SPLITS
# ============================================================

train_sequences = sample_sequences[train_idx]
train_time_deltas = sample_time_deltas[train_idx]
train_labels = sample_labels[train_idx]
train_session_ids = sample_session_ids[train_idx]

val_sequences = sample_sequences[val_idx]
val_time_deltas = sample_time_deltas[val_idx]
val_labels = sample_labels[val_idx]
val_session_ids = sample_session_ids[val_idx]

test_sequences = sample_sequences[test_idx]
test_time_deltas = sample_time_deltas[test_idx]
test_labels = sample_labels[test_idx]
test_session_ids = sample_session_ids[test_idx]


# ============================================================
# VERIFY SESSION ID SEPARATION
# ============================================================

train_ids = set(train_session_ids.tolist())
val_ids = set(val_session_ids.tolist())
test_ids = set(test_session_ids.tolist())

if train_ids.intersection(val_ids):
    raise ValueError("Data leakage detected: train and validation sessions overlap.")

if train_ids.intersection(test_ids):
    raise ValueError("Data leakage detected: train and test sessions overlap.")

if val_ids.intersection(test_ids):
    raise ValueError("Data leakage detected: validation and test sessions overlap.")

print("\nSession overlap check:")
print("Train ∩ Validation : 0")
print("Train ∩ Test       : 0")
print("Validation ∩ Test  : 0")
print("Data leakage check : PASSED")


# ============================================================
# DISPLAY SPLIT INFORMATION
# ============================================================

print("\nDataset split:")
print(f"Training   : {len(train_labels):,} sessions")
print(f"Validation : {len(val_labels):,} sessions")
print(f"Test       : {len(test_labels):,} sessions")


# ============================================================
# DISPLAY CLASS DISTRIBUTION FOR EACH SPLIT
# ============================================================

print("\nClass distribution by split:")

print("\nTraining:")
print(f"Purchase     : {np.sum(train_labels == 1):,}")
print(f"No Purchase  : {np.sum(train_labels == 0):,}")

print("\nValidation:")
print(f"Purchase     : {np.sum(val_labels == 1):,}")
print(f"No Purchase  : {np.sum(val_labels == 0):,}")

print("\nTest:")
print(f"Purchase     : {np.sum(test_labels == 1):,}")
print(f"No Purchase  : {np.sum(test_labels == 0):,}")


# ============================================================
# SAVE DATASET
# ============================================================

output_directory = os.path.dirname(OUTPUT_FILE)

os.makedirs(output_directory, exist_ok=True)

np.savez_compressed(
    OUTPUT_FILE,

    # Training
    train_sequences=train_sequences,
    train_time_deltas=train_time_deltas,
    train_labels=train_labels,
    train_session_ids=train_session_ids,

    # Validation
    val_sequences=val_sequences,
    val_time_deltas=val_time_deltas,
    val_labels=val_labels,
    val_session_ids=val_session_ids,

    # Test
    test_sequences=test_sequences,
    test_time_deltas=test_time_deltas,
    test_labels=test_labels,
    test_session_ids=test_session_ids,

    # Metadata
    random_seed=np.array(RANDOM_SEED),
    sample_size=np.array(SAMPLE_SIZE)
)


# ============================================================
# FINAL MESSAGE
# ============================================================

file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETED")
print("=" * 60)

print(f"\nSaved file:")
print(OUTPUT_FILE)

print(f"\nFile size: {file_size_mb:.2f} MB")

print("\nThe dataset is ready for:")
print("1. LSTM training")
print("2. GRU training")
print("3. Attention-based sequential modeling")

print("\nRandom seed:", RANDOM_SEED)
print("Sample size:", f"{SAMPLE_SIZE:,}")
print("Split      : 70% Train / 15% Validation / 15% Test")

print("\nNext step:")
print("Train the LSTM model using src/train_dl.py")