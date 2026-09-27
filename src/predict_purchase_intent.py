import os
import numpy as np
import lightgbm as lgb


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "machine_learning",
    "lightgbm_model.txt"
)

# Use the threshold selected during model validation.
# Change this to the exact tuned threshold from your
# lightgbm_metrics.csv if it is different.
THRESHOLD = 0.50


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("Loading trained LightGBM model...")

model = lgb.Booster(
    model_file=MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def create_features(item_ids, time_deltas):
    """
    Convert one user's current-session interaction data
    into the same 14 features used during model training.
    """

    item_ids = np.asarray(item_ids, dtype=np.float32)
    time_deltas = np.asarray(time_deltas, dtype=np.float32)

    # Remove padding / invalid item IDs
    valid_items = item_ids[item_ids > 0]

    if len(valid_items) == 0:
        valid_items = np.array([0], dtype=np.float32)

    # -----------------------------
    # Item interaction features
    # -----------------------------

    interaction_count = len(valid_items)

    unique_item_count = len(np.unique(valid_items))

    repeated_item_count = (
        interaction_count - unique_item_count
    )

    first_item_id = valid_items[0]

    last_item_id = valid_items[-1]

    mean_item_id = np.mean(valid_items)

    std_item_id = np.std(valid_items)

    min_item_id = np.min(valid_items)

    max_item_id = np.max(valid_items)

    # -----------------------------
    # Time features
    # -----------------------------

    valid_deltas = time_deltas[time_deltas > 0]

    if len(valid_deltas) == 0:

        total_time_delta = 0

        mean_time_delta = 0

        max_time_delta = 0

        std_time_delta = 0

        nonzero_time_intervals = 0

    else:

        total_time_delta = np.sum(valid_deltas)

        mean_time_delta = np.mean(valid_deltas)

        max_time_delta = np.max(valid_deltas)

        std_time_delta = np.std(valid_deltas)

        nonzero_time_intervals = len(valid_deltas)

    # ========================================================
    # EXACT 14 FEATURES
    # ========================================================

    features = np.array([
        interaction_count,
        unique_item_count,
        repeated_item_count,
        first_item_id,
        last_item_id,
        mean_item_id,
        std_item_id,
        min_item_id,
        max_item_id,
        total_time_delta,
        mean_time_delta,
        max_time_delta,
        std_time_delta,
        nonzero_time_intervals
    ], dtype=np.float32)

    return features


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_purchase_intent(item_ids, time_deltas):

    features = create_features(
        item_ids,
        time_deltas
    )

    # LightGBM expects a 2-dimensional input:
    # one row = one user/session
    features = features.reshape(1, -1)

    # IMPORTANT:
    # predict() only performs inference.
    # It does NOT train the model.
    probability = model.predict(features)[0]

    prediction = int(
        probability >= THRESHOLD
    )

    if prediction == 1:

        result = "Purchase"

    else:

        result = "No Purchase"

    return {
        "prediction": result,
        "purchase_probability": float(probability),
        "threshold": THRESHOLD
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # Example user session
    example_item_ids = [
        101,
        105,
        105,
        120,
        130
    ]

    # Time spent / time intervals between interactions
    example_time_deltas = [
        0,
        5,
        12,
        20,
        30
    ]

    result = predict_purchase_intent(
        example_item_ids,
        example_time_deltas
    )

    print("\nPrediction:")
    print(result)