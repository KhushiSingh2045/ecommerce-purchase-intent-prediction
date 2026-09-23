import os
import time
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

# Number of sessions used for the benchmark
SIMULATION_SESSIONS = 1000

# Number of partial-event stages evaluated for every session
EVENT_STAGES = 20

# Sessions used only for model warm-up
WARMUP_SESSIONS = 100

# Kept for compatibility/documentation
BATCH_SIZE = 64

DATA_PATH = "data/processed/yoochoose_dl_subset.npz"

LSTM_MODEL_PATH = "models/deep_learning/lstm_model.keras"
GRU_MODEL_PATH = "models/deep_learning/gru_model.keras"
GRU_ATTENTION_MODEL_PATH = (
    "models/deep_learning/gru_attention_model.keras"
)

XGBOOST_MODEL_PATH = (
    "models/machine_learning/xgboost_model.json"
)

LIGHTGBM_MODEL_PATH = (
    "models/machine_learning/lightgbm_model.txt"
)

METRICS_DIR = "results/metrics"
FIGURES_DIR = "results/figures"


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

warnings.filterwarnings("ignore")


# ============================================================
# TENSORFLOW CPU SETTINGS
# ============================================================

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
except Exception:
    pass


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# FEATURE NAMES
# ============================================================

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


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    print("\nLoading dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    data = np.load(DATA_PATH)

    required_keys = [
        "test_sequences",
        "test_time_deltas",
        "test_labels",
        "test_session_ids",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in data
    ]

    if missing_keys:
        raise KeyError(
            "The NPZ dataset is missing required arrays: "
            + ", ".join(missing_keys)
        )

    sequences = data["test_sequences"]
    time_deltas = data["test_time_deltas"]
    labels = data["test_labels"]
    session_ids = data["test_session_ids"]

    print(f"Test sequences       : {sequences.shape}")
    print(f"Time-deltas          : {time_deltas.shape}")
    print(f"Labels               : {labels.shape}")
    print(f"Session IDs          : {session_ids.shape}")

    if not (
        len(sequences)
        == len(time_deltas)
        == len(labels)
        == len(session_ids)
    ):
        raise ValueError(
            "Sequences, time deltas, labels and session IDs "
            "must contain the same number of sessions."
        )

    purchase_count = int(np.sum(labels == 1))
    no_purchase_count = int(np.sum(labels == 0))

    purchase_rate = (
        purchase_count / len(labels)
        if len(labels) > 0
        else 0.0
    )

    print(f"Purchase             : {purchase_count}")
    print(f"No Purchase          : {no_purchase_count}")
    print(f"Purchase rate        : {purchase_rate:.2%}")

    return (
        sequences,
        time_deltas,
        labels,
        session_ids,
    )


# ============================================================
# AGGREGATED FEATURE EXTRACTION
# ============================================================

def extract_aggregated_features(
    sequences,
    time_deltas,
):
    """
    Construct the same 14 aggregated features used by
    XGBoost and LightGBM.

    The feature order must remain identical to the order
    used during model training.
    """

    sequences = np.asarray(sequences)
    time_deltas = np.asarray(time_deltas)

    n_samples = sequences.shape[0]

    features = np.zeros(
        (n_samples, len(FEATURE_NAMES)),
        dtype=np.float32
    )

    for i in range(n_samples):

        seq = sequences[i]
        delta = time_deltas[i]

        # ----------------------------------------------------
        # Valid events
        # ----------------------------------------------------

        valid_mask = seq != 0
        valid_items = seq[valid_mask]

        if valid_items.size == 0:
            valid_items = np.array(
                [0],
                dtype=np.float32
            )

        valid_delta = delta[valid_mask]

        # ----------------------------------------------------
        # Item statistics
        # ----------------------------------------------------

        interaction_count = len(valid_items)

        unique_item_count = len(
            np.unique(valid_items)
        )

        repeated_item_count = (
            interaction_count
            - unique_item_count
        )

        first_item_id = float(
            valid_items[0]
        )

        last_item_id = float(
            valid_items[-1]
        )

        mean_item_id = float(
            np.mean(valid_items)
        )

        std_item_id = float(
            np.std(valid_items)
        )

        min_item_id = float(
            np.min(valid_items)
        )

        max_item_id = float(
            np.max(valid_items)
        )

        # ----------------------------------------------------
        # Time statistics
        # ----------------------------------------------------

        if valid_delta.size == 0:

            total_time_delta = 0.0
            mean_time_delta = 0.0
            max_time_delta = 0.0
            std_time_delta = 0.0
            nonzero_time_intervals = 0

        else:

            total_time_delta = float(
                np.sum(valid_delta)
            )

            mean_time_delta = float(
                np.mean(valid_delta)
            )

            max_time_delta = float(
                np.max(valid_delta)
            )

            std_time_delta = float(
                np.std(valid_delta)
            )

            nonzero_time_intervals = int(
                np.sum(valid_delta > 0)
            )

        # ----------------------------------------------------
        # Store features
        # ----------------------------------------------------

        features[i] = [
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
            nonzero_time_intervals,
        ]

    return features


# ============================================================
# TENSORFLOW INFERENCE FUNCTION
# ============================================================

def create_tf_inference_function(model):
    """
    Create one reusable TensorFlow inference function.

    The model is not retrained or modified.
    """

    @tf.function(
        reduce_retracing=True
    )
    def inference(inputs):

        return model(
            inputs,
            training=False
        )

    return inference


# ============================================================
# DEEP MODEL PREDICTION
# ============================================================

def predict_deep_model(
    inference_function,
    sequences,
):
    """
    Measure model-only TensorFlow inference latency.

    Input conversion is intentionally performed before
    starting the timer so that this measurement represents
    model inference only.
    """

    input_tensor = tf.convert_to_tensor(
        sequences,
        dtype=tf.int32
    )

    start = time.perf_counter()

    probabilities = inference_function(
        input_tensor
    )

    # .numpy() forces completion of the TensorFlow operation
    probabilities = probabilities.numpy()

    elapsed = (
        time.perf_counter() - start
    )

    probabilities = np.asarray(
        probabilities
    ).reshape(-1)

    return probabilities, elapsed


# ============================================================
# XGBOOST PREDICTION
# ============================================================

def predict_xgboost(
    model,
    features,
):
    """
    Measure XGBoost model-only inference latency.
    """

    start = time.perf_counter()

    probabilities = model.predict_proba(
        features
    )[:, 1]

    elapsed = (
        time.perf_counter() - start
    )

    return (
        np.asarray(
            probabilities
        ).reshape(-1),
        elapsed,
    )


# ============================================================
# LIGHTGBM PREDICTION
# ============================================================

def predict_lightgbm(
    model,
    features,
):
    """
    Measure LightGBM model-only inference latency.
    """

    start = time.perf_counter()

    probabilities = model.predict(
        features
    )

    elapsed = (
        time.perf_counter() - start
    )

    return (
        np.asarray(
            probabilities
        ).reshape(-1),
        elapsed,
    )


# ============================================================
# LOAD DATA
# ============================================================

(
    test_sequences,
    test_time_deltas,
    test_labels,
    test_session_ids,
) = load_dataset()


# ============================================================
# SELECT SIMULATION SESSIONS
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

available_sessions = len(
    test_sequences
)

simulation_count = min(
    SIMULATION_SESSIONS,
    available_sessions
)

if simulation_count == 0:
    raise ValueError(
        "No sessions are available for simulation."
    )

selected_indices = rng.choice(
    available_sessions,
    size=simulation_count,
    replace=False,
)

simulation_sequences = (
    test_sequences[selected_indices]
)

simulation_time_deltas = (
    test_time_deltas[selected_indices]
)

simulation_labels = (
    test_labels[selected_indices]
)

simulation_session_ids = (
    test_session_ids[selected_indices]
)

print("\nSimulation configuration:")
print(
    f"Sessions simulated    : "
    f"{simulation_count}"
)
print(
    f"Event stages          : "
    f"{EVENT_STAGES}"
)
print(
    f"Total observations    : "
    f"{simulation_count * EVENT_STAGES}"
)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

print("\nLoading trained models...")


# ------------------------------------------------------------
# LSTM
# ------------------------------------------------------------

if not os.path.exists(LSTM_MODEL_PATH):
    raise FileNotFoundError(
        f"LSTM model not found:\n{LSTM_MODEL_PATH}"
    )

print("Loading LSTM...")

lstm_model = tf.keras.models.load_model(
    LSTM_MODEL_PATH
)


# ------------------------------------------------------------
# GRU
# ------------------------------------------------------------

if not os.path.exists(GRU_MODEL_PATH):
    raise FileNotFoundError(
        f"GRU model not found:\n{GRU_MODEL_PATH}"
    )

print("Loading GRU...")

gru_model = tf.keras.models.load_model(
    GRU_MODEL_PATH
)


# ------------------------------------------------------------
# GRU + ATTENTION
# ------------------------------------------------------------

if not os.path.exists(
    GRU_ATTENTION_MODEL_PATH
):
    raise FileNotFoundError(
        "GRU + Attention model not found:\n"
        + GRU_ATTENTION_MODEL_PATH
    )

print("Loading GRU + Attention...")

gru_attention_model = (
    tf.keras.models.load_model(
        GRU_ATTENTION_MODEL_PATH
    )
)


# ------------------------------------------------------------
# XGBOOST
# ------------------------------------------------------------

if not os.path.exists(
    XGBOOST_MODEL_PATH
):
    raise FileNotFoundError(
        f"XGBoost model not found:\n"
        f"{XGBOOST_MODEL_PATH}"
    )

print("Loading XGBoost...")

xgb_model = xgb.XGBClassifier()

xgb_model.load_model(
    XGBOOST_MODEL_PATH
)


# ------------------------------------------------------------
# LIGHTGBM
# ------------------------------------------------------------

if not os.path.exists(
    LIGHTGBM_MODEL_PATH
):
    raise FileNotFoundError(
        f"LightGBM model not found:\n"
        f"{LIGHTGBM_MODEL_PATH}"
    )

print("Loading LightGBM...")

lgb_model = lgb.Booster(
    model_file=LIGHTGBM_MODEL_PATH
)


print(
    "\nAll trained models loaded successfully."
)


# ============================================================
# CREATE REUSABLE TENSORFLOW FUNCTIONS
# ============================================================

print(
    "\nCreating reusable TensorFlow "
    "inference functions..."
)

lstm_inference = (
    create_tf_inference_function(
        lstm_model
    )
)

gru_inference = (
    create_tf_inference_function(
        gru_model
    )
)

gru_attention_inference = (
    create_tf_inference_function(
        gru_attention_model
    )
)

print(
    "TensorFlow inference functions "
    "created successfully."
)


# ============================================================
# MODEL WARM-UP
# ============================================================

print("\n" + "=" * 70)
print("MODEL WARM-UP")
print("=" * 70)

warmup_count = min(
    WARMUP_SESSIONS,
    simulation_count
)

warmup_sequences = (
    simulation_sequences[:warmup_count]
)

warmup_time_deltas = (
    simulation_time_deltas[:warmup_count]
)

warmup_tensor = tf.convert_to_tensor(
    warmup_sequences,
    dtype=tf.int32
)

print(
    f"Warming up deep models using "
    f"{warmup_count} sessions..."
)

# TensorFlow graph tracing + initial inference
_ = lstm_inference(
    warmup_tensor
).numpy()

_ = gru_inference(
    warmup_tensor
).numpy()

_ = gru_attention_inference(
    warmup_tensor
).numpy()


print("Warming up tree models...")

warmup_features = (
    extract_aggregated_features(
        warmup_sequences,
        warmup_time_deltas
    )
)

_ = xgb_model.predict_proba(
    warmup_features
)[:, 1]

_ = lgb_model.predict(
    warmup_features
)

print(
    "Warm-up completed successfully."
)


# ============================================================
# LATENCY STORAGE
# ============================================================

event_latency_records = []

session_prediction_records = []

probability_trajectories = {
    "LSTM": [],
    "GRU": [],
    "GRU + Attention": [],
    "XGBoost": [],
    "LightGBM": [],
}


# ============================================================
# START REAL-TIME EVENT SIMULATION
# ============================================================

print("\n" + "=" * 70)
print("STARTING REFINED REAL-TIME EVENT SIMULATION")
print("=" * 70)


for event_stage in range(
    1,
    EVENT_STAGES + 1
):

    print(
        f"\nEvent "
        f"{event_stage:02d}/{EVENT_STAGES}"
    )

    # ========================================================
    # PREFIX / INPUT PREPARATION
    # ========================================================
    #
    # Only the interactions observed up to the current
    # event stage are exposed to the models.
    #
    # Remaining positions are zero-padded.
    #
    # This preparation time is measured separately.
    # ========================================================

    prefix_start = time.perf_counter()

    prefix_sequences = np.zeros_like(
        simulation_sequences
    )

    prefix_sequences[
        :,
        :event_stage
    ] = simulation_sequences[
        :,
        :event_stage
    ]

    prefix_time_deltas = np.zeros_like(
        simulation_time_deltas
    )

    prefix_time_deltas[
        :,
        :event_stage
    ] = simulation_time_deltas[
        :,
        :event_stage
    ]

    prefix_elapsed = (
        time.perf_counter()
        - prefix_start
    )

    prefix_ms_per_session = (
        prefix_elapsed
        * 1000
        / simulation_count
    )


    # ========================================================
    # DEEP LEARNING MODEL INFERENCE
    # ========================================================

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    lstm_probabilities, (
        lstm_model_time
    ) = predict_deep_model(
        lstm_inference,
        prefix_sequences
    )


    # --------------------------------------------------------
    # GRU
    # --------------------------------------------------------

    gru_probabilities, (
        gru_model_time
    ) = predict_deep_model(
        gru_inference,
        prefix_sequences
    )


    # --------------------------------------------------------
    # GRU + ATTENTION
    # --------------------------------------------------------

    gru_attention_probabilities, (
        gru_attention_model_time
    ) = predict_deep_model(
        gru_attention_inference,
        prefix_sequences
    )


    # ========================================================
    # TREE-MODEL FEATURE CONSTRUCTION
    # ========================================================

    feature_start = time.perf_counter()

    prefix_features = (
        extract_aggregated_features(
            prefix_sequences,
            prefix_time_deltas
        )
    )

    feature_elapsed = (
        time.perf_counter()
        - feature_start
    )

    feature_ms_per_session = (
        feature_elapsed
        * 1000
        / simulation_count
    )


    # ========================================================
    # XGBOOST
    # ========================================================

    xgb_probabilities, (
        xgb_model_time
    ) = predict_xgboost(
        xgb_model,
        prefix_features
    )


    # ========================================================
    # LIGHTGBM
    # ========================================================

    lgb_probabilities, (
        lgb_model_time
    ) = predict_lightgbm(
        lgb_model,
        prefix_features
    )


    # ========================================================
    # MODEL-ONLY LATENCY
    # ========================================================

    lstm_ms_per_session = (
        lstm_model_time
        * 1000
        / simulation_count
    )

    gru_ms_per_session = (
        gru_model_time
        * 1000
        / simulation_count
    )

    gru_attention_ms_per_session = (
        gru_attention_model_time
        * 1000
        / simulation_count
    )

    xgb_ms_per_session = (
        xgb_model_time
        * 1000
        / simulation_count
    )

    lgb_ms_per_session = (
        lgb_model_time
        * 1000
        / simulation_count
    )


    # ========================================================
    # END-TO-END LATENCY
    # ========================================================
    #
    # Deep-learning:
    # prefix/input preparation + model inference
    #
    # Tree models:
    # prefix preparation + feature construction +
    # model inference
    #
    # These measurements are reported separately from
    # model-only inference.
    # ========================================================

    lstm_end_to_end = (
        prefix_ms_per_session
        + lstm_ms_per_session
    )

    gru_end_to_end = (
        prefix_ms_per_session
        + gru_ms_per_session
    )

    gru_attention_end_to_end = (
        prefix_ms_per_session
        + gru_attention_ms_per_session
    )

    xgb_end_to_end = (
        prefix_ms_per_session
        + feature_ms_per_session
        + xgb_ms_per_session
    )

    lgb_end_to_end = (
        prefix_ms_per_session
        + feature_ms_per_session
        + lgb_ms_per_session
    )


    # ========================================================
    # STORE LATENCY RESULTS
    # ========================================================

    stage_records = [

        {
            "model": "LSTM",
            "event_stage": event_stage,
            "sessions": simulation_count,

            "input_preparation_ms_per_session":
                prefix_ms_per_session,

            "feature_construction_ms_per_session":
                0.0,

            "model_only_ms_per_session":
                lstm_ms_per_session,

            "end_to_end_ms_per_session":
                lstm_end_to_end,
        },

        {
            "model": "GRU",
            "event_stage": event_stage,
            "sessions": simulation_count,

            "input_preparation_ms_per_session":
                prefix_ms_per_session,

            "feature_construction_ms_per_session":
                0.0,

            "model_only_ms_per_session":
                gru_ms_per_session,

            "end_to_end_ms_per_session":
                gru_end_to_end,
        },

        {
            "model": "GRU + Attention",
            "event_stage": event_stage,
            "sessions": simulation_count,

            "input_preparation_ms_per_session":
                prefix_ms_per_session,

            "feature_construction_ms_per_session":
                0.0,

            "model_only_ms_per_session":
                gru_attention_ms_per_session,

            "end_to_end_ms_per_session":
                gru_attention_end_to_end,
        },

        {
            "model": "XGBoost",
            "event_stage": event_stage,
            "sessions": simulation_count,

            "input_preparation_ms_per_session":
                prefix_ms_per_session,

            "feature_construction_ms_per_session":
                feature_ms_per_session,

            "model_only_ms_per_session":
                xgb_ms_per_session,

            "end_to_end_ms_per_session":
                xgb_end_to_end,
        },

        {
            "model": "LightGBM",
            "event_stage": event_stage,
            "sessions": simulation_count,

            "input_preparation_ms_per_session":
                prefix_ms_per_session,

            "feature_construction_ms_per_session":
                feature_ms_per_session,

            "model_only_ms_per_session":
                lgb_ms_per_session,

            "end_to_end_ms_per_session":
                lgb_end_to_end,
        },
    ]

    event_latency_records.extend(
        stage_records
    )


    # ========================================================
    # PROBABILITY TRAJECTORIES
    # ========================================================

    probability_trajectories[
        "LSTM"
    ].append(
        float(
            np.mean(
                lstm_probabilities
            )
        )
    )

    probability_trajectories[
        "GRU"
    ].append(
        float(
            np.mean(
                gru_probabilities
            )
        )
    )

    probability_trajectories[
        "GRU + Attention"
    ].append(
        float(
            np.mean(
                gru_attention_probabilities
            )
        )
    )

    probability_trajectories[
        "XGBoost"
    ].append(
        float(
            np.mean(
                xgb_probabilities
            )
        )
    )

    probability_trajectories[
        "LightGBM"
    ].append(
        float(
            np.mean(
                lgb_probabilities
            )
        )
    )


    # ========================================================
    # FINAL-STAGE SESSION PREDICTIONS
    # ========================================================

    if event_stage == EVENT_STAGES:

        for i in range(
            simulation_count
        ):

            session_prediction_records.append(
                {
                    "session_id":
                        simulation_session_ids[i],

                    "actual_label":
                        int(
                            simulation_labels[i]
                        ),

                    "LSTM_probability":
                        float(
                            lstm_probabilities[i]
                        ),

                    "GRU_probability":
                        float(
                            gru_probabilities[i]
                        ),

                    "GRU_Attention_probability":
                        float(
                            gru_attention_probabilities[i]
                        ),

                    "XGBoost_probability":
                        float(
                            xgb_probabilities[i]
                        ),

                    "LightGBM_probability":
                        float(
                            lgb_probabilities[i]
                        ),
                }
            )


# ============================================================
# EVENT LATENCY DATAFRAME
# ============================================================

event_latency_df = pd.DataFrame(
    event_latency_records
)

event_latency_path = os.path.join(
    METRICS_DIR,
    "refined_realtime_event_latency.csv"
)

event_latency_df.to_csv(
    event_latency_path,
    index=False
)


# ============================================================
# LATENCY SUMMARY
# ============================================================

summary_records = []

models = [
    "LSTM",
    "GRU",
    "GRU + Attention",
    "XGBoost",
    "LightGBM",
]


for model_name in models:

    model_df = event_latency_df[
        event_latency_df["model"]
        == model_name
    ]

    model_latencies = (
        model_df[
            "model_only_ms_per_session"
        ].to_numpy()
    )

    input_latencies = (
        model_df[
            "input_preparation_ms_per_session"
        ].to_numpy()
    )

    feature_latencies = (
        model_df[
            "feature_construction_ms_per_session"
        ].to_numpy()
    )

    end_to_end_latencies = (
        model_df[
            "end_to_end_ms_per_session"
        ].to_numpy()
    )


    summary_records.append(
        {
            "model":
                model_name,

            "number_of_stage_measurements":
                len(model_latencies),

            "simulation_sessions_per_stage":
                simulation_count,

            "event_stages":
                EVENT_STAGES,

            "total_session_stage_observations":
                simulation_count
                * EVENT_STAGES,

            # ------------------------------------------------
            # Model-only
            # ------------------------------------------------

            "model_only_mean_ms_per_session":
                float(
                    np.mean(
                        model_latencies
                    )
                ),

            "model_only_median_ms_per_session":
                float(
                    np.median(
                        model_latencies
                    )
                ),

            "model_only_p95_ms_per_session":
                float(
                    np.percentile(
                        model_latencies,
                        95
                    )
                ),

            "model_only_min_ms_per_session":
                float(
                    np.min(
                        model_latencies
                    )
                ),

            "model_only_max_ms_per_session":
                float(
                    np.max(
                        model_latencies
                    )
                ),

            # ------------------------------------------------
            # Input preparation
            # ------------------------------------------------

            "input_preparation_mean_ms_per_session":
                float(
                    np.mean(
                        input_latencies
                    )
                ),

            "input_preparation_median_ms_per_session":
                float(
                    np.median(
                        input_latencies
                    )
                ),

            "input_preparation_p95_ms_per_session":
                float(
                    np.percentile(
                        input_latencies,
                        95
                    )
                ),

            # ------------------------------------------------
            # Feature construction
            # ------------------------------------------------

            "feature_construction_mean_ms_per_session":
                float(
                    np.mean(
                        feature_latencies
                    )
                ),

            "feature_construction_median_ms_per_session":
                float(
                    np.median(
                        feature_latencies
                    )
                ),

            "feature_construction_p95_ms_per_session":
                float(
                    np.percentile(
                        feature_latencies,
                        95
                    )
                ),

            # ------------------------------------------------
            # End-to-end
            # ------------------------------------------------

            "end_to_end_mean_ms_per_session":
                float(
                    np.mean(
                        end_to_end_latencies
                    )
                ),

            "end_to_end_median_ms_per_session":
                float(
                    np.median(
                        end_to_end_latencies
                    )
                ),

            "end_to_end_p95_ms_per_session":
                float(
                    np.percentile(
                        end_to_end_latencies,
                        95
                    )
                ),

            "end_to_end_min_ms_per_session":
                float(
                    np.min(
                        end_to_end_latencies
                    )
                ),

            "end_to_end_max_ms_per_session":
                float(
                    np.max(
                        end_to_end_latencies
                    )
                ),

            # ------------------------------------------------
            # Interpretation
            # ------------------------------------------------

            "latency_unit":
                "milliseconds per simulated session",

            "p95_interpretation":
                (
                    "P95 is calculated across the "
                    "20 event-stage batch-normalized "
                    "latency measurements. It should "
                    "not be interpreted as P95 of "
                    "20,000 independent single-request "
                    "latencies."
                ),
        }
    )


latency_summary_df = pd.DataFrame(
    summary_records
)

latency_summary_path = os.path.join(
    METRICS_DIR,
    "refined_realtime_latency_metrics.csv"
)

latency_summary_df.to_csv(
    latency_summary_path,
    index=False
)


# ============================================================
# SESSION PREDICTIONS
# ============================================================

session_predictions_df = pd.DataFrame(
    session_prediction_records
)

session_predictions_path = os.path.join(
    METRICS_DIR,
    "refined_realtime_session_predictions.csv"
)

session_predictions_df.to_csv(
    session_predictions_path,
    index=False
)


# ============================================================
# PROBABILITY TRAJECTORY FIGURE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

event_numbers = np.arange(
    1,
    EVENT_STAGES + 1
)

for model_name in models:

    plt.plot(
        event_numbers,
        probability_trajectories[
            model_name
        ],
        marker="o",
        label=model_name
    )

plt.xlabel(
    "Event Stage"
)

plt.ylabel(
    "Mean Predicted Purchase Probability"
)

plt.title(
    "Real-Time Purchase Probability Trajectory"
)

plt.xticks(
    event_numbers
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

probability_figure_path = os.path.join(
    FIGURES_DIR,
    "refined_realtime_probability_trajectory.png"
)

plt.savefig(
    probability_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# LATENCY COMPARISON FIGURE
# ============================================================

plot_df = latency_summary_df.copy()

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    plot_df["model"],
    plot_df[
        "model_only_mean_ms_per_session"
    ]
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Mean Model-Only Latency (ms/session)"
)

plt.title(
    "Real-Time Model Inference Latency Comparison"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

latency_figure_path = os.path.join(
    FIGURES_DIR,
    "refined_realtime_latency_comparison.png"
)

plt.savefig(
    latency_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# END-TO-END LATENCY FIGURE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    plot_df["model"],
    plot_df[
        "end_to_end_mean_ms_per_session"
    ]
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Mean End-to-End Latency (ms/session)"
)

plt.title(
    "Real-Time End-to-End Latency Comparison"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

end_to_end_figure_path = os.path.join(
    FIGURES_DIR,
    "refined_realtime_end_to_end_latency.png"
)

plt.savefig(
    end_to_end_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REFINED REAL-TIME LATENCY SUMMARY")
print("=" * 70)

for _, row in latency_summary_df.iterrows():

    print(
        f"\n{row['model']}"
    )

    print(
        "  Model-only mean   : "
        f"{row['model_only_mean_ms_per_session']:.4f} "
        "ms/session"
    )

    print(
        "  Model-only median : "
        f"{row['model_only_median_ms_per_session']:.4f} "
        "ms/session"
    )

    print(
        "  Model-only p95    : "
        f"{row['model_only_p95_ms_per_session']:.4f} "
        "ms/session"
    )

    print(
        "  End-to-end mean   : "
        f"{row['end_to_end_mean_ms_per_session']:.4f} "
        "ms/session"
    )

    print(
        "  End-to-end median : "
        f"{row['end_to_end_median_ms_per_session']:.4f} "
        "ms/session"
    )

    print(
        "  End-to-end p95    : "
        f"{row['end_to_end_p95_ms_per_session']:.4f} "
        "ms/session"
    )


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    f"\n{event_latency_path}"
)

print(
    latency_summary_path
)

print(
    session_predictions_path
)

print(
    probability_figure_path
)

print(
    latency_figure_path
)

print(
    end_to_end_figure_path
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print(
    "REFINED REAL-TIME SIMULATION "
    "COMPLETED SUCCESSFULLY"
)
print("=" * 70)

print(
    "\nExisting trained models were used."
)

print(
    "No model retraining was performed."
)

print(
    "TensorFlow inference functions were "
    "created once and reused across event stages."
)

print(
    "TensorFlow reduce_retracing=True "
    "was enabled."
)

print(
    "Warm-up was performed before latency "
    "measurements."
)

print(
    "Model-only and end-to-end latency "
    "were measured separately."
)

print(
    "\nNote: latency values are "
    "batch-normalized milliseconds per "
    "simulated session."
)