"""
Builds the sequential-feature representation (Path B) from the raw
YOOCHOOSE clickstream files - vectorized version. Each session becomes
an ordered, padded sequence of item clicks, labeled 1 if that session
appears in yoochoose-buys.dat (purchased) or 0 if not.

This replaces an earlier, correctness-verified but much slower
per-session Python loop with vectorized pandas/numpy operations that
produce mathematically identical output, confirmed by direct comparison.
"""
import os
import numpy as np
import pandas as pd

CLICKS_PATH = "data/raw/yoochoose/yoochoose-clicks.dat"
BUYS_PATH = "data/raw/yoochoose/yoochoose-buys.dat"
OUTPUT_PATH = "data/processed/yoochoose_sequences.npz"

MAX_SEQ_LEN = 20
# None = full ~33M-row file. Use a smaller number only for a quick test.
SAMPLE_ROWS = None


def load_clicks(path: str, sample_rows: int = None) -> pd.DataFrame:
    clicks = pd.read_csv(
        path, header=None,
        names=["session_id", "timestamp", "item_id", "category"],
        nrows=sample_rows,
    )
    clicks["category"] = clicks["category"].replace("S", "-1").astype(int)
    clicks["timestamp"] = pd.to_datetime(clicks["timestamp"])
    return clicks


def load_buy_session_ids(path: str) -> set:
    buys = pd.read_csv(
        path, header=None,
        names=["session_id", "timestamp", "item_id", "price", "quantity"],
    )
    return set(buys["session_id"].unique())


def build_labeled_sequences_vectorized(clicks: pd.DataFrame, buy_session_ids: set, max_seq_len: int = MAX_SEQ_LEN):
    clicks = clicks.sort_values(["session_id", "timestamp"]).copy()

    vocab = {item: idx + 1 for idx, item in enumerate(sorted(clicks["item_id"].unique()))}
    clicks["item_code"] = clicks["item_id"].map(vocab)

    session_codes, session_uniques = pd.factorize(clicks["session_id"], sort=True)
    clicks["session_code"] = session_codes
    clicks["position"] = clicks.groupby("session_code").cumcount()

    clicks["timestamp_unix"] = clicks["timestamp"].astype("int64") // 10**9
    clicks["time_delta"] = clicks.groupby("session_code")["timestamp_unix"].diff().fillna(0)

    clicks = clicks[clicks["position"] < max_seq_len]

    n_sessions = len(session_uniques)
    sequences = np.zeros((n_sessions, max_seq_len), dtype=np.int32)
    time_deltas = np.zeros((n_sessions, max_seq_len), dtype=np.float32)

    sequences[clicks["session_code"].values, clicks["position"].values] = clicks["item_code"].values
    time_deltas[clicks["session_code"].values, clicks["position"].values] = clicks["time_delta"].values

    labels = np.isin(session_uniques, list(buy_session_ids)).astype(np.int8)

    return sequences, time_deltas, labels, session_uniques, vocab


if __name__ == "__main__":
    clicks = load_clicks(CLICKS_PATH, sample_rows=SAMPLE_ROWS)
    buy_session_ids = load_buy_session_ids(BUYS_PATH)

    print(f"Loaded {len(clicks)} clicks across {clicks['session_id'].nunique()} sessions.")
    print(f"{len(buy_session_ids)} sessions had a purchase (in the full buys file).")

    sequences, time_deltas, labels, session_ids, vocab = build_labeled_sequences_vectorized(clicks, buy_session_ids)

    print(f"\nBuilt {len(sequences)} labeled sequences, shape {sequences.shape}")
    print(f"Vocabulary size (unique items): {len(vocab)}")
    print(f"Positive (purchase) sessions in this data: {labels.sum()} / {len(labels)} ({100*labels.mean():.1f}%)")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        sequences=sequences, time_deltas=time_deltas, labels=labels,
        session_ids=session_ids,
    )
    print(f"\nSaved -> {OUTPUT_PATH}")