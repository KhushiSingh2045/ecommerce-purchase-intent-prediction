import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

metrics_path = Path("results/metrics/lstm_metrics.csv")
output_path = Path("results/figures/lstm_test_metrics.png")

df = pd.read_csv(metrics_path)
row = df.iloc[0]

metrics = {
    "Precision": row["precision"],
    "Recall": row["recall"],
    "F1-score": row["f1_score"],
    "ROC-AUC": row["roc_auc"],
}

plt.figure(figsize=(8, 5))
plt.bar(metrics.keys(), metrics.values())

plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("LSTM Test Performance")

for i, value in enumerate(metrics.values()):
    plt.text(i, value + 0.02, f"{value:.4f}", ha="center")

plt.tight_layout()

output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved -> {output_path}")