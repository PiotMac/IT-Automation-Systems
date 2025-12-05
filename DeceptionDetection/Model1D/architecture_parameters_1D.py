import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("csv/cnn_architecture_tuning.csv")

df = df.replace("None", None)

architecture_params = [
    "conv_layers",
    "filters",
    "kernel_size",
    "dropout_rate",
    "dense_units",
    "pool_size"
]

metrics = ["accuracy", "precision", "recall", "f1"]

best_results = {}

for metric in metrics:
    idx = df[metric].idxmax()
    best_results[metric] = df.loc[idx]

print("\n========================")
print("NAJLEPSZE KONFIGURACJE")
print("========================\n")

for metric, row in best_results.items():
    print(f"Najlepsze dla {metric.upper()}:")
    print(row)
    print("\n")
