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

for param in architecture_params:
    plt.figure(figsize=(10, 6))

    grouped = df.groupby(param)[metrics].mean()

    for metric in metrics:
        plt.plot(grouped.index, grouped[metric], marker="o", label=metric)

    plt.title(f"Wpływ parametru: {param} na metryki")
    plt.xlabel(param)
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot_for_{param}")


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
