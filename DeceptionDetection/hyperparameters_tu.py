import pandas as pd
import matplotlib.pyplot as plt
import ast
import math
from pandas.api.types import CategoricalDtype

df = pd.read_csv("cnn_hyperparameter_tuning.csv")

def clean_augment(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
        return value[1]

    if isinstance(value, dict):
        return value

    return None


def parse_aug(val):
    try:
        return ast.literal_eval(val)
    except:
        return val


def categorize_augmentation(aug_dict):
    if not isinstance(aug_dict, dict):
        return "none"

    pitch = aug_dict.get("pitch", None)

    if pitch == (-1, 1):
        return "low"
    elif pitch == (-2, 2):
        return "medium"
    elif pitch == (-3, 3):
        return "large"
    else:
        return "none"

df["augment"] = df["augment"].apply(parse_aug)
df["augment_clean"] = df["augment"].apply(clean_augment)
df["augment_category"] = df["augment_clean"].apply(categorize_augmentation)

# aug_order = CategoricalDtype(
#     ["none", "low", "medium", "large"],
#     ordered=True
# )
#
# df["augment_category"] = df["augment_category"].astype(aug_order)
#
# def plot_numeric_param(df, param_name):
#     metrics = ["accuracy", "precision", "recall", "f1"]
#     grouped = df.groupby(param_name)[metrics].mean()
#
#     plt.figure()
#     for metric in metrics:
#         plt.plot(grouped.index, grouped[metric], label=metric)
#         plt.xlabel(param_name)
#         plt.ylabel(metric)
#         plt.title(f"Metrics for {param_name}")
#         plt.grid(True)
#         plt.tight_layout()
#     plt.legend()
#     plt.savefig(f"plot_for_{param_name}.png", dpi=150)
#     plt.close()
#
#
# def plot_categorical_param(df, param_name):
#     metrics = ["accuracy", "precision", "recall", "f1"]
#
#     grouped = df.groupby(param_name)[metrics].mean()
#
#     plt.figure()
#     for metric in metrics:
#         plt.plot(grouped.index, grouped[metric], label=metric)
#         plt.xticks(rotation=45, ha="right")
#         plt.xlabel(param_name)
#         plt.ylabel(metric)
#         plt.title(f"Metrics for {param_name}")
#         plt.grid(True, axis="y")
#         plt.tight_layout()
#     plt.savefig(f"plot_for_{param_name}.png", dpi=150)
#     plt.close()
#
#
# df_none_aug = df[
#     df["augment_clean"].isna() &
#     (df["N_MFCC"] == 20) &
#     (df["step"] == 1.0) &
#     (df["duration"] == 2.0)
# ]
#
# plot_numeric_param(df_none_aug, "N_MFCC")
# plot_numeric_param(df_none_aug, "duration")
# plot_numeric_param(df_none_aug, "step")
# plot_numeric_param(df_none_aug, "batch_size")
#
# # plot_categorical_param(df, "augment_category")
#
#
# print("Wszystkie wykresy zostały wygenerowane!")

metrics = ["accuracy", "precision", "recall", "f1"]

best_rows = {}

for metric in metrics:
    best_idx = df[metric].idxmax()
    best_rows[metric] = df.loc[best_idx]

# Wypisanie wyników
print("\n================= BEST RESULTS =================")
for metric, row in best_rows.items():
    print(f"\n🏆 Best {metric.upper()}: {row[metric]:.4f}")
    print("Parameters:")
    print(f"  N_MFCC:      {row['N_MFCC']}")
    print(f"  duration:    {row['duration']}")
    print(f"  step:        {row['step']}")
    print(f"  batch_size:  {row['batch_size']}")
    print(f"  augment:     {row['augment_category']}  ({row['augment_clean']})")
