import pandas as pd
import ast
import math

df = pd.read_csv("csv/cnn_hyperparameter_tuning.csv")

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

metrics = ["accuracy", "precision", "recall", "f1"]

best_rows = {}

for metric in metrics:
    best_idx = df[metric].idxmax()
    best_rows[metric] = df.loc[best_idx]


print("\n========================")
print("NAJLEPSZE KONFIGURACJE")
print("========================\n")

for metric, row in best_rows.items():
    print(f"Najlepsze dla {metric.upper()}:")

    row = row.drop(labels=['augment_clean'], errors='ignore')
    row = row.drop(labels=['augment'], errors='ignore')
    all_cols = row.index.tolist()
    hyperparams = [col for col in all_cols if col not in metrics]
    new_order = hyperparams + metrics
    row_sorted = row.reindex(new_order)

    print(row_sorted)
    print("\n")
