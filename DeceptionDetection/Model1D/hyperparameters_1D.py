import pandas as pd
import matplotlib.pyplot as plt
import ast
import math
from pandas.api.types import CategoricalDtype

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

# Wypisanie wyników
print("\n================= BEST RESULTS =================")
for metric, row in best_rows.items():
    print(f"\nBest {metric.upper()}: {row[metric]:.4f}")
    print("Parameters:")
    print(f"  N_MFCC:      {row['N_MFCC']}")
    print(f"  duration:    {row['duration']}")
    print(f"  step:        {row['step']}")
    print(f"  batch_size:  {row['batch_size']}")
    print(f"  augment:     {row['augment_category']}  ({row['augment_clean']})")
