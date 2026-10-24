
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "results" / "cleaned_network_data.csv"
OUTPUT_PATH = ROOT / "results" / "quantum_experiment_data.csv"

FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
]

df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip()

print("Rows before filtering:", len(df))

# Remove records with physically invalid negative flow duration.
invalid = df["Flow Duration"] < 0
print("Negative-duration records removed:", int(invalid.sum()))

df = df.loc[~invalid, FEATURES + ["Label"]].copy()

print("\nFinal dataset shape:", df.shape)
print("\nLabel counts:")
print(df["Label"].value_counts())

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved to: {OUTPUT_PATH}")
