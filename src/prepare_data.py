
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

csv_files = list(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV found in data/")

df = pd.read_csv(csv_files[0], low_memory=False)
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].astype(str).str.strip()

print("Original rows:", len(df))

# Remove exact duplicate records.
df = df.drop_duplicates().copy()
print("After removing duplicates:", len(df))

# Convert infinite numeric values to missing values.
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].replace(
    [np.inf, -np.inf], np.nan
)

# Replace missing numeric feature values with their column median.
feature_cols = [c for c in df.columns if c != "Label"]
for col in feature_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())

print("Remaining missing cells:", df.isna().sum().sum())
print("\nLabels after cleaning:")
print(df["Label"].value_counts())

RESULTS_DIR.mkdir(exist_ok=True)
output_path = RESULTS_DIR / "cleaned_network_data.csv"
df.to_csv(output_path, index=False)

print(f"\nSaved cleaned dataset to: {output_path}")
