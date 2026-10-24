
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV dataset found in data/")

df = pd.read_csv(csv_files[0], low_memory=False)
df.columns = df.columns.str.strip()

print("DATA QUALITY REPORT")
print("-" * 40)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Duplicate rows: {df.duplicated().sum():,}")

print("\nMissing values:")
print(f"Total missing cells: {df.isna().sum().sum():,}")
print("Columns with missing values:")
print(df.isna().sum().loc[lambda x: x > 0].to_string())

# Check numeric columns for positive or negative infinity
numeric_df = df.select_dtypes(include=np.number)
infinite_counts = np.isinf(numeric_df).sum()
infinite_counts = infinite_counts[infinite_counts > 0]

print("\nColumns containing infinite values:")
print(infinite_counts.to_string() if not infinite_counts.empty else "None")

print("\nLabels:")
print(df["Label"].astype(str).str.strip().value_counts())
