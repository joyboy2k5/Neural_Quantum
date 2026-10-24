
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
file_path = ROOT / "results" / "cleaned_network_data.csv"

df = pd.read_csv(file_path, low_memory=False)

print("CLEANED DATA VERIFICATION")
print("-" * 35)
print("Shape:", df.shape)
print("Missing cells:", df.isna().sum().sum())

numeric_df = df.select_dtypes(include=np.number)
print("Infinite values:", np.isinf(numeric_df).sum().sum())

print("\nLabel counts:")
print(df["Label"].value_counts())

print("\nFirst 5 rows:")
print(df.head())
