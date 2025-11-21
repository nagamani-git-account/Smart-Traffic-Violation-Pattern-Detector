import pandas as pd
df = pd.read_parquet("detected_violations.parquet")
print(df.shape)
print(df.head())