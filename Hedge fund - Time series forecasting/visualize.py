import pandas as pd

df = pd.read_parquet('data/train.parquet')

print(df.columns.tolist())