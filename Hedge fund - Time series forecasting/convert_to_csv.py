import pandas as pd
df = pd.read_parquet('data/test.parquet')
df.to_csv('data/test.csv')

df = pd.read_parquet('data/train.parquet')
df.to_csv('data/train.csv')

