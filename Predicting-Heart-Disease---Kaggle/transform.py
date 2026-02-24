import pandas as pd

df = pd.read_csv('data/train.csv')

df['Heart Disease'] = df['Heart Disease'].map({'Absence': 0, 'Presence': 1})

print(df['Heart Disease'].value_counts())

df.to_csv('train_encoded.csv', index=False)