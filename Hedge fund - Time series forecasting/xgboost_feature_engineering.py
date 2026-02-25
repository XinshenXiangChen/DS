import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder


# --- 1. FEATURE ENGINEERING ---

def engineer_features(df):
    df = df.copy()
    # Ensure chronological and group order
    df = df.sort_values(['code', 'sub_code', 'sub_category', 'horizon', 'ts_index'])

    # We'll apply memory features to the first few columns as an example
    # You can expand this to all 86 features if the 6-hour limit allows
    key_features = [f'feature_{i}' for i in ['a', 'b', 'c', 'd']]
    grouped = df.groupby(['code', 'sub_code', 'sub_category', 'horizon'])

    for feat in key_features:
        if feat in df.columns:
            df[f'{feat}_lag1'] = grouped[feat].shift(1)
            df[f'{feat}_roll_mean5'] = grouped[feat].transform(lambda x: x.shift(1).rolling(5).mean())
            df[f'{feat}_roll_std5'] = grouped[feat].transform(lambda x: x.shift(1).rolling(5).std())

    # --- THE WEIGHTING FIX ---
    # Only calculate weights if the 'weight' column exists (Training mode)
    if 'weight' in df.columns:
        max_ts = df['ts_index'].max()
        # Time-decay: 1.0001 is a small decay factor. Adjust if process is very unstable.
        df['time_weight'] = df['weight'] * (1.0001 ** (df['ts_index'] - max_ts))
    else:
        # For the test set, we don't need weights
        df['time_weight'] = 1.0

    return df


def preprocess_categoricals(df, encoders=None, is_train=True):
    df = df.copy()
    cat_cols = ['code', 'sub_code', 'sub_category', 'horizon']
    if is_train:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in cat_cols:
            le = encoders[col]
            # Handle unseen codes with -1
            df[col] = df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    return df, encoders


# --- 2. EXECUTION ---

# Load
train_df = pd.read_parquet('data/train.parquet')
test_df = pd.read_parquet('data/test.parquet')

# Feature Engineering
train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

# Categoricals
train_df, encoders = preprocess_categoricals(train_df, is_train=True)
test_df, _ = preprocess_categoricals(test_df, encoders=encoders, is_train=False)

# --- FEATURE SELECTION FIX ---
# We must ensure the features used for training are EXACTLY what exists in test
target = 'y_target'
# Only use columns that exist in BOTH, excluding ID/Targets/Weights
features = [c for c in train_df.columns if
            c in test_df.columns and c not in ['id', 'y_target', 'weight', 'time_weight', 'ts_index']]

print(f"Training on {len(features)} features...")

# Time-Series Split
split_ts = train_df['ts_index'].quantile(0.85)
train_split = train_df[train_df['ts_index'] <= split_ts]
val_split = train_df[train_df['ts_index'] > split_ts]

# Model
model = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.015,
    max_depth=5,
    subsample=0.6,
    colsample_bytree=0.6,
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=100,
)

# Fit
model.fit(
    train_split[features], train_split[target],
    sample_weight=train_split['time_weight'],
    eval_set=[(val_split[features], val_split[target])],
    sample_weight_eval_set=[val_split['weight']],
    verbose=100
)

# --- 3. SUBMISSION ---

test_preds = model.predict(test_df[features])

submission = pd.DataFrame({
    'id': test_df['id'],
    'prediction': test_preds
})

# Save with comma (Check if platform needs sep=';')
submission.to_csv('submission.csv', index=False)
print("Submission saved successfully!")