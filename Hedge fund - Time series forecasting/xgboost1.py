import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder


# --- 1. THE FUNCTIONS ---

def preprocess_data(df, encoders=None, is_train=True):
    """
    Handles categorical encoding and ensures 'id' is split for features.
    """
    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()

    # The 'id' contains the features: code__sub_code__sub_category__horizon__ts_index
    # However, these are already columns in your dataset description.
    # We focus on the categorical columns provided.
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
            # Handle unseen categories in test set by mapping to a 'new' class
            df[col] = df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    return df, encoders


# --- 2. THE FIT CODE ---

# Load data (Assuming .parquet based on description)
train_df = pd.read_parquet('train.parquet')

# Preprocess
train_df, encoders = preprocess_data(train_df, is_train=True)

# Define features (Drop the ones the rules forbid using as features)
features = [c for c in train_df.columns if c not in ['id', 'target', 'weight', 'ts_index']]
target = 'target'
weights = 'weight'

# Time-Series Split: Use the last 20% of ts_index for validation
# This mimics the "hidden" test set period.
split_point = train_df['ts_index'].quantile(0.8)
train_split = train_df[train_df['ts_index'] <= split_point]
val_split = train_df[train_df['ts_index'] > split_point]

# Initialize XGBoost with high regularization for low signal-to-noise
model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    n_jobs=-1,
    tree_method='hist',  # Fast for 6-hour limit
    random_state=42
)

model.fit(
    train_split[features],
    train_split[target],
    sample_weight=train_split[weights],
    eval_set=[(val_split[features], val_split[target])],
    sample_weight_eval_set=[val_split[weights]],
    early_stopping_rounds=100,
    verbose=100
)

# --- 3. THE PREDICTION CODE ---

test_df = pd.read_parquet('test.parquet')
test_processed, _ = preprocess_data(test_df, encoders=encoders, is_train=False)

# Generate predictions
test_predictions = model.predict(test_processed[features])

# Create the submission file in the exact requested format
submission = pd.DataFrame({
    'id': test_df['id'],
    'prediction': test_predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False, sep=';')
print("Submission file saved successfully!")