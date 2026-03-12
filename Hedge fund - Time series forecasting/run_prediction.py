"""
Time Series Forecasting Approach
Uses statistical time series methods (exponential smoothing, moving averages)
instead of tree-based models
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def exponential_smoothing(series, alpha=0.3):
    """
    Simple Exponential Smoothing
    
    Exponential smoothing is a time series forecasting method that gives more weight
    to recent observations while still considering older data. It's like a weighted
    average where recent values matter more than distant ones.
    
    How it works:
    - The alpha parameter (between 0 and 1) controls how much weight recent data gets
    - Alpha = 0.3 means: 30% weight to the new observation, 70% to the previous smoothed value
    - Higher alpha (e.g., 0.5) = more responsive to recent changes (more reactive)
    - Lower alpha (e.g., 0.1) = smoother, less reactive to recent changes (more stable)
    
    Formula: 
    Smoothed[t] = alpha * Actual[t] + (1 - alpha) * Smoothed[t-1]
    
    Example with alpha=0.3:
    If you have values [10, 12, 11, 13, 14]:
    - Start: Smoothed[0] = 10
    - Step 1: Smoothed[1] = 0.3*12 + 0.7*10 = 3.6 + 7 = 10.6
    - Step 2: Smoothed[2] = 0.3*11 + 0.7*10.6 = 3.3 + 7.42 = 10.72
    - Step 3: Smoothed[3] = 0.3*13 + 0.7*10.72 = 3.9 + 7.5 = 11.4
    - Step 4: Smoothed[4] = 0.3*14 + 0.7*11.4 = 4.2 + 7.98 = 12.18
    
    The prediction for the next value would be 12.18 (the last smoothed value).
    
    Why it's useful:
    - Handles noise in data by smoothing it out
    - Adapts to trends automatically
    - Simple and fast to compute
    - Works well for data with no clear seasonal patterns
    """
    result = [series[0]]
    for i in range(1, len(series)):
        result.append(alpha * series[i] + (1 - alpha) * result[i-1])
    return np.array(result)


def predict_with_exponential_smoothing(group_data, alpha=0.3):
    """Predict next value using exponential smoothing"""
    if len(group_data) == 0:
        return 0.0
    if len(group_data) == 1:
        return group_data.iloc[0]
    
    # Use exponential smoothing
    smoothed = exponential_smoothing(group_data.values, alpha=alpha)
    # Predict next value as last smoothed value
    return smoothed[-1]


def predict_with_weighted_average(group_data, window=5):
    """
    Weighted Moving Average
    
    This method takes the last N values (window) and gives exponentially decreasing
    weights to older values. The most recent value gets the highest weight, and
    each older value gets progressively less weight.
    
    How it works:
    - Takes the last 5 values (or fewer if not enough data)
    - Assigns weights that decrease exponentially: most recent = highest weight
    - Multiplies each value by its weight and sums them up
    
    Example:
    If you have values [10, 11, 12, 13, 14] (14 is most recent):
    - Weights might be: [0.05, 0.10, 0.18, 0.27, 0.40] (sums to 1.0)
    - Prediction = 10*0.05 + 11*0.10 + 12*0.18 + 13*0.27 + 14*0.40 = 12.6
    
    Why it's useful:
    - Emphasizes recent trends while still considering history
    - More responsive than simple average
    - Good for data with gradual changes
    """
    if len(group_data) == 0:
        return 0.0
    if len(group_data) == 1:
        return float(group_data.iloc[0])
    
    # Use last N values with exponential weights
    recent = group_data.tail(min(window, len(group_data)))
    recent_values = recent.values.flatten() if recent.values.ndim > 1 else recent.values
    weights = np.exp(np.linspace(-1, 0, len(recent_values)))
    weights = weights / weights.sum()
    return float(np.sum(recent_values * weights))


def predict_with_trend(group_data):
    """
    Linear Trend Extrapolation
    
    This method fits a straight line through recent data points and extends it
    forward to predict the next value. It assumes the trend will continue.
    
    How it works:
    - Takes the last 5-10 data points
    - Fits a line using linear regression (y = mx + b)
    - Extends the line one step forward to get the prediction
    
    Example:
    If recent values are [10, 11, 12, 13, 14] (clearly increasing):
    - The line might be: y = 1*x + 9 (slope of 1, intercept of 9)
    - Next value would be: 1*6 + 9 = 15
    
    Why it's useful:
    - Captures upward or downward trends
    - Works well when data has a clear direction
    - Simple and interpretable
    - Can be combined with other methods for robustness
    """
    if len(group_data) < 2:
        if len(group_data) == 1:
            return group_data.iloc[0]
        return 0.0
    
    # Use last 5-10 points for trend
    n = min(10, len(group_data))
    recent = group_data.tail(n).values
    x = np.arange(len(recent))
    
    # Linear regression
    coeffs = np.polyfit(x, recent, 1)
    # Predict next value
    next_x = len(recent)
    return coeffs[0] * next_x + coeffs[1]


def predict_ensemble(group_data):
    """
    Ensemble Forecasting Method
    
    This combines multiple forecasting methods to get a more robust prediction.
    Instead of relying on just one method, it uses several and averages them
    with weights.
    
    Methods used:
    1. Exponential Smoothing (alpha=0.2, 0.3, 0.5) - 30% weight each
    2. Weighted Moving Average - 20% weight
    3. Simple Moving Average - 10% weight
    4. Last Value (Naive) - 10% weight
    5. Linear Trend - 10% weight (if enough data)
    
    How it works:
    - Each method makes its own prediction
    - Predictions are combined using weighted average
    - Exponential smoothing gets most weight (60% total) because it's usually reliable
    - Other methods provide diversity and robustness
    
    Why it's useful:
    - More stable than single methods
    - Reduces risk of bad predictions from one method
    - Combines strengths of different approaches
    - Works well across different data patterns
    """
    if len(group_data) == 0:
        return 0.0
    if len(group_data) == 1:
        return group_data.iloc[0]
    
    predictions = []
    
    # Method 1: Exponential smoothing (multiple alphas)
    for alpha in [0.2, 0.3, 0.5]:
        try:
            pred = predict_with_exponential_smoothing(group_data, alpha=alpha)
            predictions.append(pred)
        except:
            pass
    
    # Method 2: Weighted average
    try:
        pred = predict_with_weighted_average(group_data, window=5)
        predictions.append(pred)
    except:
        pass
    
    # Method 3: Simple moving average
    try:
        pred = group_data.tail(5).mean()
        predictions.append(pred)
    except:
        pass
    
    # Method 4: Last value (naive)
    try:
        pred = group_data.iloc[-1]
        predictions.append(pred)
    except:
        pass
    
    # Method 5: Trend (if enough data)
    if len(group_data) >= 3:
        try:
            pred = predict_with_trend(group_data)
            predictions.append(pred)
        except:
            pass
    
    if len(predictions) == 0:
        return group_data.iloc[-1] if len(group_data) > 0 else 0.0
    
    # Weighted average of methods (favor exponential smoothing)
    predictions = np.array(predictions)
    weights = np.array([0.3, 0.3, 0.2, 0.1, 0.1][:len(predictions)])
    weights = weights / weights.sum()
    
    # Ensure shapes match
    if len(predictions) == len(weights):
        return np.average(predictions, weights=weights)
    else:
        # Fallback to simple average if shapes don't match
        return np.mean(predictions)


def forecast_sequential(train_df, test_df):
    """
    Sequential forecasting: for each test row, use only training data
    up to that point in time
    """
    print("Forecasting sequentially (no data leakage)...")
    
    # Sort by group and time
    train_df = train_df.sort_values(['code', 'sub_code', 'sub_category', 'horizon', 'ts_index']).reset_index(drop=True)
    test_df = test_df.sort_values(['code', 'sub_code', 'sub_category', 'horizon', 'ts_index']).reset_index(drop=True)
    
    predictions = []
    test_ids = []
    
    # Group test data by hierarchy
    test_groups = test_df.groupby(['code', 'sub_code', 'sub_category', 'horizon'])
    
    for (code, sub_code, sub_category, horizon), test_group in tqdm(test_groups, desc="Processing groups"):
        # Get training data for this group
        train_group = train_df[
            (train_df['code'] == code) &
            (train_df['sub_code'] == sub_code) &
            (train_df['sub_category'] == sub_category) &
            (train_df['horizon'] == horizon)
        ].copy()
        
        if len(train_group) == 0:
            # No training data, use median of all training
            default_pred = train_df['y_target'].median()
            predictions.extend([default_pred] * len(test_group))
            test_ids.extend(test_group['id'].values)
            continue
        
        # For each test row, use only training data up to that ts_index
        for _, test_row in test_group.iterrows():
            test_ts = test_row['ts_index']
            
            # Get training data up to (but not including) test_ts
            train_up_to_ts = train_group[train_group['ts_index'] < test_ts].copy()
            
            if len(train_up_to_ts) == 0:
                # No historical data, use group mean or global mean
                if len(train_group) > 0:
                    pred = train_group['y_target'].mean()
                else:
                    pred = train_df['y_target'].median()
            else:
                # Use time series forecasting
                y_series = train_up_to_ts['y_target']
                pred = predict_ensemble(y_series)
            
            predictions.append(pred)
            test_ids.append(test_row['id'])
    
    return pd.DataFrame({'id': test_ids, 'prediction': predictions})


def forecast_simple(train_df, test_df):
    """
    Simpler approach: forecast each group independently
    """
    print("Forecasting per group...")
    
    # Sort
    train_df = train_df.sort_values(['code', 'sub_code', 'sub_category', 'horizon', 'ts_index']).reset_index(drop=True)
    test_df = test_df.sort_values(['code', 'sub_code', 'sub_category', 'horizon', 'ts_index']).reset_index(drop=True)
    
    predictions = []
    
    # Group by hierarchy
    groups = test_df.groupby(['code', 'sub_code', 'sub_category', 'horizon'])
    
    for (code, sub_code, sub_category, horizon), test_group in tqdm(groups, desc="Processing groups"):
        # Get corresponding training data
        train_group = train_df[
            (train_df['code'] == code) &
            (train_df['sub_code'] == sub_code) &
            (train_df['sub_category'] == sub_category) &
            (train_df['horizon'] == horizon)
        ].copy()
        
        if len(train_group) == 0:
            # No training data, use median
            default_pred = train_df['y_target'].median()
            predictions.extend([default_pred] * len(test_group))
            continue
        
        # Get target series
        y_series = train_group['y_target'].sort_values()
        
        # Predict for each test row
        for _, test_row in test_group.iterrows():
            # Use all available training data (since test is future)
            pred = predict_ensemble(y_series)
            predictions.append(pred)
    
    return pd.DataFrame({
        'id': test_df['id'].values,
        'prediction': predictions
    })


def main():
    print("=" * 60)
    print("Time Series Forecasting")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    train_df = pd.read_parquet('data/train.parquet')
    test_df = pd.read_parquet('data/test.parquet')
    print(f"   Train: {len(train_df)} rows")
    print(f"   Test: {len(test_df)} rows")
    print(f"   Train target stats: mean={train_df['y_target'].mean():.4f}, "
          f"median={train_df['y_target'].median():.4f}")
    
    # Forecast
    print("\n2. Generating forecasts...")
    submission = forecast_simple(train_df, test_df)
    
    # Ensure we have all test IDs
    if len(submission) != len(test_df):
        print(f"   Warning: {len(submission)} predictions vs {len(test_df)} test rows")
        # Merge to ensure all IDs
        submission = test_df[['id']].merge(submission, on='id', how='left')
        submission['prediction'] = submission['prediction'].fillna(train_df['y_target'].median())
    
    # Sort and save
    submission = submission.sort_values('id')
    submission.to_csv('submission.csv', index=False)
    
    print(f"\n✅ Done! Saved {len(submission)} predictions to submission.csv")
    print("\nSample predictions:")
    print(submission.head(10).to_string(index=False))
    print(f"\nStats: min={submission['prediction'].min():.4f}, "
          f"max={submission['prediction'].max():.4f}, "
          f"mean={submission['prediction'].mean():.4f}, "
          f"median={submission['prediction'].median():.4f}")


if __name__ == "__main__":
    main()
