import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from tqdm import tqdm


from lstm1 import TimeSeriesLSTM


def run_prediction():
    # Configuration
    WINDOW_SIZE = 5
    MODEL_WEIGHTS = 'lstm_hedge_fund.pth'
    SCALER_PATH = 'scaler.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data and Preprocessing Assets
    print("Loading test data and scaler...")
    test_df = pd.read_parquet('data/test.parquet')
    scaler = joblib.load(SCALER_PATH)
    feat_cols = [c for c in test_df.columns if c.startswith('feature_')]

    # 2. Reconstruct Model and Load Weights
    print(f"Loading weights from {MODEL_WEIGHTS}...")
    model = TimeSeriesLSTM(input_dim=len(feat_cols), hidden_dim=64, num_layers=2)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.to(device)
    model.eval()

    # 3. Scaling
    test_df[feat_cols] = scaler.transform(test_df[feat_cols])

    # Ensure chronological order per asset
    test_df = test_df.sort_values(['code', 'ts_index'])

    # 4. Sliding Window Inference
    all_preds = []
    print("Generating predictions...")
    with torch.no_grad():
        for _, group in tqdm(test_df.groupby('code')):
            features = group[feat_cols].values.astype(np.float32)

            for i in range(len(features)):
                # Handle start-of-sequence with padding
                if i < WINDOW_SIZE - 1:
                    window = np.zeros((WINDOW_SIZE, len(feat_cols)), dtype=np.float32)
                    current_chunk = features[:i + 1]
                    window[-len(current_chunk):] = current_chunk
                else:
                    window = features[i - WINDOW_SIZE + 1: i + 1]

                # Reshape to [1, Window, Features]
                x_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
                pred = model(x_tensor).cpu().item()
                all_preds.append(pred)

    # 5. Attach predictions to DataFrame
    if len(all_preds) != len(test_df):
        raise ValueError(f"Number of predictions ({len(all_preds)}) "
                         f"does not match number of rows in test_df ({len(test_df)}).")

    test_df['prediction'] = all_preds
    submission = test_df[['id', 'prediction']]

    # Quick sanity check: show a few rows
    print("\nSample predictions:")
    print(submission.head().to_string(index=False))

    # 6. Save Final Submission
    submission.to_csv('submission.csv', index=False)
    print("\n✅ Success! Predictions saved to submission.csv")


if __name__ == "__main__":
    run_prediction()