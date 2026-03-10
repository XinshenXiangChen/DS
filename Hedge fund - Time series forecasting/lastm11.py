import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, df, window_size=5):
        # Ensure chronological order
        self.df = df.sort_values(['code', 'ts_index']).reset_index(drop=True)
        self.window_size = window_size

        # Get feature column names
        self.feat_cols = [c for c in df.columns if c.startswith('feature_')]

        # Pre-calculate indices where a full window is available
        self.valid_indices = []
        codes = self.df['code'].values
        for i in range(self.window_size, len(self.df)):
            if codes[i] == codes[i - self.window_size]:
                self.valid_indices.append(i)

        # Store as float32 to save memory
        self.features = self.df[self.feat_cols].values.astype(np.float32)
        self.targets = self.df['y_target'].values.astype(np.float32)
        self.weights = self.df['weight'].values.astype(np.float32)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.window_size

        # Return window (window_size x num_features)
        return torch.tensor(self.features[start_idx:end_idx]), \
            torch.tensor(self.targets[end_idx]), \
            torch.tensor(self.weights[end_idx])