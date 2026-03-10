import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from lastm11 import TimeSeriesDataset
from tqdm import tqdm
import joblib

# Define the Model
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(TimeSeriesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

if __name__ == "__main__":
    # Setup
    train_df = pd.read_parquet('data/train.parquet')
    feat_cols = [c for c in train_df.columns if c.startswith('feature_')]

    # Scale
    scaler = StandardScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    
    # Save scaler for inference
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved!")

    # DataLoader
    dataset = TimeSeriesDataset(train_df, window_size=5)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)

    # Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesLSTM(input_dim=len(feat_cols), hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss(reduction='none')

    print(f"Training on {device}...")
    model.train()
    for epoch in range(5):
        for x, y, w in tqdm(loader, desc=f"Epoch {epoch+1}"):
            x, y, w = x.to(device), y.to(device), w.to(device)
            optimizer.zero_grad()
            loss = (criterion(model(x).squeeze(), y) * w).mean()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'lstm_hedge_fund.pth')
    print("Model saved!")