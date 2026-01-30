import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

np.random.seed(42)
torch.manual_seed(42)

time_steps = 1200
t = np.arange(time_steps)

trend = 0.01 * t
seasonal = np.sin(2 * np.pi * t / 24)
seasonal_2 = np.sin(2 * np.pi * t / 48)
noise = np.random.normal(0, 0.3, time_steps)

target = trend + seasonal + noise
x1 = seasonal_2 + np.random.normal(0, 0.2, time_steps)
x2 = np.random.normal(0, 1, time_steps)

data = pd.DataFrame({
    "target": target,
    "x1": x1,
    "x2": x2
})

LOOKBACK = 48
HORIZON = 1

def create_sequences(df, lookback, horizon):
    X, y = [], []
    values = df.values
    for i in range(len(values) - lookback - horizon):
        X.append(values[i:i+lookback])
        y.append(values[i+lookback:i+lookback+horizon, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(data, LOOKBACK, HORIZON)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs):
        energy = torch.tanh(self.attn(encoder_outputs))
        attention_weights = torch.softmax(self.v(energy), dim=1)
        context = torch.sum(attention_weights * encoder_outputs, dim=1)
        return context, attention_weights

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output, attn_weights

def train_and_evaluate(train_X, train_y, test_X, test_y):
    train_ds = TimeSeriesDataset(train_X, train_y)
    test_ds = TimeSeriesDataset(test_X, test_y)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = LSTMAttentionModel(input_dim=3, hidden_dim=64)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds, _ = model(xb)
            loss = criterion(preds.squeeze(), yb.squeeze())
            loss.backward()
            optimizer.step()

    model.eval()
    preds_list, actual_list, attn_list = [], [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            preds, attn = model(xb)
            preds_list.extend(preds.squeeze().numpy())
            actual_list.extend(yb.squeeze().numpy())
            attn_list.append(attn.numpy())

    rmse = math.sqrt(mean_squared_error(actual_list, preds_list))
    mae = mean_absolute_error(actual_list, preds_list)
    mape = np.mean(np.abs((actual_list - preds_list) / actual_list)) * 100

    return rmse, mae, mape, np.concatenate(attn_list)

splits = [(0, 700), (0, 800), (0, 900)]

for start, end in splits:
    train_X, test_X = X[start:end], X[end:end+100]
    train_y, test_y = y[start:end], y[end:end+100]

    rmse, mae, mape, attn_weights = train_and_evaluate(
        train_X, train_y, test_X, test_y
    )

    print("Rolling Window Result")
    print("RMSE:", round(rmse, 3))
    print("MAE :", round(mae, 3))
    print("MAPE:", round(mape, 2), "%")
    print("-" * 40)
