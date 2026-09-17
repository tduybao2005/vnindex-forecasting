"""
Phase 3: LSTM model on VNINDEX 1-hour data (2012-2026).
- Input  : OHLC features, 150-candle lookback window
- Target : next candle's close price
- Saves  : lstm_predictions.csv, lstm_metrics.json
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT_FILE   = ROOT / "data" / "VNINDEX_1h_clean.csv"
PRED_FILE    = ROOT / "results" / "lstm_predictions.csv"
METRICS_FILE = ROOT / "results" / "lstm_metrics.json"

# --- Hyperparameters ---
LOOKBACK    = 150       # 150 hourly candles = 30 trading sessions (HOSE trades 5 h/day)
FEATURES    = ["open", "high", "low", "close"]
TARGET      = "close"
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15    # remaining 0.15 = test
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-4
PATIENCE    = 20        # early stopping


# ── Dataset ──────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ── Model ─────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def make_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, -1])          # last column = close (target)
    return np.array(X), np.array(y)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Phase 3: LSTM Model (VNINDEX 1-hour) ===\n")

    # --- Load data ---
    df = pd.read_csv(INPUT_FILE, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"Loaded {len(df):,} rows  {df['time'].min().date()} → {df['time'].max().date()}")

    # --- Scale features ---
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES].values)

    # Separate close scaler for inverse-transforming predictions
    close_idx = FEATURES.index(TARGET)
    close_scaler = MinMaxScaler()
    close_scaler.fit(df[[TARGET]].values)

    # --- Build sequences ---
    X, y = make_sequences(scaled, LOOKBACK)
    print(f"Sequences: X={X.shape}  y={y.shape}")

    # --- Train / Val / Test split (no shuffle — time series) ---
    n = len(X)
    t1 = int(n * TRAIN_RATIO)
    t2 = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, y_train = X[:t1], y[:t1]
    X_val,   y_val   = X[t1:t2], y[t1:t2]
    X_test,  y_test  = X[t2:], y[t2:]

    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TimeSeriesDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

    # --- Build model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = LSTMModel(len(FEATURES), HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # --- Training loop ---
    print(f"\nTraining {EPOCHS} epochs (early stopping patience={PATIENCE})...\n")
    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(X_val)

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # --- Evaluate on test set ---
    model.load_state_dict(best_state)
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds_scaled = model(X_test_t).cpu().numpy()

    # Inverse-transform predictions and actuals
    preds  = close_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    actual = close_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mape_val = mape(actual, preds)

    print(f"\n{'='*45}")
    print(f"  LSTM Test Results")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape_val:.2f}%")
    print(f"{'='*45}")

    # --- Save results ---
    test_times = df["time"].iloc[LOOKBACK + t2:].reset_index(drop=True)
    results = pd.DataFrame({
        "time"      : test_times,
        "actual"    : actual,
        "lstm_pred" : preds,
    })
    results.to_csv(PRED_FILE, index=False)

    metrics = {"LSTM": {
        "MAE" : round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape_val), 4),
    }}
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved predictions → {PRED_FILE}")
    print(f"Saved metrics     → {METRICS_FILE}")
    print(results.tail(5))


if __name__ == "__main__":
    main()
