"""
Phase 1: Baseline models, ARIMA and Prophet on daily VN-Index data (10 years).
Saves predictions and metrics to baseline_results.csv and baseline_metrics.json.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import warnings
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

TRAIN_RATIO = 0.85
INPUT_FILE = ROOT / "data" / "VNINDEX_daily_data.csv"
PRED_FILE = ROOT / "results" / "baseline_predictions.csv"
METRICS_FILE = ROOT / "results" / "baseline_metrics.json"


def load_data():
    df = pd.read_csv(INPUT_FILE, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df[["time", "close"]].dropna()
    return df


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def run_arima(train_series, test_series):
    from pmdarima import auto_arima

    print("  Fitting Auto-ARIMA...")
    model = auto_arima(
        train_series,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=5, max_q=5, max_d=2,
    )
    print(f"  Best ARIMA order: {model.order}")

    preds = []
    history = list(train_series)
    for _ in range(len(test_series)):
        model.update(history[-1:])
        fc = model.predict(n_periods=1)[0]
        preds.append(fc)
        history.append(test_series.iloc[len(preds) - 1])

    return np.array(preds)


def run_prophet(train_df, test_df):
    from prophet import Prophet

    print("  Fitting Prophet...")
    df_train = train_df.rename(columns={"time": "ds", "close": "y"})
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df_train, verbose=False)

    future = pd.DataFrame({"ds": test_df["time"]})
    forecast = model.predict(future)
    preds = forecast["yhat"].values
    return preds


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape_val = mape(y_true, y_pred)
    print(f"  {name} → MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape_val:.2f}%")
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape_val, 4)}


def main():
    print("=== Phase 1: Baseline Models (ARIMA + Prophet) ===\n")
    df = load_data()
    print(f"Loaded {len(df)} daily rows from {df['time'].min().date()} to {df['time'].max().date()}")

    split = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows\n")

    results = pd.DataFrame({"time": test_df["time"].values, "actual": test_df["close"].values})
    metrics = {}

    # --- ARIMA ---
    print("[ARIMA]")
    arima_preds = run_arima(train_df["close"], test_df["close"])
    results["arima_pred"] = arima_preds
    metrics["ARIMA"] = evaluate("ARIMA", test_df["close"].values, arima_preds)

    # --- Prophet ---
    print("\n[Prophet]")
    prophet_preds = run_prophet(train_df, test_df)
    results["prophet_pred"] = prophet_preds
    metrics["Prophet"] = evaluate("Prophet", test_df["close"].values, prophet_preds)

    results.to_csv(PRED_FILE, index=False)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved predictions → {PRED_FILE}")
    print(f"Saved metrics     → {METRICS_FILE}")
    print("\nBaseline metrics summary:")
    print(pd.DataFrame(metrics).T.to_string())


if __name__ == "__main__":
    main()
