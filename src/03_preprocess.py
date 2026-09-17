"""
Preprocesses VNINDEX_1h_data.csv:
  1. Converts Kaggle portion (UTC) to ICT (UTC+7)
  2. Deduplicates after timezone alignment
  3. Forward-fills missing candles within trading days only
  4. Validates OHLC integrity
  5. Saves to VNINDEX_1h_clean.csv
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import pandas as pd
import numpy as np

INPUT_FILE  = ROOT / "data" / "VNINDEX_1h_data.csv"
OUTPUT_FILE = ROOT / "data" / "VNINDEX_1h_clean.csv"

KAGGLE_UTC_HOURS  = {2, 3, 4, 6, 7}
TRADING_HOURS     = [9, 10, 11, 13, 14]   # HOSE ICT session hours


def main():
    print("=== Preprocessing VNINDEX 1-hour data ===\n")
    df = pd.read_csv(INPUT_FILE, parse_dates=["time"])
    print(f"Loaded {len(df):,} rows")

    # --- Step 1: Convert Kaggle UTC timestamps → ICT ---
    is_utc = df["time"].dt.hour.isin(KAGGLE_UTC_HOURS)
    print(f"Kaggle (UTC) rows : {is_utc.sum():,}")
    print(f"vnstock (ICT) rows: {(~is_utc).sum():,}")
    df.loc[is_utc, "time"] = df.loc[is_utc, "time"] + pd.Timedelta(hours=7)
    print("Timezone conversion done (UTC → ICT)\n")

    # --- Step 2: Deduplicate (keep vnstock values at overlaps) ---
    before = len(df)
    df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    print(f"Duplicates removed : {before - len(df)}")

    # --- Step 3: Forward-fill missing candles within trading days only ---
    # Build a complete index of all expected trading timestamps
    trading_dates = df["time"].dt.normalize().unique()
    expected_timestamps = pd.DatetimeIndex([
        date + pd.Timedelta(hours=h)
        for date in trading_dates
        for h in TRADING_HOURS
    ])

    df = df.set_index("time")
    df = df.reindex(expected_timestamps)   # inserts NaN rows for missing slots
    filled = df.isnull().any(axis=1).sum()
    df = df.ffill()                        # forward-fill OHLC from previous candle
    df = df.reset_index().rename(columns={"index": "time"})
    print(f"Missing candles filled: {filled}")

    # --- Step 4: OHLC integrity check ---
    bad = (
        (df["high"] < df["low"]) |
        (df["close"] > df["high"]) |
        (df["close"] < df["low"]) |
        (df["open"]  > df["high"]) |
        (df["open"]  < df["low"])
    )
    print(f"OHLC integrity violations: {bad.sum()}")

    # --- Step 5: Final summary ---
    total_days = df["time"].dt.normalize().nunique()
    print(f"\nFinal rows    : {len(df):,}")
    print(f"Trading days  : {total_days:,}")
    print(f"Candles/day   : {len(df)/total_days:.2f} (expected 5.0)")
    print(f"Date range    : {df['time'].min()} → {df['time'].max()}")
    years = (df["time"].max() - df["time"].min()).days / 365.25
    print(f"Span          : {years:.1f} years")
    print(f"Null values   : {df.isnull().sum().sum()}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved → {OUTPUT_FILE}")
    print(df.tail(5))


if __name__ == "__main__":
    main()
