"""
Merges Kaggle VNINDEX 1-hour data (2012-2024) with vnstock 1-hour data (2024-2026)
into a single complete file: VNINDEX_1h_data.csv
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from vnstock.api.quote import Quote

KAGGLE_FILE = ROOT / "data" / "HOSEVNINDEXH1.csv"
OUTPUT_FILE = ROOT / "data" / "VNINDEX_1h_data.csv"
KEEP_COLS = ["time", "open", "high", "low", "close"]


def load_kaggle():
    df = pd.read_csv(KAGGLE_FILE)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[KEEP_COLS].copy()
    return df


def fetch_vnstock():
    print("Fetching recent 1H data from vnstock (2024-07 → 2026-06-26)...")
    q = Quote(symbol="VNINDEX", source="VCI")
    df = q.history(start="2024-07-01", end="2026-06-26", interval="1H")
    df["time"] = pd.to_datetime(df["time"])
    df = df[KEEP_COLS].copy()
    return df


def main():
    print("=== Building complete VNINDEX 1-hour dataset ===\n")

    kaggle = load_kaggle()
    print(f"Kaggle:   {len(kaggle):,} rows  {kaggle['time'].min().date()} → {kaggle['time'].max().date()}")

    recent = fetch_vnstock()
    print(f"vnstock:  {len(recent):,} rows  {recent['time'].min().date()} → {recent['time'].max().date()}")

    # Merge: kaggle base + vnstock for anything after kaggle ends
    cutoff = kaggle["time"].max()
    new_rows = recent[recent["time"] > cutoff].copy()
    print(f"\nNew rows after {cutoff.date()}: {len(new_rows)}")

    combined = pd.concat([kaggle, new_rows], ignore_index=True)
    combined = combined.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\nFinal dataset: {len(combined):,} rows")
    print(f"Range: {combined['time'].min().date()} → {combined['time'].max().date()}")
    years = (combined["time"].max() - combined["time"].min()).days / 365.25
    print(f"Span:  {years:.1f} years")
    print(f"\nSaved → {OUTPUT_FILE}")
    print(combined.tail(3))


if __name__ == "__main__":
    main()
