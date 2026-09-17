from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import pandas as pd
from vnstock.api.quote import Quote

SYMBOL = "VNINDEX"
START_DATE = "2012-09-01"
END_DATE = "2026-06-26"
OUTPUT_FILE = ROOT / "data" / "VNINDEX_daily_data.csv"


def scrape_vnindex():
    print(f"Fetching daily data for {SYMBOL} from {START_DATE} to {END_DATE}...")

    q = Quote(symbol=SYMBOL, source="VCI")
    df = q.history(start=START_DATE, end=END_DATE, interval="1D")

    if df is None or df.empty:
        print("No data returned.")
        return None

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
    print(df.tail())
    return df


if __name__ == "__main__":
    scrape_vnindex()
