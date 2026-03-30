import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

# Define asset tickers to pull from yahoo finance
ASSET_CONFIG = {
    "btc": {"ticker": "BTC-USD", "prefix": "btc"},
    "eth": {"ticker": "ETH-USD", "prefix": "eth"},
    "sp500": {"ticker": "^GSPC", "prefix": "sp500"},
    "vti": {"ticker": "VTI", "prefix": "vti"},
    "agg": {"ticker": "AGG", "prefix": "agg"},
    "tlt": {"ticker": "TLT", "prefix": "tlt"},
}

# Select period 2014-2025
DEFAULT_START_DATE = "2014-01-01"
DEFAULT_END_DATE = "2026-01-01"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data"

COLUMN_ORDER = [
    "date",
    "btc_adj_close",
    "btc_volume",
    "eth_adj_close",
    "eth_volume",
    "sp500_adj_close",
    "sp500_volume",
    "vti_adj_close",
    "vti_volume",
    "agg_adj_close",
    "agg_volume",
    "tlt_adj_close",
    "tlt_volume",
]

# Pull from Yahoo Finance
def get_yf_data(ticker_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download daily data from Yahoo Finance and normalize column names."""
    print(f"Downloading {ticker_name} from {start_date} to {end_date}")
    raw_data = yf.download(ticker_name, start=start_date, end=end_date, auto_adjust=False, progress=False)

    if raw_data.empty:
        raise ValueError(f"No data found for ticker: {ticker_name}")

    df = raw_data.reset_index().copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.columns = ["date", "adj_close", "close", "high", "low", "open", "volume"]
    return df[["date", "open", "high", "low", "close", "adj_close", "volume"]]

# Select date, adjusted closing prices and volume columns 
def make_subset(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Keep only date, adjusted close, and volume, then prefix the output columns."""
    subset = df[["date", "adj_close", "volume"]].copy()
    subset.columns = ["date", f"{prefix}_adj_close", f"{prefix}_volume"]
    return subset

# Align all series by date and merge into one dataset
def merge_data(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge equities, bonds, and crypto series into one aligned dataset."""
    print("Preparing merged dataset")

    sp500 = make_subset(frames["sp500"], "sp500")
    vti = make_subset(frames["vti"], "vti")
    agg = make_subset(frames["agg"], "agg")
    tlt = make_subset(frames["tlt"], "tlt")
    btc = make_subset(frames["btc"], "btc")
    eth = make_subset(frames["eth"], "eth")

    equity_data = pd.merge(sp500, vti, on="date", how="inner")
    bond_data = pd.merge(agg, tlt, on="date", how="inner")
    crypto_data = pd.merge(btc, eth, on="date", how="left")

    all_data = pd.merge(equity_data, bond_data, on="date", how="inner")
    all_data = pd.merge(all_data, crypto_data, on="date", how="left")
    all_data = all_data[COLUMN_ORDER].sort_values("date").reset_index(drop=True)

    return all_data

# Export individual files and merged CSV
def export_data(frames: dict[str, pd.DataFrame], merged: pd.DataFrame, output_dir: Path) -> None:
    """Export each individual asset file and the merged dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting data to {output_dir}")

    for name, df in frames.items():
        filename = output_dir / f"{name}_data.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {filename}")

    merged_file = output_dir / "merged_data_v2.csv"
    merged.to_csv(merged_file, index=False)
    print(f"Saved {merged_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download financial time series data and export prepared CSVs.")
    parser.add_argument("--start_date", default=DEFAULT_START_DATE, help="Start date for data retrieval in YYYY-MM-DD format")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="End date for data retrieval in YYYY-MM-DD format")
    parser.add_argument("--output_dir", default=None, help="Directory to save output CSV files")
    parser.add_argument("--no_export", action="store_true", help="Download and prepare data but do not write any files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

    data_frames: dict[str, pd.DataFrame] = {}
    for name, config in ASSET_CONFIG.items():
        data_frames[name] = get_yf_data(config["ticker"], args.start_date, args.end_date)

    for df in data_frames.values():
        df.sort_values("date", inplace=True)

    merged_data = merge_data(data_frames)

    if args.no_export:
        print("Data prepared successfully, export skipped by --no-export")
        return

    export_data(data_frames, merged_data, output_dir)
    print("Data retrieval and export completed.")


if __name__ == "__main__":
    main()