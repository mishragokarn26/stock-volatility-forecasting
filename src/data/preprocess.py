import pandas as pd
import numpy as np
from pathlib import Path

def compute_returns(input_csv, save=True):
    """
    Compute log returns and squared returns from raw stock data.
    Uses 'Adj Close' if present, otherwise falls back to 'Close'.
    """
    df = pd.read_csv(input_csv, parse_dates=["Date"])
    # possible column names
    candidates = ["Adj Close", "Adj_Close", "AdjClose", "Close", "close"]
    price_col = next((c for c in candidates if c in df.columns), None)

    if price_col is None:
        raise KeyError(f"No price column found in {input_csv}. Columns: {df.columns.tolist()}")

    # ensure numeric
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])

    # compute returns
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    df["squared_return"] = df["log_return"] ** 2
    df.dropna(inplace=True)

    if save:
        out_path = Path("data/processed")
        out_path.mkdir(parents=True, exist_ok=True)
        ticker = Path(input_csv).stem.replace("_raw", "").replace("^", "")
        df.to_csv(out_path / f"{ticker}_processed.csv", index=False)
        print(f"Saved processed data to data/processed/{ticker}_processed.csv (price used: '{price_col}')")

    return df

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    for f in sorted(raw_dir.glob("*_raw.csv")):
        print(f"Processing {f.name} ...")
        compute_returns(str(f))
