import yfinance as yf
import pandas as pd
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.retry import retry

logger = get_logger()

@retry(max_retries=3, delay=3)
def download_data(ticker="AAPL", start="2010-01-01", end=None, save=True):
    logger.info(f"Downloading {ticker} data...")

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    df.reset_index(inplace=True)

    if save:
        path = Path("data/raw")
        path.mkdir(parents=True, exist_ok=True)

        safe_ticker = ticker.replace("^", "")
        file_path = path / f"{safe_ticker}_raw.csv"

        df.to_csv(file_path, index=False)
        logger.info(f"Saved raw data to {file_path}")

    return df