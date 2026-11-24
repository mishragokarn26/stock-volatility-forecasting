import pandas as pd
import numpy as np

def create_features(df, lags=5, rolling_window=21):
    df = df.copy()

    # Lag features
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["log_return"].shift(lag)

    # Rolling features
    df["rolling_vol"] = df["log_return"].rolling(rolling_window).std()
    df["rolling_mean"] = df["log_return"].rolling(rolling_window).mean()

    # Target variable (next day volatility)
    df["target"] = df["rolling_vol"].shift(-1)

    df.dropna(inplace=True)

    return df