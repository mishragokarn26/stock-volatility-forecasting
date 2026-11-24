import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.utils.model_saver import save_model

def train_models(df, config):
    features = [col for col in df.columns if col.startswith("lag_") or "rolling" in col]
    
    X = df[features]
    y = df["target"]

    # ---- Random Forest ----
    rf_params = config["models"]["ml"]["random_forest"]
    rf = RandomForestRegressor(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params["max_depth"],
        random_state=42
    )
    rf.fit(X, y)

    # ---- XGBoost ----
    xgb_params = config["models"]["ml"]["xgboost"]
    xgb = XGBRegressor(
        n_estimators=xgb_params["n_estimators"],
        learning_rate=xgb_params["learning_rate"],
        random_state=42
    )
    xgb.fit(X, y)

    # Save models
    save_model(rf, "models/ml/random_forest.pkl")
    save_model(xgb, "models/ml/xgboost.pkl")

    return rf, xgb, X, y