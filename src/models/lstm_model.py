import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from src.utils.model_saver import save_model

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


def train_lstm(X_train, y_train, config):
    params = config["models"]["lstm"]

    model = build_lstm((X_train.shape[1], X_train.shape[2]))

    model.fit(
        X_train, y_train,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=1
    )

    save_model(model, "models/dl/lstm_model.pkl")

    return model