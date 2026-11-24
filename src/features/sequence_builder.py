import numpy as np

def create_sequences(data, target_col="target", seq_length=20):
    X, y = [], []

    values = data.values

    for i in range(len(values) - seq_length):
        X.append(values[i:i+seq_length])
        y.append(values[i+seq_length, data.columns.get_loc(target_col)])

    return np.array(X), np.array(y)