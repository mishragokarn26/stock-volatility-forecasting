import joblib
from pathlib import Path

def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Deep learning models
    if "keras" in str(type(model)):
        model.save(path.replace(".pkl", ".h5"))
    else:
        joblib.dump(model, path)

def load_model(path):
    if path.endswith(".h5"):
        from tensorflow.keras.models import load_model
        return load_model(path)
    return joblib.load(path)