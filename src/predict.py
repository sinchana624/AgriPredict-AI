
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yield_model.joblib"

FEATURES = [
    "year",
    "crop",
    "rainfall_ann",
    "tmax_ann",
    "tmean_ann",
    "tmin_ann",
]


def predict_yield(input_data):
    """Predict crop yield in tonnes per hectare."""
    missing = [feature for feature in FEATURES if feature not in input_data]

    if missing:
        raise ValueError(
            f"Missing required features: {', '.join(missing)}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    input_df = pd.DataFrame([{
        feature: input_data[feature] for feature in FEATURES
    }])

    prediction = float(model.predict(input_df)[0])

    if not pd.notna(prediction):
        raise ValueError("Model returned an invalid prediction.")

    return prediction
