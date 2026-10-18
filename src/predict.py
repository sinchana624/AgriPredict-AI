
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "yield_model.joblib"


def predict_yield(year, crop):
    """Predict crop yield using the trained year-and-crop baseline."""
    try:
        year = int(year)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("year must be a valid integer") from None

    if isinstance(crop, str):
        crop = crop.strip()
    else:
        crop = ""

    if not crop:
        raise ValueError("crop must be a non-empty string")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    features = pd.DataFrame([{
        "year": year,
        "crop": crop,
    }])

    prediction = model.predict(features)
    return float(prediction[0])
