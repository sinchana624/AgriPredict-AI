
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "yield_model.joblib"

FEATURES = [
    "year",
    "crop",
    "rainfall_ann",
    "tmax_ann",
    "tmean_ann",
    "tmin_ann",
]


def predict_yield(year, crop, rainfall_ann, tmax_ann, tmean_ann, tmin_ann):
    """Predict crop yield using the six-feature trained model."""
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

    try:
        weather = [
            float(rainfall_ann),
            float(tmax_ann),
            float(tmean_ann),
            float(tmin_ann),
        ]
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Weather features must be numeric") from None

    if not np.isfinite(weather).all():
        raise ValueError("Weather features must be finite numbers")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    features = pd.DataFrame([{
        "year": year,
        "crop": crop,
        "rainfall_ann": weather[0],
        "tmax_ann": weather[1],
        "tmean_ann": weather[2],
        "tmin_ann": weather[3],
    }], columns=FEATURES)

    prediction = model.predict(features)
    return float(prediction[0])
