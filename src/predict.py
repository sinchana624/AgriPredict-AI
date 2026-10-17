from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "yield_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "combined_crop_yield.csv"


def predict_yield(year, crop, model_path=None):
    """Predict historical-trend crop yield in tonnes per hectare."""
    if isinstance(year, bool):
        raise ValueError("Year must be a valid integer.")

    try:
        numeric_year = float(year)
    except (TypeError, ValueError):
        raise ValueError("Year must be a valid integer.") from None

    if not numeric_year.is_integer() or not 1900 <= numeric_year <= 2200:
        raise ValueError("Year must be an integer between 1900 and 2200.")

    if not isinstance(crop, str) or not crop.strip():
        raise ValueError("Crop must be a non-empty string.")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Crop dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    available_crops = {
        str(value).strip().casefold(): str(value).strip()
        for value in data["crop"].dropna().unique()
    }

    crop_key = crop.strip().casefold()
    if crop_key not in available_crops:
        choices = ", ".join(sorted(available_crops.values()))
        raise ValueError(f"Unknown crop '{crop}'. Available crops: {choices}")

    selected_model_path = Path(model_path) if model_path else MODEL_PATH
    if not selected_model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {selected_model_path}"
        )

    model = joblib.load(selected_model_path)
    input_data = pd.DataFrame({
        "year": [int(numeric_year)],
        "crop": [available_crops[crop_key]],
    })

    prediction = float(model.predict(input_data)[0])

    if not pd.notna(prediction):
        raise ValueError("The model returned an invalid prediction.")

    return prediction


if __name__ == "__main__":
    result = predict_yield(2024, "Rice")
    print(f"Predicted rice yield for 2024: {result:.3f} tonnes/hectare")
