import pytest

from src.predict import predict_yield


def test_valid_prediction_is_finite_number():
    result = predict_yield(2024, "Rice")
    assert isinstance(result, float)
    assert result == result  # Not NaN


def test_crop_name_is_case_insensitive():
    assert predict_yield(2024, "rice") == pytest.approx(
        predict_yield(2024, "Rice")
    )


def test_unknown_crop_is_rejected():
    with pytest.raises(ValueError, match="Unknown crop"):
        predict_yield(2024, "Banana")


@pytest.mark.parametrize("year", ["not-a-year", 2024.5, 1800, True])
def test_invalid_year_is_rejected(year):
    with pytest.raises(ValueError, match="Year"):
        predict_yield(year, "Rice")


def test_empty_crop_is_rejected():
    with pytest.raises(ValueError, match="Crop"):
        predict_yield(2024, "  ")


def test_missing_model_is_reported(tmp_path):
    missing_model = tmp_path / "missing_model.joblib"
    with pytest.raises(FileNotFoundError, match="Trained model not found"):
        predict_yield(2024, "Rice", model_path=missing_model)
