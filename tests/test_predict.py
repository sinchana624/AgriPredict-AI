
import math

import pytest

from src.predict import predict_yield


def test_prediction_returns_finite_number():
    result = predict_yield(
        2024, "Rice", 850.0, 30.0, 25.0, 20.0
    )

    assert isinstance(result, float)
    assert math.isfinite(result)


def test_empty_crop_raises_error():
    with pytest.raises(ValueError, match="crop"):
        predict_yield(
            2024, " ", 850.0, 30.0, 25.0, 20.0
        )


def test_invalid_year_raises_error():
    with pytest.raises(ValueError, match="year"):
        predict_yield(
            "invalid", "Rice", 850.0, 30.0, 25.0, 20.0
        )


def test_non_numeric_weather_raises_error():
    with pytest.raises(ValueError, match="numeric"):
        predict_yield(
            2024, "Rice", "invalid", 30.0, 25.0, 20.0
        )


def test_non_finite_weather_raises_error():
    with pytest.raises(ValueError, match="finite"):
        predict_yield(
            2024, "Rice", float("nan"), 30.0, 25.0, 20.0
        )
