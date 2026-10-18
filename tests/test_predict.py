
import math

import pytest

from src.predict import predict_yield


def test_prediction_returns_finite_number():
    result = predict_yield(2024, "Rice")

    assert isinstance(result, float)
    assert math.isfinite(result)


def test_empty_crop_raises_error():
    with pytest.raises(ValueError, match="crop"):
        predict_yield(2024, "   ")


def test_invalid_year_raises_error():
    with pytest.raises(ValueError, match="year"):
        predict_yield("invalid", "Rice")
