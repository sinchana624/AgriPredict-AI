
import math

import pytest

from src.predict import predict_yield


def test_prediction_returns_finite_number():
    input_data = {
        "year": 1991,
        "crop": "Maize",
        "rainfall_ann": 1312.2178273809525,
        "tmax_ann": 33.15654761904762,
        "tmean_ann": 27.6742261904762,
        "tmin_ann": 22.74300595238095,
    }

    result = predict_yield(input_data)

    assert isinstance(result, float)
    assert math.isfinite(result)


def test_missing_feature_raises_error():
    input_data = {
        "year": 1991,
        "crop": "Maize",
    }

    with pytest.raises(ValueError, match="Missing required features"):
        predict_yield(input_data)
