"""
test_model.py
-------------
Unit tests for model inference.
Tests input validation and prediction output structure.
"""

import pytest
from src.model_inference import validate_input, predict, compute_confidence_interval


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_input():
    """Return a complete valid input dict for prediction."""
    return {
        "store_id": 1,
        "promotion": 1,
        "holiday": 0,
        "weekend": 1,
        "month": 12,
        "day_of_week": 5,
        "quarter": 4,
        "lag_7": 850.0,
        "lag_30": 800.0,
        "rolling_7": 820.0,
        "rolling_30": 810.0,
        "category": "Electronics",
    }


# ── validate_input tests ──────────────────────────────────────────────────────

class TestValidateInput:

    def test_passes_on_valid_input(self, valid_input):
        """Valid input should pass without raising."""
        validate_input(valid_input)  # should not raise

    def test_raises_on_missing_field(self, valid_input):
        """Missing any required field should raise ValueError."""
        del valid_input["category"]
        with pytest.raises(ValueError, match="Missing required fields"):
            validate_input(valid_input)

    def test_raises_on_invalid_store_id(self, valid_input):
        """store_id outside 1-5 should raise ValueError."""
        valid_input["store_id"] = 99
        with pytest.raises(ValueError, match="store_id"):
            validate_input(valid_input)

    def test_raises_on_invalid_category(self, valid_input):
        """Unknown category should raise ValueError."""
        valid_input["category"] = "InvalidCategory"
        with pytest.raises(ValueError, match="category"):
            validate_input(valid_input)

    def test_raises_on_invalid_flag(self, valid_input):
        """promotion/holiday/weekend values outside 0-1 should raise."""
        valid_input["promotion"] = 5
        with pytest.raises(ValueError, match="promotion"):
            validate_input(valid_input)


# ── compute_confidence_interval tests ────────────────────────────────────────

class TestConfidenceInterval:

    def test_interval_width(self):
        """Interval should be ±10% of prediction by default."""
        lower, upper = compute_confidence_interval(1000.0)
        assert lower == 900.0
        assert upper == 1100.0

    def test_custom_margin(self):
        """Custom margin should be applied correctly."""
        lower, upper = compute_confidence_interval(1000.0, margin=0.20)
        assert lower == 800.0
        assert upper == 1200.0


# ── predict tests ─────────────────────────────────────────────────────────────

class TestPredict:

    def test_returns_expected_keys(self, valid_input):
        """Prediction result should contain all three expected keys."""
        result = predict(valid_input)
        assert "prediction" in result
        assert "lower_bound" in result
        assert "upper_bound" in result

    def test_prediction_is_positive(self, valid_input):
        """Sales prediction should always be a positive number."""
        result = predict(valid_input)
        assert result["prediction"] > 0

    def test_bounds_are_correct(self, valid_input):
        """Lower bound should be less than prediction, upper bound greater."""
        result = predict(valid_input)
        assert result["lower_bound"] < result["prediction"] < result["upper_bound"]