"""
model_inference.py
------------------
Handles loading the best saved model and generating
predictions for new input data.

Responsibilities:
    - Load the best model from disk
    - Validate incoming input data
    - Transform raw input into model-ready features
    - Return predictions with a confidence interval
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import (
    BEST_MODEL_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_best_model():
    """
    Load the best trained model pipeline from disk.

    Returns:
        Fitted sklearn Pipeline.

    Raises:
        FileNotFoundError: If no trained model exists yet.
    """
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at '{BEST_MODEL_PATH}'. "
            "Run src/model_training.py first."
        )

    with open(BEST_MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"✅ Model loaded from '{BEST_MODEL_PATH.name}'")
    return model


# ── Input Validation ──────────────────────────────────────────────────────────

def validate_input(input_data: dict) -> None:
    """
    Validate that all required fields are present and within
    acceptable ranges.

    Args:
        input_data: Dict of feature name → value from user input.

    Raises:
        ValueError: If any field is missing or out of range.
    """
    required_fields = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    # Check all fields are present
    missing = [f for f in required_fields if f not in input_data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    # Check numeric ranges
    if not (1 <= input_data["store_id"] <= 5):
        raise ValueError("store_id must be between 1 and 5.")

    if not (1 <= input_data["month"] <= 12):
        raise ValueError("month must be between 1 and 12.")

    if not (0 <= input_data["day_of_week"] <= 6):
        raise ValueError("day_of_week must be between 0 (Monday) and 6 (Sunday).")

    if not (1 <= input_data["quarter"] <= 4):
        raise ValueError("quarter must be between 1 and 4.")

    valid_categories = ["Electronics", "Clothing", "Groceries", "Furniture", "Toys"]
    if input_data["category"] not in valid_categories:
        raise ValueError(f"category must be one of: {valid_categories}")

    for flag in ["promotion", "holiday", "weekend"]:
        if input_data[flag] not in (0, 1):
            raise ValueError(f"'{flag}' must be 0 or 1.")


# ── Input Preparation ─────────────────────────────────────────────────────────

def prepare_input(input_data: dict) -> pd.DataFrame:
    """
    Convert a raw input dict into a single-row DataFrame
    matching the model's expected feature columns.

    Args:
        input_data: Validated dict of feature name → value.

    Returns:
        Single-row DataFrame ready for model.predict().
    """
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return pd.DataFrame([input_data])[feature_cols]


# ── Confidence Interval ───────────────────────────────────────────────────────

def compute_confidence_interval(
    prediction: float,
    margin: float = 0.10
) -> tuple[float, float]:
    """
    Compute a simple ±margin% confidence interval around a prediction.

    Args:
        prediction: The point prediction value.
        margin: Fractional margin (default 10%).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    lower = prediction * (1 - margin)
    upper = prediction * (1 + margin)
    return round(lower, 2), round(upper, 2)


# ── Prediction Entry Point ────────────────────────────────────────────────────

def predict(input_data: dict) -> dict:
    """
    Full prediction pipeline: validate → prepare → predict → interval.

    Args:
        input_data: Raw input dict from user or web form.

    Returns:
        Dict with prediction, lower_bound, and upper_bound.

    Raises:
        ValueError: If input validation fails.
    """
    validate_input(input_data)

    model = load_best_model()
    df_input = prepare_input(input_data)

    prediction = round(float(model.predict(df_input)[0]), 2)
    lower, upper = compute_confidence_interval(prediction)

    return {
        "prediction": prediction,
        "lower_bound": lower,
        "upper_bound": upper,
    }


if __name__ == "__main__":
    # Quick smoke test with sample input
    sample = {
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

    result = predict(sample)
    print(f"\n🎯 Prediction    : {result['prediction']:,.2f}")
    print(f"   Lower Bound  : {result['lower_bound']:,.2f}")
    print(f"   Upper Bound  : {result['upper_bound']:,.2f}")