"""
data_preprocessing.py
---------------------
Handles all data loading, cleaning, and feature engineering
for the sales forecasting pipeline.

Responsibilities:
    - Load raw CSV data
    - Validate and clean the dataset
    - Engineer time-based and lag features
    - Split into train/test sets
    - Save processed data to disk
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    DATE_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TEST_SIZE,
    RANDOM_SEED,
)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    """
    Load the raw sales CSV from disk.

    Returns:
        DataFrame with raw sales records.

    Raises:
        FileNotFoundError: If the CSV does not exist at the configured path.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data not found at '{RAW_DATA_PATH}'. "
            "Run data/generate_data.py first."
        )

    df = pd.read_csv(RAW_DATA_PATH, parse_dates=[DATE_COLUMN])
    print(f"✅ Loaded {len(df):,} rows from '{RAW_DATA_PATH}'")
    return df


# ── Data Validation ───────────────────────────────────────────────────────────

def validate_data(df: pd.DataFrame) -> None:
    """
    Check for missing values and negative sales.
    Raises ValueError if critical issues are found.

    Args:
        df: Raw sales DataFrame to validate.

    Raises:
        ValueError: If nulls or negative sales values are detected.
    """
    null_counts = df.isnull().sum()
    if null_counts.any():
        raise ValueError(f"Dataset contains null values:\n{null_counts[null_counts > 0]}")

    if (df[TARGET_COLUMN] < 0).any():
        raise ValueError("Dataset contains negative sales values.")

    print("✅ Data validation passed.")


# ── Feature Engineering ───────────────────────────────────────────────────────

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar features from the date column.

    Adds: month, day_of_week, quarter

    Args:
        df: DataFrame with a parsed date column.

    Returns:
        DataFrame with new calendar feature columns.
    """
    df = df.copy()
    df["month"] = df[DATE_COLUMN].dt.month
    df["day_of_week"] = df[DATE_COLUMN].dt.dayofweek
    df["quarter"] = df[DATE_COLUMN].dt.quarter
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features: sales values from 7 and 30 days prior.
    Grouped by store_id and category to avoid cross-contamination.

    Adds: lag_7, lag_30

    Args:
        df: DataFrame sorted by date with a sales column.

    Returns:
        DataFrame with lag feature columns added.
    """
    df = df.copy()
    group_keys = ["store_id", "category"]

    df["lag_7"] = (
        df.groupby(group_keys)[TARGET_COLUMN]
        .shift(7)
    )
    df["lag_30"] = (
        df.groupby(group_keys)[TARGET_COLUMN]
        .shift(30)
    )
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling average features over 7 and 30 day windows.
    Grouped by store_id and category.

    Adds: rolling_7, rolling_30

    Args:
        df: DataFrame with lag features already added.

    Returns:
        DataFrame with rolling average columns added.
    """
    df = df.copy()
    group_keys = ["store_id", "category"]

    df["rolling_7"] = (
        df.groupby(group_keys)[TARGET_COLUMN]
        .transform(lambda x: x.shift(1).rolling(window=7).mean())
    )
    df["rolling_30"] = (
        df.groupby(group_keys)[TARGET_COLUMN]
        .transform(lambda x: x.shift(1).rolling(window=30).mean())
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering sequence.

    Steps:
        1. Add calendar features (month, day_of_week, quarter)
        2. Add lag features (lag_7, lag_30)
        3. Add rolling average features (rolling_7, rolling_30)
        4. Drop rows with NaN from lag/rolling calculations

    Args:
        df: Cleaned raw DataFrame.

    Returns:
        Feature-engineered DataFrame ready for modelling.
    """
    df = add_date_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    original_len = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = original_len - len(df)
    print(f"✅ Feature engineering complete. Dropped {dropped:,} rows with NaN (from lag/rolling windows).")
    return df


# ── Train/Test Split ──────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the processed DataFrame into train and test sets.

    Args:
        df: Fully processed DataFrame with all features.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    print(f"✅ Train/test split done — Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ── Pipeline Entry Point ──────────────────────────────────────────────────────

def run_preprocessing() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Execute the full preprocessing pipeline end to end.

    Steps:
        1. Load raw data
        2. Validate data
        3. Engineer features
        4. Save processed data
        5. Split into train/test

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    df = load_raw_data()
    validate_data(df)
    df = engineer_features(df)

    df.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8")
    print(f"✅ Processed data saved to '{PROCESSED_DATA_PATH}'")

    return split_data(df)


if __name__ == "__main__":
    run_preprocessing()