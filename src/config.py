"""
config.py
---------
Central configuration for the sales forecasting pipeline.
All file paths, model parameters, and feature definitions live here.
No hardcoded values should appear in any other src/ file.
"""

from pathlib import Path

# ── Project Paths ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "sales_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_sales.csv"

# ── Data Settings ─────────────────────────────────────────────────────────────
TARGET_COLUMN = "sales"
DATE_COLUMN = "date"
TEST_SIZE = 0.2
RANDOM_SEED = 42

# ── Feature Definitions ───────────────────────────────────────────────────────
# Features the model will train on (generated during preprocessing)
NUMERIC_FEATURES = [
    "store_id",
    "promotion",
    "holiday",
    "weekend",
    "month",
    "day_of_week",
    "quarter",
    "lag_7",        # sales 7 days ago
    "lag_30",       # sales 30 days ago
    "rolling_7",    # 7-day average sales
    "rolling_30",   # 30-day average sales
]

CATEGORICAL_FEATURES = ["category"]

# ── Model Parameters ──────────────────────────────────────────────────────────
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "random_state": RANDOM_SEED,
}

LINEAR_REGRESSION_PARAMS = {}

# ── Model Persistence ─────────────────────────────────────────────────────────
MODEL_FILENAMES = {
    "LinearRegression": MODELS_DIR / "linear_regression.pkl",
    "RandomForest": MODELS_DIR / "random_forest.pkl",
    "GradientBoosting": MODELS_DIR / "gradient_boosting.pkl",
}
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"