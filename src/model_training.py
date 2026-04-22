"""
model_training.py
-----------------
Handles training, evaluation, and persistence of all three
forecasting models. Compares performance and saves the best model.

Responsibilities:
    - Build sklearn pipelines for each algorithm
    - Train and evaluate all three models
    - Save all models and flag the best one
    - Print a comparison report
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score

from src.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    RANDOM_FOREST_PARAMS,
    GRADIENT_BOOSTING_PARAMS,
    LINEAR_REGRESSION_PARAMS,
    MODEL_FILENAMES,
    BEST_MODEL_PATH,
    MODELS_DIR,
)


# ── Preprocessor ─────────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales numeric features
    and one-hot encodes categorical features.

    Returns:
        Configured but unfitted ColumnTransformer.
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


# ── Model Definitions ─────────────────────────────────────────────────────────

def get_model_definitions() -> dict:
    """
    Return a dict of model name → unfitted sklearn estimator.
    Parameters are pulled from config.py.

    Returns:
        Dict mapping model name to estimator instance.
    """
    return {
        "LinearRegression": LinearRegression(**LINEAR_REGRESSION_PARAMS),
        "RandomForest": RandomForestRegressor(**RANDOM_FOREST_PARAMS),
        "GradientBoosting": GradientBoostingRegressor(**GRADIENT_BOOSTING_PARAMS),
    }


# ── Pipeline Builder ──────────────────────────────────────────────────────────

def build_pipeline(estimator) -> Pipeline:
    """
    Wrap a preprocessor and estimator into a single sklearn Pipeline.

    Args:
        estimator: An unfitted sklearn estimator.

    Returns:
        Full sklearn Pipeline with preprocessing + model steps.
    """
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", estimator),
    ])


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate a fitted pipeline on the test set.

    Metrics: MAE, R², MAPE

    Args:
        pipeline: A fitted sklearn Pipeline.
        X_test: Test feature DataFrame.
        y_test: True sales values.

    Returns:
        Dict with mae, r2, and mape scores.
    """
    y_pred = pipeline.predict(X_test)

    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "mape": mean_absolute_percentage_error(y_test, y_pred) * 100,
    }


# ── Model Persistence ─────────────────────────────────────────────────────────

def save_model(pipeline: Pipeline, name: str) -> None:
    """
    Save a fitted pipeline to disk using pickle.

    Args:
        pipeline: Fitted sklearn Pipeline to save.
        name: Model name key matching MODEL_FILENAMES in config.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODEL_FILENAMES[name]
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"   💾 Saved → {path.name}")


# ── Training Orchestrator ─────────────────────────────────────────────────────

def train_all_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Train, evaluate, and save all three models.
    Identifies and saves the best model by R² score.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target values.
        y_test: Test target values.

    Returns:
        Dict mapping model name → metrics dict.
    """
    model_definitions = get_model_definitions()
    results = {}
    fitted_pipelines = {}

    print("\n" + "=" * 50)
    print("   MODEL TRAINING REPORT")
    print("=" * 50)

    for name, estimator in model_definitions.items():
        print(f"\n🔧 Training {name}...")

        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)
        results[name] = metrics
        fitted_pipelines[name] = pipeline

        save_model(pipeline, name)

        print(f"   MAE  : {metrics['mae']:,.2f}")
        print(f"   R²   : {metrics['r2']:.4f}")
        print(f"   MAPE : {metrics['mape']:.2f}%")

    # ── Pick and save best model by R² ────────────────────────────────────────
    best_name = max(results, key=lambda n: results[n]["r2"])
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(fitted_pipelines[best_name], f)

    print("\n" + "=" * 50)
    print(f"🏆 Best Model : {best_name}")
    print(f"   R²         : {results[best_name]['r2']:.4f}")
    print(f"   MAE        : {results[best_name]['mae']:,.2f}")
    print(f"   MAPE       : {results[best_name]['mape']:.2f}%")
    print("=" * 50)

    return results


if __name__ == "__main__":
    from src.data_preprocessing import run_preprocessing
    X_train, X_test, y_train, y_test = run_preprocessing()
    train_all_models(X_train, X_test, y_train, y_test)