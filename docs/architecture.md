# 🏛️ Architecture Documentation

## Overview

The sales forecasting system follows a linear pipeline architecture where each
module has a single responsibility. Data flows in one direction — from raw CSV
through preprocessing, model training, inference, and finally the web interface.

---

## Module Responsibilities

### `data/generate_data.py`
Generates a realistic synthetic retail sales dataset with seasonality, promotions,
holidays, and weekend effects. Outputs `data/sales_data.csv`.

### `src/config.py`
Central configuration file. All file paths, model parameters, and feature
definitions live here. No other module hardcodes any values — they all import
from config.

### `src/data_preprocessing.py`
Loads raw data, validates it, engineers features, and splits into train/test sets.
Feature engineering includes calendar features, lag features, and rolling averages.

### `src/model_training.py`
Builds sklearn pipelines for all three models, trains and evaluates them, saves
each model to disk, and flags the best performer by R² score.

### `src/model_inference.py`
Loads the best saved model, validates incoming input, prepares it into the correct
format, generates a prediction, and returns a confidence interval.

### `src/web_app.py`
Flask application with a single route. Serves the prediction form on GET and
processes form submissions on POST. Passes results to the HTML template.

---

## Data Flow

generate_data.py
│
▼
sales_data.csv
│
▼
data_preprocessing.py
├── validate_data()
├── add_date_features()
├── add_lag_features()
├── add_rolling_features()
└── split_data()
│
▼
model_training.py
├── LinearRegression
├── RandomForestRegressor
└── GradientBoostingRegressor
│
▼
best_model.pkl
│
▼
model_inference.py
├── validate_input()
├── prepare_input()
└── predict()
│
▼
web_app.py → index.html

---

## Design Principles

**Single Responsibility**
Each file handles exactly one concern. Preprocessing does not know about models.
Models do not know about the web app. This makes each module independently
testable and reusable.

**Centralised Configuration**
All constants live in `config.py`. Changing a model parameter, file path, or
feature list requires editing exactly one file.

**No Data Leakage**
Train/test split happens after feature engineering but the lag and rolling
features are computed using `.shift()` so future values never leak into past
windows.

**Fail Fast**
Both `validate_data()` and `validate_input()` raise descriptive errors immediately
rather than letting bad data propagate silently through the pipeline.

