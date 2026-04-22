# 📈 Sales Forecasting — End-to-End ML Pipeline

A complete machine learning system that predicts daily retail sales based on
store, product, and date features. Built as part of a data engineering internship
project covering the full ML workflow from data generation to web deployment.

---

## 🏗️ Project Structure

sales-forecasting/
├── data/                        # Raw and processed datasets
├── src/
│   ├── config.py                # Central settings and constants
│   ├── data_preprocessing.py    # Cleaning and feature engineering
│   ├── model_training.py        # Train and compare 3 ML models
│   ├── model_inference.py       # Load model and generate predictions
│   └── web_app.py               # Flask web interface
├── models/                      # Saved model files (generated locally)
├── templates/
│   └── index.html               # Prediction form UI
├── tests/
│   ├── test_preprocessing.py    # Unit tests for preprocessing
│   └── test_model.py            # Unit tests for inference
├── requirements.txt
└── README.md

---

## ⚙️ Quick Start

### 1. Clone the repository
```powershell
git clone https://github.com/zzysyed/sales-forecasting.git
cd sales-forecasting
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Generate the dataset
```powershell
python data/generate_data.py
```

### 4. Run preprocessing
```powershell
python -m src.data_preprocessing
```

### 5. Train the models
```powershell
python -m src.model_training
```

### 6. Launch the web app
```powershell
python -m src.web_app
```

Then open your browser at `http://127.0.0.1:5000`

### 7. Run tests
```powershell
python -m pytest tests/ -v
```

---

## 🤖 Models Compared

| Model | MAE | R² | MAPE |
|---|---|---|---|
| Linear Regression | 49.40 | 0.9408 | 11.44% |
| Random Forest | 37.83 | 0.9663 | 8.13% |
| Gradient Boosting | 37.43 | 0.9678 | 8.01% |

✅ **Best Model: Gradient Boosting** — selected automatically and saved as `best_model.pkl`

---

## 🔧 Features Used

| Feature | Description |
|---|---|
| `store_id` | Store identifier (1–5) |
| `category` | Product category |
| `month` | Month of year (1–12) |
| `quarter` | Quarter of year (1–4) |
| `day_of_week` | Day of week (0=Mon, 6=Sun) |
| `promotion` | Promotion active (0 or 1) |
| `holiday` | Public holiday (0 or 1) |
| `weekend` | Weekend day (0 or 1) |
| `lag_7` | Sales 7 days prior |
| `lag_30` | Sales 30 days prior |
| `rolling_7` | 7-day rolling average sales |
| `rolling_30` | 30-day rolling average sales |

---

## 💡 Business Insights

- **Seasonality** is the strongest sales driver — December sales are 50% above baseline
- **Promotions** lift daily sales by approximately 30%
- **Weekends** add a 15% uplift across all categories
- **Gradient Boosting** outperforms Linear Regression by 30% on MAE

---

## 🏛️ Architecture

generate_data.py
↓
data_preprocessing.py  →  config.py (shared settings)
↓
model_training.py
↓
model_inference.py
↓
web_app.py  →  templates/index.html

Each module has a single responsibility — changes to one do not affect others.

---

## 🧪 Testing

11 unit tests across 2 test files covering:
- Data validation (null checks, negative value checks)
- Feature engineering correctness
- Train/test split integrity
- Input validation for predictions
- Confidence interval calculations
- Prediction output structure

