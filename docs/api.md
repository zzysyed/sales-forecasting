# 🔌 API Documentation

## Web Interface

### `GET /`
Renders the sales prediction form.

**Response:** HTML page with input form.

---

### `POST /`
Accepts form data, runs prediction, returns result on the same page.

**Form Fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `store_id` | integer | ✅ | Store identifier (1–5) |
| `category` | string | ✅ | Product category |
| `month` | integer | ✅ | Month of year (1–12) |
| `quarter` | integer | ✅ | Quarter (1–4) |
| `day_of_week` | integer | ✅ | 0=Monday, 6=Sunday |
| `promotion` | integer | ✅ | 0 or 1 |
| `holiday` | integer | ✅ | 0 or 1 |
| `weekend` | integer | ✅ | 0 or 1 |
| `lag_7` | float | ✅ | Sales 7 days ago |
| `lag_30` | float | ✅ | Sales 30 days ago |
| `rolling_7` | float | ✅ | 7-day rolling average |
| `rolling_30` | float | ✅ | 30-day rolling average |

**Success Response:** Same page rendered with prediction result block showing:
- Predicted sales value
- Lower confidence bound (−10%)
- Upper confidence bound (+10%)

**Error Response:** Same page rendered with error message block.

---

## Python API

The inference module can also be used directly in Python:

```python
from src.model_inference import predict

result = predict({
    "store_id": 1,
    "category": "Electronics",
    "month": 12,
    "quarter": 4,
    "day_of_week": 5,
    "promotion": 1,
    "holiday": 0,
    "weekend": 1,
    "lag_7": 850.0,
    "lag_30": 800.0,
    "rolling_7": 820.0,
    "rolling_30": 810.0,
})

print(result["prediction"])     # e.g. 1155.90
print(result["lower_bound"])    # e.g. 1040.31
print(result["upper_bound"])    # e.g. 1271.49
```

**Returns:**

| Key | Type | Description |
|---|---|---|
| `prediction` | float | Predicted daily sales |
| `lower_bound` | float | Lower confidence bound |
| `upper_bound` | float | Upper confidence bound |

**Raises:**
- `ValueError` — if any input field is missing or out of range
- `FileNotFoundError` — if no trained model exists on disk

