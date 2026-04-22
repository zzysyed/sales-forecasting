"""
web_app.py
----------
Flask web application providing a simple interface
for generating sales predictions from the trained model.

Responsibilities:
    - Serve the prediction form (GET /)
    - Accept form submissions and return predictions (POST /)
    - Handle and display input validation errors gracefully
"""

from flask import Flask, render_template, request
from src.model_inference import predict

app = Flask(__name__, template_folder="../templates")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    """
    GET  : Render the empty prediction form.
    POST : Process form input, run prediction, render result.
    """
    prediction_result = None
    error_message = None

    if request.method == "POST":
        try:
            input_data = _parse_form(request.form)
            prediction_result = predict(input_data)
        except ValueError as e:
            error_message = str(e)

    return render_template(
        "index.html",
        result=prediction_result,
        error=error_message,
    )


# ── Form Parser ───────────────────────────────────────────────────────────────

def _parse_form(form) -> dict:
    """
    Parse and type-cast raw HTML form data into the dict
    format expected by model_inference.predict().

    Args:
        form: Flask request.form ImmutableMultiDict.

    Returns:
        Dict of feature name → correctly typed value.

    Raises:
        ValueError: If any field cannot be cast to its expected type.
    """
    try:
        return {
            "store_id":   int(form["store_id"]),
            "promotion":  int(form["promotion"]),
            "holiday":    int(form["holiday"]),
            "weekend":    int(form["weekend"]),
            "month":      int(form["month"]),
            "day_of_week": int(form["day_of_week"]),
            "quarter":    int(form["quarter"]),
            "lag_7":      float(form["lag_7"]),
            "lag_30":     float(form["lag_30"]),
            "rolling_7":  float(form["rolling_7"]),
            "rolling_30": float(form["rolling_30"]),
            "category":   form["category"],
        }
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid form input: {e}")


if __name__ == "__main__":
    app.run(debug=True)