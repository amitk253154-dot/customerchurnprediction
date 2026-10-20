import os

import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, send_file, url_for

from src.customer_chrun.logger import logging
from src.customer_chrun.pipelines.prediction_pipeline import (
    FEATURE_COLUMNS,
    CustomData,
    PredictPipeline,
)

application = Flask(__name__)
app = application
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024      # reject uploads larger than 5 MB

MODEL_MISSING_MSG = ("Model not found. Run "
                     "python -m src.customer_chrun.pipelines.train_pipeline first.")
try:
    predictor = PredictPipeline()                        # model is loaded once, at startup
except Exception as e:
    logging.error(f"Could not load the model: {e}")
    predictor = None


@app.context_processor
def inject_model_status():
    return {"model_ready": predictor is not None}


SAMPLE_PATH = os.path.join("artifacts", "test_data.csv")
OUTPUT_PATH = os.path.join("artifacts", "predictions.csv")
MAX_ROWS_SHOWN = 100

YES_NO = ["No", "Yes"]
ADDON = ["No", "Yes", "No internet service"]

# field name -> (label, options). Options are plain values or (value, text) pairs.
CHOICES = {
    "gender": ("Gender", ["Male", "Female"]),
    "SeniorCitizen": ("Senior citizen", [("0", "No"), ("1", "Yes")]),
    "Partner": ("Partner", YES_NO),
    "Dependents": ("Dependents", YES_NO),
    "PhoneService": ("Phone service", ["Yes", "No"]),
    "MultipleLines": ("Multiple lines", ["No", "Yes", "No phone service"]),
    "InternetService": ("Internet service", ["Fiber optic", "DSL", "No"]),
    "OnlineSecurity": ("Online security", ADDON),
    "OnlineBackup": ("Online backup", ADDON),
    "DeviceProtection": ("Device protection", ADDON),
    "TechSupport": ("Tech support", ADDON),
    "StreamingTV": ("Streaming TV", ADDON),
    "StreamingMovies": ("Streaming movies", ADDON),
    "Contract": ("Contract", ["Month-to-month", "One year", "Two year"]),
    "PaperlessBilling": ("Paperless billing", ["Yes", "No"]),
    "PaymentMethod": ("Payment method", ["Electronic check", "Mailed check",
                                         "Bank transfer (automatic)", "Credit card (automatic)"]),
}
FIELDS = list(CHOICES) + ["tenure", "MonthlyCharges", "TotalCharges"]


def build_batch_result(df: pd.DataFrame):
    """Score every customer in df and return a summary plus a table sorted by risk."""
    scores, flags = predictor.predict(df)

    table = pd.DataFrame({
        "Customer": df["customerID"].values if "customerID" in df.columns else np.arange(1, len(df) + 1),
        "Contract": df["Contract"].values,
        "Tenure": df["tenure"].values,
        "Monthly charges": df["MonthlyCharges"].values,
        "Risk score (%)": (scores.astype(float) * 100).round(1),
        "Prediction": np.where(flags == 1, "Likely to churn", "Unlikely to churn"),
    })
    if "Churn" in df.columns:
        table["Actual"] = df["Churn"].values

    table = table.sort_values("Risk score (%)", ascending=False).reset_index(drop=True)
    table.to_csv(OUTPUT_PATH, index=False)               # full results, for the download button

    flagged = int((table["Prediction"] == "Likely to churn").sum())
    summary = {"total": len(table), "flagged": flagged,
               "flagged_pct": round(flagged / len(table) * 100, 1),
               "shown": min(len(table), MAX_ROWS_SHOWN)}
    if "Actual" in table.columns:
        actual = table["Actual"] == "Yes"
        summary["actual_churners"] = int(actual.sum())
        summary["caught"] = int((actual & (table["Prediction"] == "Likely to churn")).sum())

    return summary, list(table.columns), table.head(MAX_ROWS_SHOWN).values.tolist()


@app.route("/", methods=["GET", "POST"])
def index():
    """Predict churn for one customer."""
    result, error, form = None, None, {}
    if request.method == "POST":
        form = {f: request.form.get(f, "") for f in FIELDS}
        try:
            if predictor is None:
                raise RuntimeError(MODEL_MISSING_MSG)
            data = CustomData(**form).get_data_as_data_frame()
            scores, flags = predictor.predict(data)
            result = {"score": round(float(scores[0]) * 100, 1), "churn": bool(flags[0])}
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            error = MODEL_MISSING_MSG if predictor is None else "Could not make a prediction. Please check the values you entered."
    return render_template("home.html", choices=CHOICES, result=result, error=error, form=form)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    """Predict churn for many customers from a CSV file."""
    summary, columns, rows, error = None, None, None, None
    if request.method == "POST":
        try:
            if predictor is None:
                raise ValueError(MODEL_MISSING_MSG)
            mode = request.form.get("mode", "upload")
            if mode in ("sample", "churner", "stayed"):
                if not os.path.exists(SAMPLE_PATH):
                    raise ValueError("Sample file not found. Run the training pipeline first.")
                df = pd.read_csv(SAMPLE_PATH)
                if mode != "sample":                     # one random real customer
                    wanted = "Yes" if mode == "churner" else "No"
                    df = df[df["Churn"] == wanted].sample(1)
            else:
                file = request.files.get("file")
                if file is None or file.filename == "":
                    raise ValueError("Please choose a CSV file.")
                df = pd.read_csv(file)

            missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(f"The file is missing these columns: {', '.join(missing)}")

            summary, columns, rows = build_batch_result(df)
        except ValueError as e:
            error = str(e)
        except Exception as e:
            logging.error(f"Batch prediction failed: {e}")
            error = "Could not read or score this file. Use the same columns as the Telco dataset."
    return render_template("batch.html", summary=summary, columns=columns, rows=rows, error=error)


@app.route("/download")
def download():
    if not os.path.exists(OUTPUT_PATH):
        return redirect(url_for("batch"))
    return send_file(OUTPUT_PATH, as_attachment=True, download_name="churn_predictions.csv")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)