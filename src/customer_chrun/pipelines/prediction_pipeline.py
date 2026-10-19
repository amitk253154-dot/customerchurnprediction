import os
import sys

import pandas as pd

from src.customer_chrun.components.data_transformation import DataTransformation
from src.customer_chrun.exception import CustomException
from src.customer_chrun.logger import logging
from src.customer_chrun.utils import load_object

# Raw input columns the model expects (customerID is optional and ignored)
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]


class PredictPipeline:
    def __init__(self, model_path=os.path.join("artifacts", "model.pkl")):
        artifact = load_object(model_path)          # {"model": Pipeline, "threshold": float}
        self.model = artifact["model"]
        self.threshold = artifact["threshold"]
        self.cleaner = DataTransformation()

    def predict(self, features: pd.DataFrame):
        """Return (churn risk score, churn flag) for each row of raw customer data."""
        try:
            missing = [c for c in FEATURE_COLUMNS if c not in features.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            data = self.cleaner.clean_data(features[FEATURE_COLUMNS])   # same cleaning as training
            scores = self.model.predict_proba(data)[:, 1]        # preprocessing happens inside the pipeline
            flags = (scores >= self.threshold).astype(int)
            logging.info(f"Predicted {len(data)} row(s), threshold {self.threshold:.3f}")
            return scores, flags
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """Holds the raw inputs for one customer and turns them into a one-row DataFrame."""

    def __init__(self, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
                 MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
                 DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,
                 PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges):
        self.gender = gender
        self.SeniorCitizen = int(SeniorCitizen)
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = int(tenure)
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = float(MonthlyCharges)
        # If TotalCharges is not known, approximate it as tenure x monthly charges
        self.TotalCharges = float(TotalCharges) if str(TotalCharges).strip() else self.tenure * self.MonthlyCharges

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame([vars(self)])
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Quick test with real rows from the saved test split
    sample = pd.read_csv(os.path.join("artifacts", "test_data.csv"))
    rows = pd.concat([sample[sample.Churn == "Yes"].head(3), sample[sample.Churn == "No"].head(3)])

    scores, flags = PredictPipeline().predict(rows.drop(columns=["Churn"]))
    for actual, flag, score in zip(rows["Churn"], flags, scores):
        print(f"  Actual: {actual:<3} | Predicted: {flag} | Churn risk score: {score:.4f}")

