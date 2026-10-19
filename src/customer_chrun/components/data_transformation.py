import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.customer_chrun.exception import CustomException
from src.customer_chrun.logger import logging


@dataclass
class DataTransformationConfig:
    target_column: str = "Churn"
    id_column: str = "customerID"


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Row-level cleaning from the notebook. No statistics are learned here, so it is safe on train and test."""
        cfg = self.transformation_config
        df = df.copy()

        # 1. Remove the unique identifier
        df.drop(columns=[cfg.id_column], errors="ignore", inplace=True)

        # 2. Blank TotalCharges strings -> NaN -> numeric
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce"
            )

        # 3. Keep only valid target labels
        if cfg.target_column in df.columns:
            df[cfg.target_column] = df[cfg.target_column].astype("string").str.strip()
            df = df[df[cfg.target_column].isin(["Yes", "No"])].copy()
        return df

    def get_data_transformer_object(self, numeric_features, categorical_features):
        """Same (UNFITTED) preprocessing as the notebook. It is fitted later, inside each model pipeline."""
        try:
            numeric_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            categorical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ])
            logging.info(f"Numeric features: {numeric_features}")
            logging.info(f"Categorical features: {categorical_features}")

            return ColumnTransformer(transformers=[
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, categorical_features),
            ])
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """Clean the data and return features, 0/1 targets and an UNFITTED preprocessor.
        Fitting happens inside the model pipelines (per CV fold), so nothing leaks."""
        cfg = self.transformation_config
        try:
            train_df = self.clean_data(pd.read_csv(train_path))
            test_df = self.clean_data(pd.read_csv(test_path))
            logging.info(f"Read and cleaned train {train_df.shape} and test {test_df.shape}")

            X_train = train_df.drop(columns=[cfg.target_column])
            y_train = train_df[cfg.target_column].map({"No": 0, "Yes": 1}).astype(int)
            X_test = test_df.drop(columns=[cfg.target_column])
            y_test = test_df[cfg.target_column].map({"No": 0, "Yes": 1}).astype(int)

            numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()
            preprocessor = self.get_data_transformer_object(numeric_features, categorical_features)

            return X_train, y_train, X_test, y_test, preprocessor

        except Exception as e:
            logging.error("Exception occurred in data transformation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.customer_chrun.components.data_ingestion import DataIngestion

    train_path, test_path = DataIngestion().initiate_data_ingestion()
    X_train, y_train, X_test, y_test, _ = DataTransformation().initiate_data_transformation(train_path, test_path)
    print(X_train.shape, X_test.shape, y_train.mean().round(4), y_test.mean().round(4))