import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.customer_chrun.logger import logging
from src.customer_chrun.exception import CustomException


@dataclass
class DataIngestionConfig:
    source_data_path: str = os.path.join("notebook", "data", "WA_Fn-UseC_-Telco-Customer-Churn (1).csv")
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    test_data_path: str = os.path.join("artifacts", "test_data.csv")
    target_column: str = "Churn"
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        cfg = self.ingestion_config
        logging.info("Data ingestion started")
        try:
            data = pd.read_csv(cfg.source_data_path)
            logging.info(f"Read the data from the csv file, shape: {data.shape}")

            if data.empty or cfg.target_column not in data.columns:
                raise ValueError("Dataset is empty or the target column is missing")
            if data[cfg.target_column].isna().any():
                raise ValueError("Target column has missing values, cannot stratify")

            os.makedirs(os.path.dirname(cfg.raw_data_path), exist_ok=True)
            data.to_csv(cfg.raw_data_path, index=False)
            logging.info("Created the raw data file")

            logging.info("Splitting the data into train and test")
            train_data, test_data = train_test_split(
                data,
                test_size=cfg.test_size,
                random_state=cfg.random_state,
                stratify=data[cfg.target_column],
            )
            train_data.to_csv(cfg.train_data_path, index=False)
            test_data.to_csv(cfg.test_data_path, index=False)
            logging.info(f"Data ingestion completed. Train: {train_data.shape}, Test: {test_data.shape}")

            return cfg.train_data_path, cfg.test_data_path

        except Exception as e:
            logging.error("Exception occurred while ingesting the data")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print(train_path, test_path)