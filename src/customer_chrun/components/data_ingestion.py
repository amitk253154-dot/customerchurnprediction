import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.customer_chrun.logger import logging
from src.customer_chrun.exception import CustomException


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")
    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    test_data_path: str = os.path.join("artifacts", "test_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            data = pd.read_csv(os.path.join("notebook", "data", "WA_Fn-UseC_-Telco-Customer-Churn (1).csv"))
            logging.info(f"Read the data from the csv file, shape: {data.shape}")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Created the raw data file")

            logging.info("Splitting the data into train and test")
            train_data, test_data = train_test_split(
                data, test_size=0.2, random_state=42, stratify=data["Churn"]
            )
            logging.info("Data splitting is done")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Created the train and test data files")
            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.error("Exception occurred while ingesting the data")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print(train_path, test_path)