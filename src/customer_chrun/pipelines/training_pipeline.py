import sys

from src.customer_chrun.components.data_ingestion import DataIngestion
from src.customer_chrun.components.data_transformation import DataTransformation
from src.customer_chrun.components.model_tranier import ModelTrainer
from src.customer_chrun.exception import CustomException
from src.customer_chrun.logger import logging


class TrainPipeline:
    """Runs the full training flow: ingestion -> transformation -> model training."""

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            train_path, test_path = DataIngestion().initiate_data_ingestion()

            X_train, y_train, X_test, y_test, preprocessor = (
                DataTransformation().initiate_data_transformation(train_path, test_path)
            )

            metrics = ModelTrainer().initiate_model_trainer(
                X_train, y_train, X_test, y_test, preprocessor
            )

            logging.info("Training pipeline completed")
            return metrics

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    results = TrainPipeline().run_pipeline()
    for key, value in results.items():
        print(f"{key}: {value}")