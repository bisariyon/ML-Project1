from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.logger import logging
from src.exception import CustomException

import sys

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline")

            # ======================
            # DATA INGESTION
            # ======================

            data_ingestion = DataIngestion()

            train_data_path, test_data_path = (
                data_ingestion.initiate_data_ingestion()
            )

            logging.info("Data ingestion completed")


            # ======================
            # DATA TRANSFORMATION
            # ======================

            data_transformation = DataTransformation()

            train_arr, test_arr, _ = (
                data_transformation.inititate_data_transformation(
                    train_data_path,
                    test_data_path
                )
            )

            logging.info("Data transformation completed")


            # ======================
            # MODEL TRAINING
            # ======================

            model_trainer = ModelTrainer()

            r2, best_model_name, best_model, trained_models, model_report = (
                model_trainer.initiate_model_trainer(
                    train_arr,
                    test_arr
                )
            )

            logging.info("Model training completed")

            print("=" * 50)
            print(f"Best Model Name : {best_model_name}")
            print(f"\nBest Model Object :\n{best_model}")
            print(f"\nFinal R2 Score : {r2}")
            print(f"\nTrained models: {trained_models}")
            print(f"\nModels report: {model_report}")
            print("=" * 50)

            return r2

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = TrainPipeline()
    obj.run_pipeline()

