from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.logger import logging


if __name__ == "__main__":

    try:
        logging.info("Starting complete training pipeline")

        # =========================
        # DATA INGESTION
        # =========================
        logging.info("Starting Data Ingestion")

        ingestion = DataIngestion()

        train_path, test_path = ingestion.initiate_data_ingestion()

        logging.info("Data Ingestion Completed")


        # =========================
        # DATA TRANSFORMATION
        # =========================
        logging.info("Starting Data Transformation")

        transformation = DataTransformation()

        train_arr, test_arr, _ = transformation.inititate_data_transformation(
            train_path,
            test_path
        )

        logging.info("Data Transformation Completed")


        # =========================
        # MODEL TRAINING
        # =========================
        logging.info("Starting Model Training")

        trainer = ModelTrainer()

        r2_score, best_model_name, best_model , trained_models,model_report = trainer.initiate_model_trainer(
            train_arr,
            test_arr
        )

        logging.info("Model Training Completed")


        # =========================
        # RESULTS
        # =========================
        print("\n")
        print("=" * 50)

        print(f"Best Model Name : {best_model_name}")

        print(f"\nBest Model Object :\n{best_model}")

        print(f"\nFinal R2 Score : {r2_score}")

        print(f"\nTrained models: {trained_models}")

        print(f"\nModels report: {model_report}")

        print("=" * 50)


    except Exception as e:

        logging.exception(e)

        print(e)