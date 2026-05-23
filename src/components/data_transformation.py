"""
1. Create Data transformation config class
    - It will contain 'preprocessor' path

2. Create Data tranformation class
    - init function 
    - get_data_transformer_object function
        - Get numerical and categorical features
        - Create pipeline for both numerical and categorical features transformation
        - Create column transformer (preprocessor) for preprocessing columns
        - Return preprocessor
    - inititate data transformation
        - Take train data and test data paths as input parameters
        - Read these csv files
        - Call get_data_transformer_object to get the preprocessor
        - Create X_train X_test y_train y_test from the train and test data
        - Apply preprocessor to X_train and X_test
        - Combine to get preprocessed train data and test data
        - Save the preprocessor object
        - Return train data and test data along with preprocessor path
"""


from dataclasses import dataclass
import os
import sys

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            logging.info(f"Numerical columns: {numerical_columns}")


            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            logging.info(f"Categorical columns: {categorical_columns}")


            logging.info("Creating numeric pipeline")
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )


            logging.info("Creating Categorical Pipeline")
            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("onehotencoder", OneHotEncoder())
                ]
            )


            logging.info("Creating column tranformer")
            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_pipeline",numerical_pipeline,numerical_columns),
                    ("categorical_pipeline",categorical_pipeline,categorical_columns)
                ],
                remainder='passthrough'
            )


            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def inititate_data_transformation(self,train_data_path,test_data_path):

        try:
            train_data_df = pd.read_csv(train_data_path)
            test_data_df = pd.read_csv(test_data_path)
            logging.info("Read train and test data sets")

            preprocessor = self.get_data_transformer_object()
            logging.info("Obtained the preprocessor object")

            target_column_name="math_score"

            X_train_df = train_data_df.drop(columns=target_column_name, axis=1)
            y_train_df = train_data_df[target_column_name]
            logging.info("Created X_train and y_train")
            
            X_test_df = test_data_df.drop(columns=target_column_name, axis=1)
            y_test_df = test_data_df[target_column_name]
            logging.info("Created X_test and y_test")


            logging.info("Applying preprocessor")
            X_train_arr = preprocessor.fit_transform(X_train_df)
            X_test_arr = preprocessor.transform(X_test_df)
            logging.info("Preprocessing completed")

            train_arr = np.column_stack(
                (
                    X_train_arr,
                    np.array(y_train_df)
                )
            )            
            logging.info("Combined X_train y_train to form training dataset")

            test_arr = np.c_[
                X_test_arr,
                np.array(y_test_df)
            ]
            logging.info("Combined X_test y_test to form testing dataset")


            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj =  preprocessor
            )
            logging.info("Saved preprocessing object")


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)

