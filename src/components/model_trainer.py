"""
1. Create model trainer config
    - Path where model will be stored

2. Create model trainer
    - init function
    - initiate model trainer function
        - Create X_train y_train X_test y_test
        - Create model dictionary with all model names and models
        - Create params for hyperparameter tuning
        - Pass all data to a fucntion that will train, test and return score metrics
        - Find the best model
        - Save this best model
        - Predict using this model and return the score
"""

from dataclasses import dataclass

import os
import sys

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig():
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr, test_arr):
        try:
            logging.info("Entered model training class")

            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            logging.info(f"Created the X_train, y_train, X_test, y_test")

            models = {
                'Linear Regression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'KNN': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'XGBoost' : XGBRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'CatBoost':CatBoostRegressor(verbose=False)
            }
            
            params = {
                'Linear Regression' : {},
                'Lasso':{
                    'alpha' : [0.1, 0.01, 1 ,10, 100],
                    'selection' : ['cyclic','random'],
                },
                'Ridge': {
                    'alpha' : [0.1, 0.01, 1 ,10, 100],
                    'solver': ['auto','sag','saga']
                },
                'KNN': {
                    'n_neighbors': list(range(1, 11)),
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'p': [1, 2]
                },
                'Decision Tree': {
                    'criterion' : ['squared_error', 'friedman_mse', 'absolute_error'],
                    'splitter':['best','random'],
                    'max_depth':[1,2,3,4,5,10,15,20,25],
                    'max_features':['sqrt','log2']
                },
                'Random Forest' : {
                    'max_depth' : [None, 5, 10, 15, 20],
                    'max_features' : ['sqrt', 'log2'],
                    'min_samples_split' : [2, 5, 10, 20],
                    # 'n_estimators' : [100, 200, 300, 400, 500, 1000],
                    'n_estimators' : [5,10],
                },
                'XGBoost' : {
                    'max_depth' : [5,8,12,20,30],
                    'learning_rate' : [0.1],
                    'n_estimators' : [100,200],
                    'colsample_bytree' : [0.5,0.8,1,0.3,0.4]
                },
                'AdaBoost': {
                    'n_estimators': [50, 100, 150, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3, 1],
                },
                'CatBoost':{
                    'depth': [6,8],
                    'learning_rate': [0.01, 0.1],
                    'iterations': [100, 200],
                },
            }

            logging.info("Now sending data for model fititng and evaluation")
            model_report, trained_models = evaluate_model(X_train, y_train, X_test, y_test, models, params)

            best_model_name = max(model_report, key=model_report.get)
            best_model = trained_models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            logging.info("Model training class fininshed")
            r2 = r2_score(y_test, predicted)
            
            return (
                r2,
                best_model_name,
                best_model,
                trained_models,
                model_report
            )

        except Exception as e:
            raise CustomException(e,sys)  
          