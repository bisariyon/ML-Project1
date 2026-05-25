import os
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        best_models = {}

        logging.info("Starting model training")
        for model_name, model in models.items():
            logging.info(f"Training started for {model_name}")
            param = params[model_name]

            grid_search = GridSearchCV(
                model,
                param,
                cv=3,
                verbose=2,
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            # y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            test_model_score = r2_score(y_test,y_test_pred)

            report[model_name] = test_model_score
            best_models[model_name] = best_model

            logging.info(f"Training completed for {model_name}")

        logging.info("Starting model ended")
        return report, best_models

    except Exception as e:
        raise CustomException(e, sys)
