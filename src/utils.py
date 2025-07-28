import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        best_model = None
        best_score = -np.inf

        for name, model in models.items():
            gs = GridSearchCV(model, param[name], cv=3)
            gs.fit(X_train, y_train)

            model = gs.best_estimator_  # ✅ use best parameters

            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = test_model_score

            if test_model_score > best_score:
                best_score = test_model_score
                best_model = model

        return report, best_model  # ✅ return both

    except Exception as e:
        raise CustomException(e, sys)
