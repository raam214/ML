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

        print(f"[DEBUG] Saving object to: {file_path}")  # Add this line

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        print("[DEBUG] Object saved successfully.")  # Add this line

    except Exception as e:
        print("[ERROR] Failed to save object:", e)  # Add this line
        raise CustomException(e, sys)
