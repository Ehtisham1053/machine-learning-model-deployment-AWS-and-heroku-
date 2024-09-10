import os
import sys
import pickle
from sklearn.metrics import r2_score
from src.exception import Custom_exception_handling

# Function to save objects like models, preprocessors, etc.
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise Custom_exception_handling(e, sys)

# Function to load objects like models, preprocessors, etc.
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise Custom_exception_handling(e, sys)

# Function to evaluate the model with a given dataset
def evaluate_model(model, X_train, y_train, X_test, y_test):
    try:
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        test_model_score = r2_score(y_test, y_test_pred)

        return test_model_score

    except Exception as e:
        raise Custom_exception_handling(e, sys)
