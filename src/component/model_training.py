import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from src.exception import Custom_exception_handling
from src.logger import logging
from src.utils import save_object, load_object
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray, feature_columns: list, target_column: str):
        try:
            logging.info("Starting model training")

            # Validate the number of columns in train_array and test_array
            num_features = len(feature_columns)
            num_columns = num_features + 1  # Number of feature columns plus target column
            
            if num_columns != train_array.shape[1]:
                raise ValueError(f"Number of columns in train_array ({train_array.shape[1]}) does not match feature_columns + 1 ({num_columns}) for target column")

            # Split the data into features and target
            X_train = train_array[:, :-1]  # All columns except the last one
            y_train = train_array[:, -1]   # Last column
            X_test = test_array[:, :-1]    # All columns except the last one
            y_test = test_array[:, -1]     # Last column

            # Print shapes and first few rows of data for verification
            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}")
            logging.info(f"y_test shape: {y_test.shape}")

            # Load the preprocessor
            preprocessor = load_object(self.model_trainer_config.preprocessor_obj_file_path)

            # Apply preprocessing to the data
            X_train_transformed = preprocessor.transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Initialize and train the RandomForestRegressor model
            model = RandomForestRegressor(
                max_depth=30, 
                min_samples_leaf=1, 
                min_samples_split=2, 
                n_estimators=300
            )
            model.fit(X_train_transformed, y_train)

            # Predict on the test set
            y_test_pred = model.predict(X_test_transformed)

            # Calculate R-squared score
            test_model_score = r2_score(y_test, y_test_pred)

            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            # Convert R-squared score to percentage
            accuracy = test_model_score * 100
            logging.info(f"Model accuracy: {accuracy:.2f}%")

            return accuracy

        except ValueError as ve:
            logging.error(f"ValueError occurred: {ve}")
            raise Custom_exception_handling(ve, sys)
        except Exception as e:
            logging.error(f"Exception occurred: {e}")
            raise Custom_exception_handling(e, sys)



        