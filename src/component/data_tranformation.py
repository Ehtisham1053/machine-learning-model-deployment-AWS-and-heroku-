from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from src.utils import save_object
from src.exception import Custom_exception_handling
from dataclasses import dataclass
import numpy as np
import logging
import sys
import pandas as pd
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")
    train_data_path: str = os.path.join("artifacts", "train.npy")
    test_data_path: str = os.path.join("artifacts", "test.npy")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor(self):
        try:
            logging.info("Creating preprocessing pipeline")
            
            # Numerical pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),  # Fill missing values with the mean
                    ("scaler", StandardScaler())  # Feature scaling
                ]
            )

            # Define column transformer with feature columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", numerical_pipeline, slice(0, 10))  # Apply on all 10 feature columns
                ]
            )
            
            logging.info("Preprocessing pipeline created")
            return preprocessor

        except Exception as e:
            raise Custom_exception_handling(e, sys)

    def initiate_data_transformation(self, df: pd.DataFrame):
        try:
            logging.info("Starting data transformation")

            # Assign column names
            column_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
            if df.columns.tolist() != column_names:
                df.columns = column_names
                logging.info("Column names assigned to the dataset")

            # Replace 'g' with 0 and 'h' with 1 in the target column 'class'
            df['class'] = df['class'].replace({'g': 0, 'h': 1})

            # Splitting features and target
            X = df.drop('class', axis=1)
            y = df['class']

            # Splitting into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Applying oversampling to balance the data
            ros = RandomOverSampler()
            X_train, y_train = ros.fit_resample(X_train, y_train)

            logging.info("Data split into training and testing sets")

            # Temporarily bypassing the preprocessing pipeline
            preprocessor = self.get_preprocessor()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Use the raw features directly
            X_train_transformed = X_train
            X_test_transformed = X_test

            logging.info(f"X_train_transformed shape: {X_train_transformed.shape}")
            logging.info(f"X_test_transformed shape: {X_test_transformed.shape}")
            logging.info(f"y_train shape: {y_train.shape}")
            logging.info(f"y_test shape: {y_test.shape}")

            # Combine scaled features with targets for both train and test sets
            train_data = np.c_[X_train_transformed, np.array(y_train)]
            test_data = np.c_[X_test_transformed, np.array(y_test)]

            # Save preprocessor and datasets
            logging.info("Saving preprocessor object")
            save_object(
                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
                 obj=preprocessor
             )

            logging.info("Saving processed training and testing data")
            np.save(self.data_transformation_config.train_data_path, train_data)
            np.save(self.data_transformation_config.test_data_path, test_data)

            logging.info(f"train_data shape: {train_data.shape}")
            logging.info(f"test_data shape: {test_data.shape}")

            return train_data, test_data

        except Exception as e:
            raise Custom_exception_handling(e, sys)
