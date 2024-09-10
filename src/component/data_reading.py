import os
import sys
import pandas as pd
from src.exception import Custom_exception_handling
from src.logger import logging
from dataclasses import dataclass

from src.component.data_tranformation import DataTransformation
from src.component.model_training import ModelTrainer



@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the dataset from the specified path
            df = pd.read_csv('dataset/magic04.data')
            logging.info('Read the dataset as dataframe')

            # Ensure the artifacts directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save the entire dataset to the artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise Custom_exception_handling(e, sys)



if __name__ == "__main__":
    obj = DataIngestion()
    raw_data_path = obj.initiate_data_ingestion()

    # Load the data into a DataFrame
    df = pd.read_csv(raw_data_path)

    # Perform data transformation
    data_transformation = DataTransformation()
    train_data, test_data = data_transformation.initiate_data_transformation(df)

    # Model training
    train_features = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']
    target_column = 'class'

    model_trainer = ModelTrainer()
    accuracy = model_trainer.initiate_model_trainer(train_data, test_data, train_features, target_column)
    print(f"Model accuracy: {accuracy:.2f}%")



