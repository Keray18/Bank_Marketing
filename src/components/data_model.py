import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from dataclasses import dataclass
from src.utils import save_object
from sklearn.model_selection import train_test_split
import xgboost as xgb


@dataclass
class DataModelConfig:
    trained_model_file: str = os.path.join('artifact', 'model.pkl')


class DataModel:
    def __init__(self):
        self.data_model_config = DataModelConfig()

    def initiate_data_model(self, new_data_path):
        try:
            logging.info("Splitting the dataset....")
            df = pd.read_csv(new_data_path)
            x = df.drop("deposit", axis=1)
            y = df['deposit']
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42)
            logging.info("Splitting Completed! Fitting the model....")

            model = xgb.XGBClassifier(
                objective='binary:logistic', random_state=42)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            if (accuracy < 0.6):
                raise CustomException("No Best Model Found.")
            logging.info("Model has been created successfully")
            print(f"Model is {accuracy}% accurate.")

            save_object(
                file_path=self.data_model_config.trained_model_file,
                obj=model
            )

        except Exception as e:
            raise CustomException(e, sys)
