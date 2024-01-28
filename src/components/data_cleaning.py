import os
import sys
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
import pandas as pd


@dataclass
class DataCleaningConfig:
    cleaned_data_path: str = os.path.join('artifact', "cleaned.csv")


class DataCleaning:
    def __init__(self):
        self.data_cleaned_config = DataCleaningConfig()

    def initiate_data_cleaning(self, new_data_path):
        try:
            df = pd.read_csv(new_data_path)
            logging.info("Data cleaning has been initiated....")

            df = df[df['balance'] > 0]
            df['pdays'] = df['pdays'].replace(-1, 0)
            df['job'] = df['job'].str.replace(".", "")
            df = df[df['job'] != 'unknown']
            df = df[df['education'] != 'unknown']
            logging.info(
                "Data has been cleaned. Beginning data preprocessing....")

            non_numerical_columns = df.select_dtypes(
                exclude=['int64', 'float64'])

            le = LabelEncoder()
            for col in non_numerical_columns:
                df[col] = le.fit_transform(df[col])

            df.to_csv(self.data_cleaned_config.cleaned_data_path,
                      index=False, header=True)
            logging.info("Data Encoded! Saving the new dataset.")

            return (
                self.data_cleaned_config.cleaned_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
