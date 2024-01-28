from src.components.data_ingestion import DataIngestionConfig
from src.components.data_ingestion import DataIngestion

from src.components.data_cleaning import DataCleaningConfig
from src.components.data_cleaning import DataCleaning

from src.components.data_model import DataModelConfig
from src.components.data_model import DataModel


if __name__ == "__main__":
    obj = DataIngestion()
    raw_data = obj.initiate_data_ingestion()

    data_cleaning = DataCleaning()
    cleaned_data = data_cleaning.initiate_data_cleaning(raw_data)

    data_model = DataModel()
    data_model.initiate_data_model(cleaned_data)
