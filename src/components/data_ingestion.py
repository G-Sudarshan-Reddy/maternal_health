import os, sys
import pandas as pd 
import numpy as np 

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import modeltrainer, modeltrainerconfig

from dataclasses import dataclass

from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_path = os.path.join('artifacts', 'train.csv')
    test_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion process started")
            df = pd.read_csv("Maternal Health Risk Data Set.csv")

            os.makedirs(os.path.dirname(self.ingestion_config.train_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initated")
            train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)

            train_data.to_csv(self.ingestion_config.train_path, index = False, header = True)
            test_data.to_csv(self.ingestion_config.test_path, index = False, header = True)

            logging.info("Train test split completed succesfully")

            return(
                self.ingestion_config.train_path,
                self.ingestion_config.test_path
            )


        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation= DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    model = modeltrainer()
    print(model.initiate_model_trainer(train_arr,test_arr))