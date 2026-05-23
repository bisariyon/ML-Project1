"""
1. Create data ingestion config class
    - This will store train_data_path, test_data_path, raw_data_path

2. Create data ingestion class
    - init function
    - initiate data ingestion function 
        - Read data from various sources
        - Rename columns if needed
        - Save raw data file (to csv)
        - Train test split
        - Save train and test data
        - return train and test data path

3. This will be the main file having __name__ == '__main__' so all calings will be done from here

"""

from dataclasses import dataclass
import logging
import os
import random
import sys

from numpy.random import rand
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src import logger
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv') 
    test_data_path:str = os.path.join('artifacts','test.csv') 
    raw_data_path:str = os.path.join('artifacts','raw.csv') 


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered data ingestion component')
        
        try:
            df = pd.read_csv('notebook\data\StudentsPerformance.csv')
            logging.info('Read data from csv')

            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace("/", "_")
            )
            logging.info('Renamed columns')


            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            logging.info('Created artifacts directory')
        

            df.to_csv(self.data_ingestion_config.raw_data_path,header=True,index=False)
            logging.info('Saved raw data to artifacts folder')


            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df,test_size=0.2, random_state=42)


            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            logging.info('Train set saved to artifacts')

            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            logging.info('Test set saved to artifacts')


            logging.info('Data Ingestion Completed')

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)


if __name__ == '__main__':

    data_ingestion_obj = DataIngestion()
    train_data_path, test_data_path = data_ingestion_obj.initiate_data_ingestion()

    print(train_data_path)
    print(test_data_path)
