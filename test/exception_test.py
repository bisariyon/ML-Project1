import sys
from src.exception import CustomException
from src.logger import logging

try:
    print("Program Started")
    a = 1 / 0
except Exception as e:
    logging.error(e)
    raise CustomException(e, sys)