from src.exception import CustomException
from src.logger import logging
import sys

logging.debug("Debug log")  #This will not be printed since level=logging.INFO,

logging.info("Program has started")
logging.error("This is error message")
logging.info("This is info message")
logging.warning("This is warning")

try:
    raise Exception("Some error occurred")
except Exception as e:
    # logging.exception(e)
    # logging.error(CustomException(e,sys))
    raise CustomException(e, sys)
finally:
    logging.info("Program has ended")