from src.logger import logging

logging.debug("Debug log")  #This will not be printed since level=logging.INFO,

logging.info("Program has started")
logging.info("This is info message")
logging.warning("This is warning")
logging.error("This is error message")