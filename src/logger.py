"""
Imported required libraries for logging file setup. 
Primary Library:-  Logging
For path and directory purpose:- OS
For datetime and time purpose:- Datetime.

"""

import logging
import os 
from datetime import datetime

# logfile name same as date and specific time.
LOG_FILE=f"{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"

# creating log file folder for each logging events.
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

# path of each log file.
LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

# basic configuration of log file.
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] - %(levelname)s - %(message)s",
    level=logging.info,
)

if __name__=="__main__":
    logging.info("Logging has started")