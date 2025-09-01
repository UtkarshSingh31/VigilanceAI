import logging
import os 

from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)