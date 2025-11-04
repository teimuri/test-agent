import logging
import os
from datetime import datetime
class CL_Logger:
    def __init__(self,id,name="trainer", log_dir="logs"):
        self.id=id
        self.LOGGING_TYPES = {"plot" : self.plot, "info": self.info}

    def logger(self,type,*args,**kwargs):
        if type in self.LOGGING_TYPES:
            self.LOGGING_TYPES[type](*args,**kwargs)
            
    def plot(self,title,data):
        pass

    def info(self,message):
        self.logger("base_info",message)
        pritn("Wow")

cl_logger = CL_Logger(id=1)

def logger_callback(logger_func):
    logger_func.LOGGING_TYPES.update(cl_logger.LOGGING_TYPES)


class BaseLogger:
    def __init__(self, name="trainer", log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        self.base_logger = logging.getLogger(name)
        self.base_logger.setLevel(logging.INFO)

        if not self.base_logger.handlers:
            fh = logging.FileHandler(log_path)
            ch = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            self.base_logger.addHandler(fh)
            self.base_logger.addHandler(ch)

        self.LOGGING_TYPES = {"base_info": self.base_logger.info}
    
    def logger(self,type,*args,**kwargs):
        if type in self.LOGGING_TYPES:
            self.LOGGING_TYPES[type](*args,**kwargs)

logger = BaseLogger()
logger.logger("base_info","This is a base logger info message.")
logger_callback(logger)
logger.logger("info","This is a custom info message from CL_Logger.")

