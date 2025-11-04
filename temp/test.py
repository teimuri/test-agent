import logging
import os
from datetime import datetime

from cl_logger import logger_callback,cfg



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
cfg.update({"task_id":12345})
cl_logger = logger_callback(BaseLogger)
cl_logger.logger("info","This is a custom info message from CL_Logger.")

