import logging
import os
from datetime import datetime
from clearml import Task

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

        
    def info(self,message):
        self.base_logger.info(message)

task = Task.init(
    project_name="taha-sama",  # Name of the ClearML project
    task_name=f"API Training",  # Name of the task
    task_type=Task.TaskTypes.optimizer,  # Type of the task (could also be "training", "testing", etc.)
)
logger = BaseLogger()
logger.info("This is a base logger info message.")
cfg.update({"task":task})
cl_logger = logger_callback(BaseLogger)
cl_logger.info("This is a custom info message from CL_Logger.")
data = {
    "X":[1,2,3],
    "Y":[4,5,6],
}
cl_logger.plot("test-plot","series1",data)

cl_logger.scaler("test-scaler","accuracy",0.45,iteration=1)
cl_logger.scaler("test-scaler","accuracy",0.75,iteration=2)
cl_logger.scaler("test-scaler","accuracy",0.80,iteration=3)
cl_logger.scaler("test-scaler","accuracy",0.82,iteration=4)

task.close()
