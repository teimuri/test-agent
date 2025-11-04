from ml_trainer import AutoTrainer
from clearml import Task
class CL_Logger():
    def __init__(self):
        pass

    def plot(self,title,data):
        pass

    def info(self,message):
        self.logger.info(message)
        pritn("Wow")

    def logger_modifier(self,logger_func):
        logger_func.plot = self.plot
        logger_func.info = self.info
        self.logger = logger_func


def get_logger(name="trainer", log_dir="logs", logger_callback=None):
    # print(logger_callback)

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger_callback(logger) or logger, log_path
cl_logger = CL_Logger()
logger, _ =get_logger(log_dir="logs", logger_callback = cl_logger.logger_modifier)
logger.info("Starting ClearML Task")
