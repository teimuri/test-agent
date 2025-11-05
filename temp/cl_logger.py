from clearml import Task
import matplotlib.pyplot as plt
cfg = {'task': None}
def logger_callback(BaseLogger):
    class CL_Logger(BaseLogger):
        def __init__(self,name="trainer", log_dir="logs"):
            super().__init__(name,log_dir)
            self.task = cfg['task']
            self.logger = self.task.get_logger()

        def scaler(self, title, series, value, iteration=0):
            self.logger.report_scalar(
                title=title,
                series=series,
                value=value,
                iteration=iteration,
            )
        
        def plot(self,title,series,data,iteration=0):
            print("Plotting data:",data)
            plt.plot(**data)
            self.logger.report_matplotlib_figure(
                title=title,
                series=series,
                figure=plt.gcf(),
                iteration=iteration,
            )
        
        def hyperparameters(self,params,name="Hyperparameters"):
            self.task.connect(params,name=name)


    cl_logger = CL_Logger()
    return cl_logger