my_id = 55
def logger_callback(BaseLogger):
    class CL_Logger(BaseLogger):
        def __init__(self,id,name="trainer", log_dir="logs"):
            super().__init__(name,log_dir)
            self.id=id

            self.LOGGING_TYPES.update({"plot" : self.plot, "info": self.info})


        def plot(self,title,data):
            pass

        def info(self,message):
            self.logger("base_info",message)
            print(f"Wow id is: {self.id}")

    cl_logger = CL_Logger(my_id)
    return cl_logger