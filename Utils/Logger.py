import os
import time, atexit
from datetime import datetime
from pathlib import Path

class Logger:
    def __init__(self, log_path, log_name=""):
        if not os.path.exists(log_path):
            Path(log_path).mkdir(parents=True, exist_ok=True)
        self.path = Path(log_path)
        self.log_name = log_name + datetime.now().strftime("%a%d%B-%H_%M") + ".log"
        self.log_path = self.path / self.log_name
        self.log_file = open(self.log_path, "w+")
        self.log_file.close()
        print("<Saving logs to: [{}]>".format(self.path / log_name))
        atexit.register(self.cleanup)
    def cleanup(self):
        print("<Closing log file>")
        self.log_file.close()
    def log(self, message):
        print("<Writing log...>")
        with open(self.log_path, "a") as f:
            f.write(message)
        