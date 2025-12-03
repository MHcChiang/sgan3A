import sys
import os

class Logger(object):
    def __init__(self, file_path=None):
        self.terminal = sys.stdout
        self.file = None
        if file_path:
            self.set_file(file_path)

    def set_file(self, file_path):
        if self.file is not None:
            self.file.close()
        self.file = open(file_path, "a")

    def write(self, message):
        # Required for AgentFormer's data_generator
        self.terminal.write(message)
        # if self.file:
        #     self.file.write(message)
        #     self.file.flush()

    def flush(self):
        # Required for python file-like object compatibility
        self.terminal.flush()
        if self.file:
            self.file.flush()

    def info(self, message):
        # Convenience method for your training script
        self.write(message + '\n')
        
    def close(self):
        if self.file:
            self.file.close()