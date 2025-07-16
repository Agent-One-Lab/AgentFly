import os
from typing import Union
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter, DEBUG


class Logger:
    def __init__(self, logger, local_rank):
        self.logger = logger
        self.local_rank = local_rank

    def info(self, msg):
        if self.local_rank in [-1, 0]:
            self.logger.info(msg)

    def debug(self, msg):
        if self.local_rank in [-1, 0]:
            self.logger.debug(msg)

    def warning(self, msg):
        if self.local_rank in [-1, 0]:
            self.logger.warning(msg)
    
    def error(self, msg):
        if self.local_rank in [-1, 0]:
            self.logger.error(msg)


def get_logger(directory, filename, level="INFO", local_rank=-1):
    print(f"Local rank: {local_rank}")
    os.makedirs(directory, exist_ok=True)
    if filename is None:
        filename = "log"
    filename = directory + '/' + filename.replace('/', '_')
    logger = getLogger(__name__)
    logger.propagate = False
    logger.handlers.clear()
    if level == "INFO":
        handler2 = FileHandler(filename=f"{filename}.log")
        handler2.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler2)
        logger.setLevel(INFO)
    elif level == "DEBUG":
        handler1 = StreamHandler()
        handler1.setFormatter(Formatter("%(message)s"))
        handler2 = FileHandler(filename=f"{filename}.log")
        handler2.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(DEBUG)
    else:
        raise ValueError(f"Unknown level: {level}")

    logger = Logger(logger, local_rank)

    return logger