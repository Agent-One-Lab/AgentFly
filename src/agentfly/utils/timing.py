""":class:`Timer`: record how long named sections take, through a context manager (``timing_context``) or a decorator (``timed_function``)."""

import time
from contextlib import contextmanager
from typing import Callable


class Timer:
    def __init__(self):
        self.timing_data = {}

    @contextmanager
    def timing_context(self, name: str):
        start_time = time.time()
        yield
        if name not in self.timing_data:
            self.timing_data[name] = 0
        self.timing_data[name] += time.time() - start_time  # Store execution time

    def timed_function(self, name: str):
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                with self.timing_context(name):  # Enter context
                    return func(*args, **kwargs)  # Execute function

            return wrapper

        return decorator
