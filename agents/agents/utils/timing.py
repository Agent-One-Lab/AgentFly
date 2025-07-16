

from contextlib import contextmanager
import time
from typing import Callable, Dict


    
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
    

# Example usage
if __name__ == "__main__":
    timer = Timer()

    @timer.timed_function("example_function")
    def example():
        time.sleep(1)  # Simulating work

    with timer.timing_context("context_block"):
        time.sleep(2)  # Simulating work inside a context

    example()

    print(timer.timing_data)