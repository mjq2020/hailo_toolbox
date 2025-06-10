import contextlib
import time
from functools import wraps


class Timer(contextlib.ContextDecorator):
    def __init__(self, name: str = None, number: int = 1):
        self.name = name
        self.start_time = None
        self.number = number

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = (self.end_time - self.start_time) / self.number
        print(
            f"{self.name} took".rjust(30),
            f"{self.elapsed_time:.4f} seconds".center(16),
            f"FPS:{1 / self.elapsed_time:.2f}".ljust(20),
            f"Number of runs:{self.number}".ljust(20),
        )
        return self.elapsed_time

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper
