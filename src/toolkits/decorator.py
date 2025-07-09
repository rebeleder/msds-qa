import logging
from functools import wraps

logging.basicConfig(level=logging.INFO)


def test_it(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        result = func(*args, **kwargs)

        logging.info(f"{func.__name__}() return {result}")
        return result

    return wrapper
