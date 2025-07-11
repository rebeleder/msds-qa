import logging
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


# def handler_exception(func):
#     """
#     异常处理修饰器
#     """

#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except Exception as e:
#             logging.error(f"Error in {func.__name__}: {e}")
#             raise

#     return wrapper


def test_it(func):  #

    @wraps(func)
    def wrapper(*args, **kwargs):

        result = func(*args, **kwargs)

        logging.info(f"{func.__name__}() return {result}")
        return result

    return wrapper


def check_db_exists(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_db_exists:
            raise ValueError("数据库不存在，无法执行此操作")
        return func(self, *args, **kwargs)

    return wrapper


def parallel_map(
    func: callable,
    container: list,
    max_workers=10,
    enable_tqdm: bool = False,
) -> list:
    """
    并行映射函数，使用线程池执行函数

    :param func: 要执行的函数

    :param container: 要处理的容器

    :param max_workers: 最大工作线程数

    :return: 函数执行结果列表
    """

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if enable_tqdm:
            results = list(
                tqdm(
                    executor.map(func, container),
                    total=len(container),
                    desc="Loading",
                    colour="green",
                )
            )
        else:
            results = list(executor.map(func, container))
    return results
