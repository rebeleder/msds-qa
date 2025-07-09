import os


class FileChecker:
    def __init__(self):

        self.supported_suffixes = self.get_supported_suffixes()

    def is_suffix_valid(self, file_path: str, suffix: list[str]) -> bool:
        """
        检查文件后缀是否有效

        :param file_path: 文件路径
        :param suffix: 期望的文件后缀

        :return: 如果文件后缀有效，返回 True，否则返回 False
        """
        if os.path.splitext(file_path)[-1] in suffix:
            return True
        else:
            return False

    def is_prefix_valid(self, file_path: str, prefix: str) -> bool: ...

    @classmethod
    def get_supported_suffixes(cls) -> list[str]:
        """
        返回支持的文件后缀列表

        :return: 支持的文件后缀列表
        """
        return [".pdf"]

    def is_file_valid(self, file_path: str) -> bool:
        """
        检查文件是否存在且后缀有效

        :param file_path: 文件路径

        :return: 如果文件存在且后缀有效，返回 True，否则返回 False
        """
        if not os.path.exists(file_path):
            return False
        if self.is_suffix_valid(file_path, self.supported_suffixes):
            return True
        else:
            return False
