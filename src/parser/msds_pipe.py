import itertools
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from src.db import FaissDB
from src.model import OllamaClient
from src.parser import FileChecker

ollama_client = OllamaClient()


def get_files_from_kb_space(kb_path: str) -> list[str]:
    """
    获取指定路径下的所有PDF文件

    :param kb_path: 存储MSDS的知识库路径

    :return: 知识文件列表
    """
    file_checker = FileChecker()
    if not os.path.exists(kb_path):
        raise ValueError(f"路径 {kb_path} 不存在")

    files = []
    for file in os.listdir(kb_path):
        file_path = os.path.join(kb_path, file)
        if file_checker.is_file_valid(file_path):
            files.append(file_path)

    return files


class MSDSParser:
    def __init__(self, files: list[str] | str):
        self.files: list[str] = files if isinstance(files, list) else [files]
        self.loader = PyPDFLoader

    def invoke(self) -> list[Document]:
        documents = [self.loader(file).load_and_split() for file in self.files]
        documents = list(itertools.chain.from_iterable(documents))
        return documents


class MSDS2DB:
    def __init__(
        self,
        files: list[str] | str,
        db_path: str = "/root/Documents/msds-qa/kb",
        embed_model: OllamaEmbeddings = ollama_client.get_embed_model(),
    ) -> None:
        self.files: list[str] = files if isinstance(files, list) else [files]
        self.parser = MSDSParser
        self.db = FaissDB(db_path, embed_model=embed_model)

    def invoke(self) -> FAISS:
        documents = self.parser(self.files).invoke()
        db = self.db.create_db(documents)

        os.makedirs(os.path.join(self.db.db_path, "files"), exist_ok=True)

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(
                executor.map(
                    lambda file: shutil.copy(
                        file,
                        os.path.join(self.db.db_path, "files", os.path.basename(file)),
                    ),
                    self.files,
                )
            )
        return db


if __name__ == "__main__":
    kb_files = get_files_from_kb_space("/root/Documents/msds-qa/assets")[:5]
    msds2db = MSDS2DB(files=kb_files)
    msds2db.invoke()
