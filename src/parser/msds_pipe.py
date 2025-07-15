import itertools
import os
import shutil

# import sys
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(D:\Users\Li\Documents\GitHub\msds-qa))))
# sys.path.append(project_root)

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from src.config import hp
from src.db import FaissDB
from src.model import OllamaClient, SiliconflowClient
from src.parser import FileChecker
from src.toolkits import parallel_map

# client = OllamaClient()
client = SiliconflowClient()


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
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=hp.max_chunk_size)  # fmt: skip
        self.loader = PyPDFLoader

    def format_context(self, context: Document) -> Document:
        source = context.metadata["source"]
        file_name = os.path.basename(source)
        context.page_content = f"<{file_name}>\n: {context.page_content}"
        return context

    def invoke(self) -> list[Document]:

        def load_and_format(file) -> list[Document]:
            docs = self.loader(file).load_and_split(self.text_splitter)
            return [self.format_context(doc) for doc in docs]

        documents = list(
            itertools.chain.from_iterable(
                parallel_map(
                    load_and_format,
                    self.files,
                    max_workers=10,
                    enable_tqdm=True,
                )
            )
        )
        return documents


class MSDS2DB:

    def __init__(
        self,
        files: list[str] | str,
        db_path: str = "./kb",
        embed_model: Embeddings = client.get_embed_model(),
    ) -> None:
        self.files: list[str] = files if isinstance(files, list) else [files]
        self.parser = MSDSParser
        self.db_path: str = db_path
        self.db = self.get_db()

    def get_documents(self) -> FAISS:
        documents = self.parser(self.files).invoke()

        return documents

    def get_db(self) -> FAISS:
        if os.path.exists(self.db_path) and os.path.isdir(self.db_path):
            documents = []
        else:
            documents = self.get_documents()

        db = FaissDB(
            db_path=self.db_path,
            embed_model=client.get_embed_model(),
            documents=documents,
        )
        return db.get_db()


if __name__ == "__main__":
    print("/n 欢迎来到MSDS-QA系统")
    kb_files = get_files_from_kb_space("./assets")[:5]
    msds2db = MSDS2DB(files=kb_files)
