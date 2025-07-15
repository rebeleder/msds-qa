import os

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from src.db import FaissDB
from src.model import OllamaClient, SiliconflowClient
from src.parser import MsdsParser

client = OllamaClient()
# client = SiliconflowClient()


class Msds2DB:

    def __init__(
        self,
        files: list[str] | str,
        db_path: str = "/root/Documents/msds-qa/kb",
        embed_model: Embeddings = client.get_embed_model(),
    ) -> None:
        self.files: list[str] = files if isinstance(files, list) else [files]
        self.parser = MsdsParser
        self.db_path: str = db_path
        self.db = self.get_db()

    def get_documents(self) -> list[Document]:
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
    from src.toolkits import get_files_from_kb_space

    kb_files = get_files_from_kb_space("/root/Documents/msds-qa/assets")[:5]
    msds2db = Msds2DB(files=kb_files)
