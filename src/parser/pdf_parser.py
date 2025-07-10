import itertools

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader


class PDFParser:
    def __init__(self, files: list[str]):
        self.loader = PyPDFLoader
        self.files: list[str] = self._check_pdf_files(files)

    def _check_pdf_files(self, files: list[str] | str) -> list[str]:

        files = [files] if isinstance(files, str) else files
        return files

    def invoke(self) -> list[Document]:
        documents = [self.loader(file).load_and_split(self.text_splitter) for file in self.files]  # fmt: skip
        documents = list(itertools.chain.from_iterable(documents))
        return documents
