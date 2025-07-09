import os

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


class FaissDB:
    def __init__(
        self,
        db_path: str,
        embed_model: Embeddings,
    ) -> None:
        self.db_path: str = db_path
        self.embed_model: Embeddings = embed_model
        self.db: FAISS = self.load_db()

    def is_db_exists(self) -> bool:
        """检查FAISS数据库是否存在"""
        return os.path.exists(self.db_path)

    def create_db(self, documents: list[Document]) -> FAISS:
        """创建FAISS数据库"""
        try:
            db = FAISS.from_documents(
                documents,
                self.embed_model,
            )
            db.save_local(self.db_path)
            return db
        except Exception:
            raise ValueError("无法创建FAISS数据库，请检查文档和嵌入模型是否正确")

    def load_db(self) -> FAISS:
        """加载FAISS数据库"""
        try:
            if self.is_db_exists():
                db = FAISS.load_local(
                    self.db_path,
                    self.embed_model,
                    allow_dangerous_deserialization=True,
                )
                return db
        except Exception:
            raise ValueError(f"无法加载路径位于 {self.db_path} 的FAISS数据库")

    def get_db(self) -> FAISS:
        """获取FAISS数据库实例"""
        if not self.db:
            raise ValueError("FAISS数据库未加载或不存在")
        return self.db

    def save_db(self, db_path: str = None):
        """保存FAISS数据库"""
        db_path = db_path if db_path else self.db_path
        try:
            self.db.save_local(db_path)
        except Exception:
            raise ValueError(f"无法保存FAISS数据库到路径 {db_path}")

    # def delete_db(self) -> None:
    #     """删除FAISS数据库"""
    #     try:
    #         if os.path.exists(self.db_path):
    #             os.remove(self.db_path)
    #         else:
    #             raise FileNotFoundError(f"FAISS数据库文件不存在")
    #     except Exception:
    #         raise ValueError(f"无法删除FAISS数据库文件 {self.db_path}")

    def add_to_db(self, documents: list[Document]) -> None:
        """将文档添加到FAISS数据库"""
        try:
            self.db.add_documents(documents)
            self.save_db(self.db)
        except Exception:
            raise ValueError("无法将文档向量化并添加到FAISS数据库")

    def get_retriever(self, search_type: str = "similarity", k: int = 5):
        """获取检索器"""
        return self.db.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k},
        )


if __name__ == "__main__":
    from langchain_ollama import OllamaEmbeddings

    embed_model = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://192.168.215.3:11434",
    )

    real_documents_data = [
        {
            "content": "无线路由器能发射无线信号，允许设备连接到互联网。",
            "source": "无线路由器说明书.docx",
        },
        {
            "content": "设备连接到无线路由器后，可以访问互联网。",
            "source": "无线路由器说明书.docx",
        },
        {
            "content": "访问无线网后可以休闲娱乐，观看视频、浏览网页等。",
            "source": "科技生活.pdf",
        },
    ]

    real_docs = [
        Document(page_content=d["content"], metadata={"source": d["source"]})
        for d in real_documents_data
    ]

    db = FAISS.from_documents(real_docs, embed_model)

    db.save_local("/root/Documents/msds-qa/kb")
