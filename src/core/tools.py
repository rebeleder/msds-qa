from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools.simple import Tool
from langchain_core.vectorstores import VectorStore


class ToolSet:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_retriever_tool(
        db: VectorStore,
        name: str,
        description: str,
    ) -> Tool:
        """创建一个检索工具"""
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.15},
        )
        retriever = create_retriever_tool(
            retriever=retriever,
            name=name,
            description=description,
        )
        return retriever
