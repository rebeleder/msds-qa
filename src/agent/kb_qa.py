import os
from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class KbQaQuery(BaseModel):
    query: str = Field(..., description="要回答的问题")
    # kb_name: str = Field(..., description="知识库名称")


class KbQaAgent(BaseTool):
    name: str = "kb_qa_agent"
    description: str = "大语言模型用知识库回答wifi问题"
    args_schema: Type[BaseModel] = KbQaQuery

    def __init__(self, db: FAISS, chat_model: BaseChatModel, **kwargs):
        super().__init__(**kwargs)
        self._db = db
        self._chat_model = chat_model

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """大语言模型使用知识库回答wifi问题"""
        retriever = self._db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        context = retriever.invoke(query)
        context = self.format_context(context)

        prompt = (
            f"""
            已知以下信息片段
            >>>{context}<<<
            请回答以下问题：
            >>>{query}<<<
            现在开始回答问题：
            """,
        )

        response = self._chat_model.invoke(prompt)
        return response.content if response else "不知道捏～"

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """大语言模型使用知识库回答知识库问题"""
        retriever = self._db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        context = await retriever.ainvoke(query)
        context = self.format_context(context)

        prompt = (
            f"""
            已知以下信息片段
            >>>{context}<<<
            请回答以下问题：
            >>>{query}<<<
            现在开始回答问题：
            """,
        )

        response = await self._chat_model.ainvoke(prompt)
        return response.content if response else "不知道捏～"

    def format_context(self, context: list[Document]) -> str:
        ret = ""
        for idx, doc in enumerate(context):
            ret += f"\n\n{os.path.splitext(os.path.basename(doc.metadata['source']))[0]} 相关信息 \n ---------- {idx + 1}：{doc.page_content}----------\n\n"

        return ret.strip()
