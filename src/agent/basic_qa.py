from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class BasicQaQuery(BaseModel):
    query: str = Field(..., description="要回答的问题")


class BasicQaAgent(BaseTool):
    name: str = "basic_qa_agent"
    description: str = "用通用大语言模型回答问题"
    args_schema: Type[BaseModel] = BasicQaQuery

    def __init__(self, chat_model: BaseChatModel, **kwargs):
        super().__init__(**kwargs)
        self._chat_model = chat_model

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """使用通用大语言模型回答问题"""
        response = self._chat_model.invoke(query)
        return response.content if response else "不知道捏～"

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """使用通用大语言模型回答问题"""
        response = await self._chat_model.ainvoke(query)
        return response.content if response else "不知道捏～"
