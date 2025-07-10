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
    """基本问答代理
    Args:
        name (str): 工具名称
        description (str): 工具描述

    Returns:
        基本问答代理工具实例 (BaseTool)
    """
    name: str
    description: str
    args_schema: Type[BaseModel] = BasicQaQuery

    def __init__(self, chat_model: BaseChatModel, **kwargs) -> None:
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
