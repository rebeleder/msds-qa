from typing import Type, Union

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.toolkits import ChemicalsDataSearchEngine, ChemInfoModel


class ChemInfoQuery(BaseModel):
    chem_name: str = Field(..., description="要查询的化学品名称")


class ChemInfoRetriever(BaseTool):
    name: str = "ChemInfoRetriever"
    description: str = (
        "从危险化学品信息数据库检索化学品信息的工具，输入必须为标准的化学品名称"
    )

    search_engine: ChemicalsDataSearchEngine = ChemicalsDataSearchEngine()
    args_schema: Type[BaseModel] = ChemInfoQuery

    def _run(self, chem_name: str) -> Union[ChemInfoModel, None]:
        """执行化学品信息查询"""
        chem_id = self.search_engine.get_idenDataId(chem_name)

        if chem_id:
            chem_info = self.search_engine.get_chemInfo(chem_id)

            return ChemInfoModel(**chem_info)
        return None
