from langchain_core.messages.base import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.core import ToolSet
from src.db import FaissDB
from src.model import OllamaClient, SiliconflowClient

chat_model = SiliconflowClient().get_chat_model()
embed_model = OllamaClient().get_embed_model()


def get_graph(tools: list) -> CompiledStateGraph:
    llm_with_tool = chat_model.bind_tools(tools)

    def call_model(state) -> dict[str, BaseMessage]:
        return {"messages": llm_with_tool.invoke(state["messages"])}

    app = StateGraph(MessagesState)
    app.add_node("agent", call_model)
    app.add_node("tools", ToolNode(tools))

    app.add_conditional_edges("agent", tools_condition)
    app.add_edge("tools", "agent")
    app.set_entry_point("agent")

    checkpointer = MemorySaver()
    app = app.compile(checkpointer=checkpointer)

    return app


db = FaissDB(db_path="/root/Documents/msds-qa/kb", embed_model=embed_model)


tools = [
    ToolSet.get_retriever_tool(
        db=db.get_db(),
        name="chemical_msds_retriever",
        description="用于回答各种化学物质知识的工具",
    ),
]


graph = get_graph(tools)
out = graph.invoke(
    {
        "messages": [
            {
                "role": "system",
                "content": "你是一个有帮助的问答机器人，你拥有很多工具可以使用，你需要根据用户的提问来决定是否调用工具，自主进行回答用户的问题，必要时可以调用工具获取额外的信息。",
            },
            {
                "role": "human",
                "content": "1,1-二甲基环己烷与火反应会生成某些有害气体，请问这些气体有哪些？",
            },
        ]
    },
    config={"configurable": {"thread_id": 42}},
)
