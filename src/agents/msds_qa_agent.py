from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.core import ToolSet

assert load_dotenv()


def get_graph(tools: list[BaseTool], chat_model: BaseChatModel) -> CompiledStateGraph:
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


if __name__ == "__main__":
    from src.model import SiliconflowClient

    client = SiliconflowClient()
    chat_model = client.get_chat_model()
    embed_model = client.get_embed_model()
    tools = [ToolSet.get_nrcc_chem_info_tool()]

    graph = get_graph(tools, chat_model)

    out = graph.invoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个有帮助的问答机器人，你拥有很多工具可以使用，自主进行回答用户的问题，必要时可以调用工具获取额外的信息。",
                },
                {
                    "role": "human",
                    "content": "白日依山尽，下一句是什么？",
                },
            ]
        },
        config={"configurable": {"thread_id": 1}},
    )

    print(out["messages"][-1].content)
