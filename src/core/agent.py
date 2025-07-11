from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.core import ToolSet
from src.db import FaissDB
from src.model import OllamaClient

chat_model = OllamaClient().get_chat_model()
embed_model = OllamaClient().get_embed_model()


def get_graph(tools: list) -> CompiledStateGraph:
    def call_model(state):
        llm_with_tool = chat_model.bind_tools(tools, tool_choice="any")
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
        name="异丁烷知识问答助手",
        description="能够回答异丁烷的性质、用途、危害和急救措施等问题。",
    ),
]


graph = get_graph(tools)

for event in graph.invoke(
    {"messages": "调用工具查询：氢化钙泄漏的时候应该怎么进行急救措施？"},
    config={"configurable": {"thread_id": 42}},
    # stream_mode="messages",
    debug=False,
):
    # print(event)
    pass
