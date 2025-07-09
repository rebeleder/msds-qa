from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate

from src.agent import BasicQaAgent, KbQaAgent
from src.db import FaissDB
from src.model import OllamaClient

ollama_client = OllamaClient()
chat_model, embed_model = ollama_client.get_chat_embed_model()

db = FaissDB(
    db_path="/root/Documents/msds-qa/kb",
    embed_model=embed_model,
).get_db()


tools = [BasicQaAgent(chat_model=chat_model), KbQaAgent(db=db, chat_model=chat_model)]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that can answer questions about normal topics and knowledge bases.",
        ),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),  # Agent 用于记录思考和工具调用的地方
    ]
)


agent = create_tool_calling_agent(chat_model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(agent_executor.invoke({"query": "在知识库中查询一下：连上wifi能看电影吗？"}))
