from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate

from src.agent import BasicQaAgent, KbQaAgent
from src.db import FaissDB
from src.model import OllamaClient, SiliconflowClient

load_dotenv()

client = OllamaClient()
chat_model = client.get_chat_model()
embed_model = client.get_embed_model()


db = FaissDB(db_path="/root/Documents/msds-qa/kb", embed_model=embed_model).get_db()


tools = [
    BasicQaAgent(
        chat_model=chat_model,
        name="通用问答代理",
        description="用通用大语言模型回答通用问题",
    ),
    KbQaAgent(
        db=db,
        chat_model=chat_model,
        name="化工行业化学物质问答代理",
        description="专为化学物质相关问题设计，当用户询问某些物种的msds相关内容时，使用知识库回答问题",
    ),
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个专业的问答助手，能够借助工具回答用户的问题。你需要严格地按照回答者的角度来回答问题。问什么答什么",
        ),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),  # Agent 用于记录思考和工具调用的地方
    ]
)


agent = create_tool_calling_agent(chat_model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"query": "2-丙醇会造成哪些健康危害？"})
