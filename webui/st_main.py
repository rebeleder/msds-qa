import streamlit as st
from src.core.agent import graph
from dotenv import load_dotenv
import json, os

load_dotenv() # 加载环境变量

# 设置消息历史记录文件的路径
HISTORY_FILE = "./webui/chat_history.json"

def save_chat_history(messages):
    """
    将聊天历史保存到文件
    参数:
        messages: 包含所有聊天消息的列表
    """
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False)

def load_chat_history():
    """
    从文件加载聊天历史
    返回:
        list: 包含聊天记录的列表，如果文件不存在返回空列表
    """
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

# 页面配置
st.set_page_config(page_title="MSDSQA", page_icon=":robot:")
st.title("MSDS🧪问答系统")

# 初始化或加载聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# 添加清除历史按钮
if st.button("清除聊天历史"):
    st.session_state.messages = []
    save_chat_history([])
    st.experimental_rerun()

# 显示所有历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户输入
if prompt := st.chat_input("请输入MSDS相关问题"):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        messages = [
            {
                "role": "system",
                "content": "你是一个有帮助的问答机器人，你拥有很多工具可以使用，你需要根据用户的提问来决定是否调用工具，自主进行回答用户的问题，必要时可以调用工具获取额外的信息。",
            },
            {
                "role": "human",
                "content": prompt,
            },
        ]

        # 调用主入口获取回答
        out = graph.invoke(
            {
                "messages": messages
            },
            config={"configurable": {"thread_id": 42}},
        )

        # 显示回答
        answer = out["messages"][-1].content
        st.markdown(answer)
        
        # 添加助手回答到历史记录
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # 保存更新后的聊天历史
        save_chat_history(st.session_state.messages)
        
        # 输出当前聊天历史到控制台
        print(st.session_state.messages)  