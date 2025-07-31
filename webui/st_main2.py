from langchain_core.messages.base import BaseMessage
from src.config import hp
from openai import OpenAI
import streamlit as st
import re, json, pprint, os

st.title("MSDS🧪CHAT")

# 导入模型变量
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = hp.siliconflow_chat_model


# 设置消息历史记录文件的路径
HISTORY_FILE = "./webui/st_main2_chat_history.json"

# 如果聊天历史文件不存在，则创建一个空文件
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False)

# 保存聊天历史到文件
def save_chat_history(messages):
    """
    将聊天历史保存到文件
    参数:
        messages: 包含所有聊天消息的列表
    """
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False)

# 定义返回聊天历史的函数
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


# 将聊天历史加载到session_state中
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# 将session_state.messages中的消息显示在聊天界面上   
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 添加agent模式开关
use_agent = st.sidebar.toggle('启用Agent模式', value=False)
# 添加思考模式开关
think_mode = st.sidebar.toggle('启用思考模式', value=True)

# 添加清除历史消息按钮
if st.sidebar.button("清除聊天历史"):
    st.session_state.messages = []
    save_chat_history([])
    st.experimental_rerun()


# 处理用户输入
if prompt := st.chat_input("请输入MSDS相关问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        thinking_placeholder = st.empty()
        full_response = ""
        thinking_content = ""
        
        if use_agent:
            # 调用agent.py的功能
            from src.core.agent import graph

            out = graph.invoke(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一个有帮助的问答机器人，你拥有很多工具可以使用，你需要根据用户的提问来决定是否调用工具，自主进行回答用户的问题，必要时可以调用工具获取额外的信息。",
                        },
                        {
                            "role": "human",
                            "content": prompt,
                        },
                    ]
                },
                config={"configurable": {"thread_id": 42}},
            )

            # 添加调试打印
            print(f"Debug Output - out变量内容：{out}")
            # 序列化消息
            serializable = {
                "messages": [
                    {
                    "type": type(m).__name__,
                    "content": m.content
                    }
                for m in out["messages"]
            ]
            }
            # 打印序列化后的消息
            pprint.pprint(serializable)

            # 处理返回消息
            if isinstance(out["messages"][0], BaseMessage):
                full_response = out["messages"][-1].content
            else:
                full_response = json.dumps(out["messages"][-1], ensure_ascii=False, indent=2)
        else:
            # 普通模式，直接调用API
            # 创建空的占位符用于流式输出
            if think_mode:
                # 如果启用思考模式，使用expander来显示思考内容
                thinking_placeholder = st.expander("思考内容", expanded=True).empty() # 使用 empty() 创建占位符
                message_placeholder = st.empty()
            else:
                # 如果不启用思考模式，直接使用空的占位符
                thinking_placeholder = st.empty()

            client = OpenAI(
                base_url=hp.siliconflow_base_url,
            )

            # 根据思考模式添加前缀
            processed_prompt = f"/think {prompt}" if think_mode else f"/nothink {prompt}"

            # 调用OpenAI API进行流式响应
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": processed_prompt if m is st.session_state.messages[-1] else m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            
            # 收集完整的响应
            if think_mode:
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.reasoning_content is not None :
                        # 更新思考内容
                        thinking_content += delta.reasoning_content
                        # 使用container来更新内容，避免重复
                        thinking_placeholder.markdown(thinking_content + "▌")
                    elif delta.content is not None:
                        # 更新回答内容
                        full_response += delta.content
                        message_placeholder.markdown(full_response + "▌")
            elif not think_mode:
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content is not None:
                        # 更新回答内容
                        full_response += delta.content
                        message_placeholder.markdown(full_response + "▌")
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        save_chat_history(st.session_state.messages)