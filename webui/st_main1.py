import streamlit as st
from src.core.agent import graph
from dotenv import load_dotenv

load_dotenv() # 加载环境变量
st.set_page_config(page_title="MSDSQA", page_icon=":robot:")

st.title("MSDS🧪问答系统")

if "messages" not in st.session_state:
    st.session_state.messages = []
# 初始化消息列表
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入MSDS相关问题"):
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

            # 调用主入口
        out = graph.invoke(
                {
                    "messages": messages
                },
                config={"configurable": {"thread_id": 42}},
            )

            # 显示回答
        answer = out["messages"][0].content
        st.markdown("回答：", answer)


# if __name__ == "__main__":
#     main()
