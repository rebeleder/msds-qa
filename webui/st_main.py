import streamlit as st
from src.core.agent import graph
from dotenv import load_dotenv

load_dotenv() # 加载环境变量
def main():
    st.title("MSDS问答系统")

    # 创建输入框
    question = st.text_input("请输入你的问题")

    if st.button("提交"):
        if question:
            # 构建消息
            messages = [
                {
                    "role": "system",
                    "content": "你是一个有帮助的问答机器人，你拥有很多工具可以使用，你需要根据用户的提问来决定是否调用工具，自主进行回答用户的问题，必要时可以调用工具获取额外的信息。",
                },
                {
                    "role": "human",
                    "content": question,
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
            answer = out["messages"][-1].content
            st.write("回答：", answer)
        else:
            st.warning("请输入问题")

if __name__ == "__main__":
    main()
