from openai import OpenAI
import streamlit as st
from src.config import hp
import re

st.title("MSDS CHAT")

client = OpenAI(
    base_url=hp.siliconflow_base_url,
    )

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = hp.siliconflow_chat_model

if "messages" not in st.session_state:
    st.session_state.messages = []

# 添加思考模式开关
think_mode = st.sidebar.toggle('启用思考模式', value=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # 根据思考模式添加前缀
    processed_prompt = f"/think {prompt}" if think_mode else f"/nothink {prompt}"
    
    # 将用户的原始输入添加到会话状态和显示中
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 创建空的占位符用于流式输出
        if think_mode:
            # 如果启用思考模式，使用expander来显示思考内容
            thinking_placeholder = st.expander("思考内容", expanded=True).empty() # 使用 empty() 创建占位符
        else:
            # 如果不启用思考模式，直接使用空的占位符
            thinking_placeholder = st.empty()

        # 模型思考内容和最终回答内容的占位符
        message_placeholder = st.empty()
        full_response = ""
        thinking_content = ""
        
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
                    print(delta.reasoning_content)
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

    # 移除光标并整理最终显示
    if thinking_content:
        thinking_placeholder.markdown(thinking_content)
        message_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
