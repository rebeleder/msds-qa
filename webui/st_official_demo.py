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
think_mode = st.sidebar.toggle('启用思考模式', value=False)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    # 根据思考模式添加前缀
    processed_prompt = f"/think {prompt}" if think_mode else f"/nothink {prompt}"
    
    # 向用户显示原始输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 创建一个空的占位符用于流式输出
        message_placeholder = st.empty()
        full_response = ""
        
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": processed_prompt if m is st.session_state.messages[-1] else m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        
        # 收集完整的响应
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                # 实时更新占位符内容
                message_placeholder.markdown(full_response + "▌")
        
        # 移除光标
        message_placeholder.markdown(full_response)
        
        # 检查是否包含深度思考内容
        if "Thinking" in full_response:
            try:
                # 分离思考过程和最终答案
                parts = full_response.split(r'(Thinking.*?done')
                thinking_process = parts[0].replace("思考过程：", "").strip()
                final_answer = parts[1].strip()
                
                # 清除原有内容
                message_placeholder.empty()
                
                # 显示最终答案
                st.markdown(final_answer)
                
                # 使用expander显示思考过程
                with st.expander("查看思考过程", expanded=True):
                    st.markdown(thinking_process)
                
                # 更新response只包含最终答案
                full_response = final_answer
                
            except Exception as e:
                st.error(f"解析思考过程时出错: {str(e)}")
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
