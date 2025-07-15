from openai import OpenAI
import streamlit as st

st.title("MSDS CHAT")

client = OpenAI(
    api_key="sk-385f3f6adee3414e8db87e9de84c7f78", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "deepseek-r1"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
