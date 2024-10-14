import streamlit as st
import numpy as np
from mas.streamlit.propensity import run_meeting
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="MAS",
    page_icon=":robot:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("MAS - Multi Agent System")

c = st.container()

prompt = c.chat_input("Say something")
if prompt:
    c.write(f"User has sent the following prompt: {prompt}")

st.header("MAS, GPT-4o-mini 비교")

col1, col2 = st.columns(2, gap="medium")

with col1:
    st.subheader("MAS")

    with st.chat_message("user"):
        st.write("Hello 👋", prompt)
        if prompt:
            run_meeting(topic = prompt)
            # 걸린 시간 추가
            
        # st.line_chart(np.random.randn(30, 3))

with col2:
    st.subheader("GPT-4o-mini")

    with st.chat_message("user"):
        st.write("Hello 👋", prompt)
        # openai 채팅
        if prompt:
            llm = ChatOpenAI(model="gpt-4o-mini")
            parser = StrOutputParser()
            chain = llm | parser
            st.write(chain.invoke(prompt))
        # st.line_chart(np.random.randn(30, 3))