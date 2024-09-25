import streamlit as st
import numpy as np


st.title("MAS - Multi Agent System")

# add_selectbox = st.sidebar.selectbox(
#     'How would you like to be contacted?',
#     ('Email', 'Home phone', 'Mobile phone')
# )

c = st.container()

prompt = c.chat_input("Say something")
if prompt:
    c.write(f"User has sent the following prompt: {prompt}")

st.header("MAS, GPT-4o-mini 비교")

st.subheader("MAS")

with st.chat_message("user"):
    st.write("Hello 👋", prompt)
    st.line_chart(np.random.randn(30, 3))

st.divider()


st.subheader("GPT-4o-mini")

with st.chat_message("user"):
    st.write("Hello 👋", prompt)
    st.line_chart(np.random.randn(30, 3))

