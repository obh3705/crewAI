from mas.streamlit.runmeeting import run_meeting
import streamlit as st
import pandas as pd
import numpy as np

st.title("회의 진행 페이지")

tab1, tab2 = st.tabs(["전체 보기", "회의록 보기"])

@st.cache_data
# Read the markdown file
def read_md_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    

with tab1:
    st.write("전체 보기")
    # 회의 결과 출력
    if 'topic' in st.session_state:
        run_meeting(topic = st.session_state.topic)
    else:
        st.write("회의 결과가 없습니다. '회의' 페이지에서 회의를 시작하세요.")

# TODO: session state 처리 더 알아보기
with tab2:
    st.write("회의록 보기")
    if 'topic' in st.session_state:
        st.session_state.md = read_md_file(".//langchain_response.md")
        st.write(st.session_state.md)
    else:
        st.write("회의록이 없습니다.")

    if 'md' not in st.session_state:
        st.write("No data to download")
    else:
        st.download_button(
            label="Download data as Markdown",
            data=st.session_state.md,
            file_name="회의록.md",
            mime="text/markdown",
        )

sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
selected = st.feedback("thumbs")
if selected is not None:
    st.markdown(f"You selected: {sentiment_mapping[selected]}")