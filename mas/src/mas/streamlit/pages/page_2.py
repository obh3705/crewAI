import streamlit as st
import pandas as pd
import numpy as np

st.title("회의 진행 페이지")

tab1, tab2 = st.tabs(["전체 보기", "회의록 보기"])

with tab1:
    st.write("전체 보기")

# TODO: session state 처리 더 알아보기
with tab2:
    # Read the markdown file
    def read_md_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    # TODO: 경로 수정
    # Path to the markdown file
    md_file_path = '/Users/obyeonghyeon/Desktop/Programing/crewAI/mas/src/mas/report.md'

    

    st.write("회의록 보기")
    # state 처리
    if 'md' not in st.session_state:
        st.session_state.md = read_md_file(md_file_path)
    else:
        # st.write("md in session state")
        st.write(st.session_state.md)


# # Read the markdown file
# def read_md_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return file.read()

# # TODO: 경로 수정
# # Path to the markdown file
# md_file_path = '/Users/obyeonghyeon/Desktop/Programing/crewAI/mas/src/mas/report.md'

# # state 처리
# if 'md' not in st.session_state:
#     st.session_state.md = read_md_file(md_file_path)
# md = read_md_file(md_file_path)

st.download_button(
    label="Download data as Markdown",
    data=st.session_state.md,
    file_name="large_df.md",
    mime="text/markdown",
)

sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
selected = st.feedback("thumbs")
if selected is not None:
    st.markdown(f"You selected: {sentiment_mapping[selected]}")