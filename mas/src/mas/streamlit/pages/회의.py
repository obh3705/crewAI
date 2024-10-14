# from mas.streamlit.pages.test import run_meeting
# from mas.streamlit.langgraph import run_meeting
import streamlit as st

st.title("회의 페이지")

st.write("회의를 시작합니다.")

st.text_input("프로젝트 이름", placeholder="프로젝트의 이름을 입력하세요")
st.date_input("회의 일자")
st.multiselect("담당자 설정", ["김철수", "박영희", "이영수"])
topic = st.text_area("회의 목표", placeholder="목표를 입력하세요. 미입력시 자유롭게 토론합니다.")


if st.button("회의 시작"):
    st.session_state.topic = topic
    st.write("회의가 시작되었습니다. '회의 진행' 페이지로 이동하세요.")
    st.switch_page("pages/회의 진행.py")