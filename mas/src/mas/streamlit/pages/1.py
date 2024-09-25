import streamlit as st

st.title("회의 페이지")

st.write("회의를 시작합니다.")

st.text_input("프로젝트 이름", "프로젝트의 이름을 입력하세요")
st.date_input("회의 일자")
st.multiselect("담당자 설정", ["김철수", "박영희", "이영수"])
st.text_area("회의 목표", "목표를 입력하세요")


st.button("회의 시작")