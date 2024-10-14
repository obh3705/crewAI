import json
import streamlit as st

st.title("에이전트 설정")

st.write("에이전트를 설정합니다.")

# ./agent.json 파일을 읽어 에이전트의 이름과 성격을 출력
with open('./agent.json', 'r') as f:
    agent = json.load(f)
    st.write(f"에이전트 이름: {agent['name']}")
    st.write(f"에이전트 성격: {agent['analysis']}")