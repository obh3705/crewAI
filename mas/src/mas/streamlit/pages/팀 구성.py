import streamlit as st
import graphviz

st.title("팀 설정")

st.write("팀을 설정합니다.")

graph = graphviz.Digraph()
graph.edge("run", "initr")
graph.edge("intr", "runbl")
graph.edge("runbl", "run")

st.graphviz_chart(graph)