import streamlit as st
import pandas as pd
import numpy as np

st.title("회의 진행 페이지")

tab1, tab2 = st.tabs(["전체 보기", "회의록 보기"])

with tab1:
    st.write("전체 보기")

with tab2:
    st.write("회의록 보기")


# TODO: 회의록을 데이터 프레임으로 만들기
df = pd.DataFrame(
    np.random.randn(50, 20),
    columns=('col %d' % i for i in range(20))
)

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

# 회의록 데이터프레임으로 만들기
csv = convert_df(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name="large_df.csv",
    mime="text/csv",
)

sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
selected = st.feedback("thumbs")
if selected is not None:
    st.markdown(f"You selected: {sentiment_mapping[selected]}")