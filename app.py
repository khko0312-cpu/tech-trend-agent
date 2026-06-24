import streamlit as st
from agent import run_agent

st.title("기술 트렌드 모니터링 Agent")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("질문을 입력하세요")

if st.button("실행") and user_input:
    with st.spinner("처리 중..."):
        result = run_agent(user_input)
    st.session_state.history.append((user_input, result))

for q, r in reversed(st.session_state.history):
    st.markdown(f"**질문:** {q}")
    st.markdown(r)
    st.divider()