import os
import streamlit as st

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from agent import run_agent

st.title("기술 트렌드 모니터링 Agent")

# 앱이 처음 켜질 때 DB가 비어있으면 자동으로 한 번 수집
if "initialized" not in st.session_state:
    with st.spinner("초기 데이터 수집 중... (최초 1회, 약 1~2분 소요)"):
        if collection.count() == 0:
            execute_tool("collect_latest_articles", {})
    st.session_state.initialized = True
    
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