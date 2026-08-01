import streamlit as st
from rag.ingest import build_index
from agents.graph import run_query

st.set_page_config(page_title="Sri Lanka Migration Advisory Agent", page_icon="🧳")
st.title("🧳 Sri Lanka Foreign Employment Advisory Agent")
st.caption("Ask about agency verification, country legal rules, or remittance planning.")

@st.cache_resource
def load_index_once():
    try:
        build_index()
    except Exception as e:
        pass

with st.spinner("Initializing knowledge base..."):
    load_index_once()

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

user_input = st.chat_input("e.g., Is Blue Ocean Manpower SLBFE registered?")
if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            result = run_query(user_input)
            answer = result.get("final_answer") or result.get("draft_answer") or "No answer generated."
        except Exception as e:
            answer = f"Error executing pipeline: {e}"
        st.markdown(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})
