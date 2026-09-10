import streamlit as st
from rag import build_index
from chatbot import answer_college_query, answer_study_query

st.set_page_config(page_title="College AI Assistant")
st.title("🎓 College AI Assistant")

@st.cache_resource
def get_index():
    return build_index()

index, chunks = get_index()

mode = st.radio("Choose mode:", ["📘 College Info", "📚 Study Help"], horizontal=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask something...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    if mode == "📘 College Info":
        reply = answer_college_query(query, index, chunks)
    else:
        reply = answer_study_query(query)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)