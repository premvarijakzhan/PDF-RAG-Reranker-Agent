# streamlit_app.py
import streamlit as st
from agentic_rag import RAGOrchestrator

st.set_page_config(page_title="Agentic PDF RAG", layout="wide")
st.title("Agentic RAG")

if "rag" not in st.session_state:
    st.session_state.rag = None

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded.read())
    if st.button("Ingest PDF"):
        with st.spinner("Processing and indexing…"):
            rag = RAGOrchestrator()
            rag.ingest("temp.pdf")
            st.session_state.rag = rag
        st.success("📚 PDF ingested!")

if st.session_state.rag:
    question = st.text_input("Ask a question about your document:")
    if question:
        with st.spinner("Thinking…"):
            answer = st.session_state.rag.query(question)
        st.markdown("**Answer:**")
        st.write(answer)