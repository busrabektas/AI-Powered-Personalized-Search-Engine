import streamlit as st
from src.rag import generate_answer

st.set_page_config(page_title="📚 RAG Wikipedia Search", layout="wide")

st.title("Wikipedia Search with RAG (Qdrant + LLM)")
st.markdown("Search Wikipedia with a smart assistant powered by **vector search** and **LLMs**.")

query = st.text_input("Enter your question", placeholder="e.g. History of anarchism")

top_k = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=3)

if query:
    with st.spinner("Searching and generating answer..."):
        answer, sources = generate_answer(query, top_k=top_k)

    st.subheader("🧠 Answer")
    st.write(answer)

    st.subheader("📄 Sources")
    for i, (title, url) in enumerate(sources, 1):
        st.markdown(f"**{i}. [{title}]({url})**")

