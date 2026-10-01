

import streamlit as st

from src.ingest import rebuild_index
from src.chain import build_qa_chain

st.set_page_config(page_title="DocuQuery AI", page_icon="📄", layout="centered")

st.title("📄 DocuQuery AI")
st.caption("Ask questions about your documents - powered by RAG (LangChain + ChromaDB)")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, 4)
    st.divider()
    if st.button("🔨 Build / Rebuild Vector DB", use_container_width=True):
        with st.spinner("Loading, chunking, embedding..."):
            stats = rebuild_index()
        st.success(f"Indexed {stats['documents']} docs → {stats['chunks']} chunks")

    st.divider()
    st.info(
        "Put PDF/TXT files in `data/sample_docs/`, then rebuild the index. "
        "LLM defaults to a free local model."
    )

# ---------------- Chat ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.qa_chain is None:
                st.session_state.qa_chain = build_qa_chain(top_k=top_k)
            result = st.session_state.qa_chain.invoke({"question": question})
            answer = result["answer"]
            st.markdown(answer)

            # Show source chunks so the answer is traceable
            with st.expander("🔍 Sources"):
                for i, doc in enumerate(result["source_documents"], 1):
                    st.markdown(f"**Source {i}** (page {doc.metadata.get('page', 'N/A')})")
                    st.caption(doc.page_content[:400] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
