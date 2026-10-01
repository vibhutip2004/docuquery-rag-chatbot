# DocuQuery AI — RAG based Document Q&A Chatbot

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end **Retrieval-Augmented Generation (RAG)** application that lets you upload PDF/TXT
documents, stores them in a **vector database**, and answers your questions using an LLM —
with **source citations** from the documents.

Built as a learning/portfolio project covering the full modern Gen-AI stack:
**Python • LangChain • Embeddings • ChromaDB/FAISS • LLMs • Streamlit**

---

## ✨ Features

- 📄 **Multi-format ingestion** — load PDF and TXT files from a folder or upload via UI
- ✂️ **Smart chunking** — `RecursiveCharacterTextSplitter` with overlap for better retrieval
- 🧠 **Local embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (free, no API key needed)
- 🗄️ **Vector stores** — ChromaDB (persistent) with optional FAISS swap
- 💬 **Conversational memory** — follow-up questions work ("...and what about chapter 3?")
- 🔍 **Source citations** — see exactly which document chunks the answer came from
- 🤖 **Pluggable LLM** — OpenAI / Groq / **fully local HuggingFace model (no API key)**
- 🖥️ **Clean Streamlit UI** — chat history, upload panel, retrieval settings sidebar

## 🏗️ Architecture

```
PDF/TXT docs
     │
     ▼
┌─────────────────┐     ┌──────────────┐
│ Document Loader │────▶│ Text Splitter│  (LangChain)
└─────────────────┘     └──────┬───────┘
                               │ chunks
                               ▼
                   ┌──────────────────────┐
                   │ Embedding Model      │  (all-MiniLM-L6-v2)
                   └──────────┬───────────┘
                              │ vectors
                              ▼
                   ┌──────────────────────┐
                   │ ChromaDB (persistent)│ ◀──── user question (also embedded)
                   └──────────┬───────────┘
                              │ top-k similar chunks
                              ▼
                   ┌──────────────────────┐
                   │ LLM + Prompt         │  (OpenAI / Groq / local HF)
                   └──────────┬───────────┘
                              ▼
                   Answer + source citations
```

## 🚀 Getting Started

```bash
git clone https://github.com/<your-username>/docuquery-rag-chatbot.git
cd docuquery-rag-chatbot

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

### Run with a local model (no API key required) ⭐
```bash
streamlit run app.py
```
The app falls back to a small local HuggingFace model (`flan-t5-small`) automatically
if no API key is set. Perfect for demos on a laptop.

### Optional: use a stronger cloud LLM
```bash
cp .env.example .env
# fill in OPENAI_API_KEY or GROQ_API_KEY, then:
streamlit run app.py
```

## 🖥️ Usage

1. Put your documents in `data/sample_docs/` (a sample file is included)
2. Click **"Build / Rebuild Vector DB"** in the sidebar
3. Ask questions in the chat box — answers come with cited source chunks

Try with the included sample doc:
> *"What is the difference between supervised and unsupervised learning?"*

## 📁 Project Structure

```
docuquery-rag-chatbot/
├── app.py                  # Streamlit UI entry point
├── src/
│   ├── ingest.py           # Load, split, embed & store documents
│   ├── llm.py              # Pluggable LLM provider (OpenAI/Groq/local HF)
│   └── chain.py            # Conversational RAG chain with memory
├── scripts/
│   └── build_index.py      # CLI: build vector DB without the UI
├── data/sample_docs/       # Sample documents
├── chroma_db/              # Persistent vector store (auto-created)
├── requirements.txt
├── .env.example
├── .gitignore
└── LICENSE
```

## 🧰 Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| Orchestration | LangChain |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB (FAISS interchangeable) |
| LLM | OpenAI / Groq / HuggingFace (local) |
| UI | Streamlit |

## 🗺️ Roadmap / Future Work

- [ ] Hybrid search (BM25 + vector) for better retrieval
- [ ] Reranking with cross-encoder
- [ ] Evaluation metrics (faithfulness, answer relevancy)
- [ ] Docker deployment
- [ ] Multi-modal support (images in PDFs)

## 📚 What I Learned

- How text is **tokenized, chunked and embedded** into dense vectors
- How **similarity search** over a vector DB retrieves relevant context
- How **RAG grounds an LLM** on private data and reduces hallucination
- Trade-offs between local small models vs. cloud LLMs

## 📄 License

MIT — feel free to fork and build on it.
