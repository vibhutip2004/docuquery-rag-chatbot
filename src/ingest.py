"""Load documents, split into chunks, embed and store in a vector database."""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data/sample_docs"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(folder: str = DATA_DIR):
    """Load all PDF and TXT files from a folder as LangChain Documents."""
    docs = []
    folder = Path(folder)
    for file in sorted(folder.iterdir()):
        if file.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load())
        elif file.suffix.lower() == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
            docs.extend(loader.load())
    if not docs:
        raise FileNotFoundError(f"No PDF/TXT files found in '{folder}'")
    return docs


def split_chunks(documents, chunk_size: int = 500, chunk_overlap: int = 100):
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def get_embeddings():
    """Local, free embedding model (no API key required)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore(chunks, persist_dir: str = CHROMA_DIR):
    """Create and persist a ChromaDB vector store from chunks."""
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
    )
    vectorstore.persist()
    return vectorstore


def load_vectorstore(persist_dir: str = CHROMA_DIR):
    """Load an existing ChromaDB vector store."""
    return Chroma(
        persist_directory=persist_dir, embedding_function=get_embeddings()
    )


def rebuild_index(folder: str = DATA_DIR, persist_dir: str = CHROMA_DIR):
    """Full pipeline: load -> split -> embed -> store. Returns stats."""
    docs = load_documents(folder)
    chunks = split_chunks(docs)
    build_vectorstore(chunks, persist_dir)
    return {"documents": len(docs), "chunks": len(chunks)}
