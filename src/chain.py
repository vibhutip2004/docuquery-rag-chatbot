"""Conversational RAG chain with chat memory and source citations."""

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from src.ingest import load_vectorstore
from src.llm import get_llm


def build_qa_chain(top_k: int = 4):
    """Build a conversational QA chain over the vector store.

    - top_k: number of relevant chunks retrieved per question
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": top_k}
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return qa_chain
