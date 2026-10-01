"""Pluggable LLM provider: OpenAI / Groq / fully local HuggingFace model."""

import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()


def get_llm(provider: str = LLM_PROVIDER):
    """Return a LangChain LLM based on the chosen provider.

    - openai : best quality, needs OPENAI_API_KEY
    - groq   : fast + free tier, needs GROQ_API_KEY
    - local  : free, runs on your machine (default, no key needed)
    """
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    if provider == "groq" and os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

    # Fallback: small local model - zero cost, works offline
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline

    hf_pipe = pipeline(
        task="text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=256,
    )
    return HuggingFacePipeline(pipeline=hf_pipe)
