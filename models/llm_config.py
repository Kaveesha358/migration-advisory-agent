import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

def _get_secret(name: str) -> str:
    if hasattr(st, "secrets") and name in st.secrets:
        return st.secrets[name]
    return os.environ.get(name, "")

def get_router_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=_get_secret("GROQ_API_KEY"),
    )

def get_synthesis_llm():
    return ChatOpenAI(
        model="anthropic/claude-3.5-haiku",
        temperature=0.2,
        base_url="https://openrouter.ai/api/v1",
        api_key=_get_secret("OPENROUTER_API_KEY"),
    )

def get_critic_llm():
    return ChatOpenAI(
        model="anthropic/claude-3.5-haiku",
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=_get_secret("OPENROUTER_API_KEY"),
    )
