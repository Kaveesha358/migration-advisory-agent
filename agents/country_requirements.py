from rag.vectorstore import retrieve
from models.llm_config import get_synthesis_llm
from agents.state import MigrationAgentState

PROMPT = """Using ONLY the context below, answer the user's question about destination-country requirements.
Context:
{context}

Question: {question}"""

def country_requirements_node(state: MigrationAgentState) -> dict:
    hits = retrieve(state["user_query"], k=4, where={"doc_type": "country"})
    context = "\n\n---\n\n".join(h["text"] for h in hits) or "No relevant context found."
    llm = get_synthesis_llm()
    response = llm.invoke(PROMPT.format(context=context, question=state["user_query"]))
    return {"country_requirements": response.content}
