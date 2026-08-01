from rag.vectorstore import retrieve
from models.llm_config import get_synthesis_llm
from agents.state import MigrationAgentState

PROMPT = """Using ONLY the context below, answer the user's question about remittance rules.
Context:
{context}

Question: {question}"""

def remittance_advisor_node(state: MigrationAgentState) -> dict:
    hits = retrieve(state["user_query"], k=4, where={"doc_type": "remittance"})
    context = "\n\n---\n\n".join(h["text"] for h in hits) or "No relevant context found."
    llm = get_synthesis_llm()
    response = llm.invoke(PROMPT.format(context=context, question=state["user_query"]))
    return {"remittance_advice": response.content}
