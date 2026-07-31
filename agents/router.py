from models.llm_config import get_router_llm
from agents.state import MigrationAgentState

ROUTER_PROMPT = """You are a query router for a Sri Lankan foreign-employment advisory assistant.
Classify the user's question into exactly ONE category:
- "agency"     -> asking whether a recruitment agency is SLBFE-registered
- "country"    -> asking about visa/legal requirements for a destination country
- "remittance" -> asking about sending/receiving money, forex rules
- "other"      -> anything else

Reply with ONLY the single category word, nothing else.
User question: {query}"""

def router_node(state: MigrationAgentState) -> dict:
    llm = get_router_llm()
    prompt = ROUTER_PROMPT.format(query=state["user_query"])
    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if category not in {"agency", "country", "remittance"}:
        category = "other"

    return {"query_type": category}

def route_decision(state: MigrationAgentState) -> str:
    return state["query_type"]
