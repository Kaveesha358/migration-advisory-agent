from models.llm_config import get_critic_llm
from agents.state import MigrationAgentState

PROMPT = """Review this draft. Check:
1. Has SLBFE/embassy disclaimer
2. No unverified claims stated as fact
3. Professional tone

If acceptable, reply strictly with: APPROVED
Otherwise, list what needs fixing.

Draft:
{draft}"""

def critic_node(state: MigrationAgentState) -> dict:
    llm = get_critic_llm()
    response = llm.invoke(PROMPT.format(draft=state["draft_answer"]))
    verdict = response.content.strip()
    already_revised = state.get("revision_count", 0)

    if verdict.upper().startswith("APPROVED") or already_revised >= 1:
        return {"critique": verdict, "needs_revision": False, "final_answer": state["draft_answer"]}
    return {"critique": verdict, "needs_revision": True, "revision_count": already_revised + 1}

def critic_decision(state: MigrationAgentState) -> str:
    return "revise" if state.get("needs_revision") else "approved"
