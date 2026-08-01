from models.llm_config import get_synthesis_llm
from agents.state import MigrationAgentState

DISCLAIMER = "\n\n---\n*Informational guidance only - verify with SLBFE (slbfe.gov.lk) or the relevant embassy.*"

PROMPT = """Combine these worker outputs into one clear response:
Agency: {agency}
Country: {country}
Remittance: {remittance}
Question: {question}
Feedback to fix: {feedback}"""

def synthesizer_node(state: MigrationAgentState) -> dict:
    llm = get_synthesis_llm()
    prompt = PROMPT.format(
        agency=state.get("agency_check_result", "") or "N/A",
        country=state.get("country_requirements", "") or "N/A",
        remittance=state.get("remittance_advice", "") or "N/A",
        question=state["user_query"],
        feedback=state.get("critique", "") or "None - first draft.",
    )
    response = llm.invoke(prompt)
    return {"draft_answer": response.content + DISCLAIMER}
