from langgraph.graph import StateGraph, END
from agents.state import MigrationAgentState
from agents.router import router_node, route_decision
from agents.agency_verifier import agency_verifier_node
from agents.country_requirements import country_requirements_node
from agents.remittance_advisor import remittance_advisor_node
from agents.synthesizer import synthesizer_node
from agents.critic import critic_node, critic_decision

def other_node(state: MigrationAgentState) -> dict:
    return {"remittance_advice": "Please ask about agency verification, country requirements, or remittance."}

def build_graph():
    graph = StateGraph(MigrationAgentState)
    graph.add_node("router", router_node)
    graph.add_node("agency_worker", agency_verifier_node)
    graph.add_node("country_worker", country_requirements_node)
    graph.add_node("remittance_worker", remittance_advisor_node)
    graph.add_node("other_worker", other_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision, {
        "agency": "agency_worker",
        "country": "country_worker",
        "remittance": "remittance_worker",
        "other": "other_worker",
    })

    for w in ["agency_worker", "country_worker", "remittance_worker", "other_worker"]:
        graph.add_edge(w, "synthesizer")

    graph.add_edge("synthesizer", "critic")
    graph.add_conditional_edges("critic", critic_decision, {"revise": "synthesizer", "approved": END})

    return graph.compile()

def run_query(user_query: str) -> MigrationAgentState:
    app = build_graph()
    initial_state: MigrationAgentState = {
        "messages": [], "user_query": user_query, "query_type": "",
        "agency_check_result": "", "country_requirements": "", "remittance_advice": "",
        "draft_answer": "", "critique": "", "needs_revision": False,
        "revision_count": 0, "final_answer": "",
    }
    return app.invoke(initial_state)
