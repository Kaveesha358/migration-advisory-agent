from langgraph.graph import StateGraph, START, END
from agents.state import MigrationAgentState
from agents.country_requirements import country_requirements_node
from agents.remittance_advisor import remittance_advisor_node
from agents.synthesizer import synthesizer_node

def build_parallel_graph():
    graph = StateGraph(MigrationAgentState)
    graph.add_node("country_worker", country_requirements_node)
    graph.add_node("remittance_worker", remittance_advisor_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "country_worker")
    graph.add_edge(START, "remittance_worker")
    graph.add_edge("country_worker", "synthesizer")
    graph.add_edge("remittance_worker", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()
