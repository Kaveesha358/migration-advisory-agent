from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class MigrationAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    query_type: str
    agency_check_result: str
    country_requirements: str
    remittance_advice: str
    draft_answer: str
    critique: str
    needs_revision: bool
    revision_count: int
    final_answer: str
