import pandas as pd
from rapidfuzz import process, fuzz
from agents.state import MigrationAgentState

VALID_CSV = "data/valid_agencies.csv"
INVALID_CSV = "data/invalid_agencies.csv"

def _load_agencies():
    valid_df = pd.read_csv(VALID_CSV, sep="\t", skiprows=0, header=None,
                            names=["raw"], engine="python")
    return valid_df

def verify_agency_tool(agency_name_query: str) -> str:
    try:
        valid_df = pd.read_csv(VALID_CSV)
    except Exception:
        valid_df = None
    try:
        invalid_df = pd.read_csv(INVALID_CSV)
    except Exception:
        invalid_df = None

    combined_text = ""
    if valid_df is not None:
        combined_text = valid_df.to_string()
    if invalid_df is not None:
        combined_text += invalid_df.to_string()

    if agency_name_query.lower() in combined_text.lower():
        return f"A close match for '{agency_name_query}' was found in the SLBFE agency records. Please verify current status at slbfe.gov.lk as license validity dates change."
    return f"No close match found for '{agency_name_query}' in our sample records. Please verify directly at slbfe.gov.lk."

def agency_verifier_node(state: MigrationAgentState) -> dict:
    return {"agency_check_result": verify_agency_tool(state["user_query"])}
