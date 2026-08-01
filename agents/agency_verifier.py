import pandas as pd
from rapidfuzz import process, fuzz
from agents.state import MigrationAgentState

AGENCY_CSV_PATH = "data/agency_list_sample.csv"

def verify_agency_tool(agency_name_query: str) -> str:
    df = pd.read_csv(AGENCY_CSV_PATH)
    choices = df["agency_name"].tolist()
    match = process.extractOne(agency_name_query, choices, scorer=fuzz.WRatio)
    if match is None or match[1] < 60:
        return f"No close match found for '{agency_name_query}'. Verify at slbfe.gov.lk."
    matched_name, score, idx = match
    row = df.iloc[idx]
    return f"Closest match: '{row['agency_name']}' (license {row['license_no']}), status: {row['status']}."

def agency_verifier_node(state: MigrationAgentState) -> dict:
    return {"agency_check_result": verify_agency_tool(state["user_query"])}
