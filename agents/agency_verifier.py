import pandas as pd
from rapidfuzz import fuzz
from agents.state import MigrationAgentState

VALID_CSV = "data/valid_agencies.csv"
INVALID_CSV = "data/invalid_agencies.csv"


def _load_lines(path: str) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception:
        return []


def verify_agency_tool(agency_name_query: str) -> str:
    valid_lines = _load_lines(VALID_CSV)
    invalid_lines = _load_lines(INVALID_CSV)

    query_lower = agency_name_query.lower()

    def best_match(lines):
        best_score, best_line = 0, None
        for line in lines:
            score = fuzz.partial_ratio(query_lower, line.lower())
            if score > best_score:
                best_score, best_line = score, line
        return best_score, best_line

    valid_score, valid_line = best_match(valid_lines)
    invalid_score, invalid_line = best_match(invalid_lines)

    THRESHOLD = 70

    if valid_score >= THRESHOLD and valid_score >= invalid_score:
        return (
            f"A close match was found in the SLBFE **valid/registered** agency records "
            f"(match: \"{valid_line[:120]}\"). Please confirm current validity dates "
            f"directly at slbfe.gov.lk, as license status can change."
        )
    elif invalid_score >= THRESHOLD:
        return (
            f"⚠️ A close match was found in the SLBFE **invalid/cancelled** agency records "
            f"(match: \"{invalid_line[:120]}\"). Please verify directly at slbfe.gov.lk "
            f"before proceeding, and exercise caution."
        )
    else:
        return (
            f"No close match found for '{agency_name_query}' in our sample records. "
            f"Please verify directly at slbfe.gov.lk."
        )


def agency_verifier_node(state: MigrationAgentState) -> dict:
    return {"agency_check_result": verify_agency_tool(state["user_query"])}
