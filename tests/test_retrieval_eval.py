from rag.ingest import build_index
from rag.vectorstore import retrieve

SAMPLE_QUERIES = [
    "What is the minimum wage for a housemaid in Saudi Arabia?",
    "How much money can I remit monthly to Sri Lanka?",
]

if __name__ == "__main__":
    build_index()
    for q in SAMPLE_QUERIES:
        print("QUERY:", q)
        for h in retrieve(q, k=3):
            print(" -", h["text"][:150])
