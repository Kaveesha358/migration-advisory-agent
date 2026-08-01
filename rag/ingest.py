import os
from pypdf import PdfReader
from rag.chunking import split_documents
from rag.vectorstore import index_chunks

DOCS_DIR = "data/sample_docs"

def _guess_metadata(filename: str) -> dict:
    name = filename.lower()
    doc_type = "remittance" if "remittance" in name or "cbsl" in name else ("agency" if "agency" in name else "country")
    country = "general"
    for c in ["saudi", "korea", "qatar", "uae", "kuwait", "israel"]:
        if c in name:
            country = c
            break
    return {"doc_type": doc_type, "country": country, "source_file": filename}

def load_raw_documents() -> list[dict]:
    docs = []
    if not os.path.exists(DOCS_DIR):
        return docs
    for fname in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, fname)
        if fname.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        elif fname.endswith(".pdf"):
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            continue
        docs.append({"text": text, "metadata": _guess_metadata(fname)})
    return docs

def build_index():
    raw_docs = load_raw_documents()
    chunks = split_documents(raw_docs)
    collection = index_chunks(chunks)
    return collection.count()
