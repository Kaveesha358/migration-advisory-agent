from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 2400
CHUNK_OVERLAP = 400

def split_documents(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append({"text": piece, "metadata": {**doc["metadata"], "chunk_index": i}})
    return chunks
