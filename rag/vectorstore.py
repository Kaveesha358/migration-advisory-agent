import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "migration_corpus"

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=_embedding_fn)

def index_chunks(chunks: list[dict]):
    collection = get_collection()
    if collection.count() > 0:
        return collection
    collection.add(
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[f"chunk-{i}" for i in range(len(chunks))],
    )
    return collection

def retrieve(query: str, k: int = 4, where: dict | None = None) -> list[dict]:
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=k, where=where)
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]
