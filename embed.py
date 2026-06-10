"""
Embedding and retrieval for The Unofficial Guide.

Takes the chunks from ingest.py, embeds them with all-MiniLM-L6-v2, and
stores them in a persistent ChromaDB collection together with their source
metadata. Also provides retrieve() for semantic search.

Build (or rebuild) the index:
    python embed.py
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import load_and_chunk

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "unofficial_guide"
DB_PATH = Path(__file__).parent / "chroma_db"

_model = None


def get_model():
    """Load the embedding model once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _client():
    return chromadb.PersistentClient(path=str(DB_PATH))


def build_index():
    """Embed every chunk and (re)load it into ChromaDB with metadata.

    Uses cosine distance so scores are comparable to the 0-1 range the
    project describes (lower = more similar). Rebuilds from scratch each run
    so the index always matches the current documents.
    """
    records = load_and_chunk()
    print(f"Embedding {len(records)} chunks with {MODEL_NAME}...")
    embeddings = get_model().encode(
        [r["text"] for r in records], show_progress_bar=True
    )

    client = _client()
    # start clean so re-running never leaves stale chunks behind
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[f"{r['source']}::{r['chunk_index']}" for r in records],
        embeddings=[e.tolist() for e in embeddings],
        documents=[r["text"] for r in records],
        metadatas=[
            {
                "source": r["source"],
                "source_title": r["source_title"],
                "url": r["url"],
                "chunk_index": r["chunk_index"],
            }
            for r in records
        ],
    )
    print(f"Stored {collection.count()} chunks in ChromaDB at {DB_PATH}")
    return collection


def get_collection():
    """Open the existing collection (build_index must have been run first)."""
    return _client().get_collection(COLLECTION_NAME)


def retrieve(query, k=5):
    """Return the top-k most similar chunks to the query.

    Each result is a dict: {text, source, source_title, url, chunk_index, distance}.
    """
    query_embedding = get_model().encode([query])[0].tolist()
    result = get_collection().query(query_embeddings=[query_embedding], n_results=k)

    hits = []
    for doc, meta, dist in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({
            "text": doc,
            "source": meta["source"],
            "source_title": meta["source_title"],
            "url": meta["url"],
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        })
    return hits


if __name__ == "__main__":
    build_index()

    # Test retrieval against the evaluation-plan questions (planning.md)
    test_queries = [
        "How much does a Berkeley Student Cooperative standard single room cost per semester, and what's included?",
        "How many work-shift hours per week do co-op house members do, and how is that different for co-op apartment residents?",
        "What problems have tenants reported at Evans Manor?",
        "When should a student sign a lease for an August move-in?",
        "What's the cheapest way for a UC Berkeley student to live off campus?",
    ]

    for query in test_queries:
        print("\n" + "=" * 75)
        print(f"QUERY: {query}")
        print("=" * 75)
        for rank, hit in enumerate(retrieve(query, k=3), start=1):
            snippet = hit["text"][:220] + ("..." if len(hit["text"]) > 220 else "")
            print(f"\n  #{rank}  distance={hit['distance']:.3f}  "
                  f"[{hit['source']} #{hit['chunk_index']}]")
            print(f"      {snippet}")
