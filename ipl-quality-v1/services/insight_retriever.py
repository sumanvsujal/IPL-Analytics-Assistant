"""
Insight Retriever — Semantic search over insights.json
=======================================================
Uses sentence-transformers for embeddings + FAISS for vector search.
Falls back to simple keyword matching if sentence-transformers is unavailable.

Build index:  python -m app.services.insight_retriever --build
Query:        Used programmatically via retrieve_insights()
"""

import json, pickle
from pathlib import Path
import numpy as np

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "analytics"
_INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "app" / "services" / "_index"
_INSIGHTS_FILE = _DATA_DIR / "insights.json"
_INDEX_FILE = _INDEX_DIR / "faiss_index.bin"
_META_FILE = _INDEX_DIR / "insight_meta.pkl"

# ── Load insights ───────────────────────────────────────────────────────

def load_insights() -> list[dict]:
    with open(_INSIGHTS_FILE) as f:
        return json.load(f)


# ── Embedding model (lazy-loaded) ──────────────────────────────────────

_MODEL = None

def _get_model():
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            _MODEL = "fallback"
    return _MODEL


def _embed(texts: list[str]) -> np.ndarray:
    model = _get_model()
    if model == "fallback":
        return None
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ── Build index ─────────────────────────────────────────────────────────

def build_index():
    """Build FAISS index from insights.json. Run once before using retriever."""
    import faiss

    insights = load_insights()
    texts = [ins["insight"] for ins in insights]

    print(f"Embedding {len(texts)} insights …")
    embeddings = _embed(texts)
    if embeddings is None:
        raise RuntimeError("sentence-transformers not available. pip install sentence-transformers")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (embeddings are normalized → cosine sim)
    index.add(embeddings.astype(np.float32))

    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(_INDEX_FILE))
    with open(_META_FILE, "wb") as f:
        pickle.dump(insights, f)

    print(f"✓ Index built: {len(texts)} vectors, dim={dim}")
    print(f"  Saved to: {_INDEX_FILE}")


# ── Load index ──────────────────────────────────────────────────────────

_INDEX = None
_META = None

def _load_index():
    global _INDEX, _META
    if _INDEX is not None:
        return
    if not _INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Insight index not found at {_INDEX_FILE}. "
            f"Run: python -m app.services.insight_retriever --build"
        )
    import faiss
    _INDEX = faiss.read_index(str(_INDEX_FILE))
    with open(_META_FILE, "rb") as f:
        _META = pickle.load(f)


# ── Retrieve ────────────────────────────────────────────────────────────

def retrieve_insights(query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve the top-k most relevant insights for a query.

    Returns list of dicts with 'insight', 'category', 'score', 'entities'.
    Falls back to keyword matching if FAISS is unavailable.
    """
    model = _get_model()

    # Fallback: keyword matching
    if model == "fallback":
        return _keyword_fallback(query, top_k)

    _load_index()
    q_vec = _embed([query]).astype(np.float32)
    scores, indices = _INDEX.search(q_vec, top_k)

    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0:
            continue
        ins = _META[idx]
        results.append({
            "insight": ins["insight"],
            "category": ins["category"],
            "entities": ins.get("entities", {}),
            "score": float(score),
        })
    return results


def _keyword_fallback(query: str, top_k: int) -> list[dict]:
    """Simple keyword overlap scoring when vector search is unavailable."""
    insights = load_insights()
    q_words = set(query.lower().split())
    scored = []
    for ins in insights:
        i_words = set(ins["insight"].lower().split())
        overlap = len(q_words & i_words)
        scored.append((overlap, ins))
    scored.sort(key=lambda x: -x[0])
    return [
        {"insight": ins["insight"], "category": ins["category"],
         "entities": ins.get("entities", {}), "score": float(score)}
        for score, ins in scored[:top_k]
    ]


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="Build the FAISS index")
    ap.add_argument("--query", type=str, help="Test a query against the index")
    args = ap.parse_args()

    if args.build:
        build_index()
    elif args.query:
        results = retrieve_insights(args.query)
        for r in results:
            print(f"  [{r['category']}] score={r['score']:.3f}")
            print(f"    {r['insight'][:150]}…")
    else:
        print("Usage: --build to create index, --query 'text' to search")
