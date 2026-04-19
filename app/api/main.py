"""
IPL Analytics API — FastAPI backend
=====================================
Wraps the existing hybrid RAG assistant with HTTP endpoints.

Run from project root:
    uvicorn app.api.main:app --reload --port 8000

Endpoints:
    GET  /health          — system status
    POST /chat            — full hybrid answer
    POST /debug/retrieve  — retrieval only (no LLM), for debugging/demos
"""

import json
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import (
    ChatRequest, ChatResponse,
    RetrieveRequest, RetrieveResponse,
    HealthResponse,
)
from app.services.query_router import classify_query
from app.services.orchestrator import _retrieve_structured, answer_query
from app.services.insight_retriever import retrieve_insights
from app.services.context_builder import build_context

logger = logging.getLogger("ipl_api")

# ── Paths ────────────────────────────────────────────────────────────────

_PROJECT = Path(__file__).resolve().parent.parent.parent
_ADAPTER_DIR = _PROJECT / "finetuning" / "outputs" / "final_adapter"
_DATA_DIR = _PROJECT / "data" / "analytics"


# ── App lifecycle ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup if adapter exists, or just log a warning."""
    try:
        from app.services.generator import load_model, is_finetuned
        load_model()
        logger.info(f"Model loaded (fine-tuned={is_finetuned()})")
    except Exception as e:
        logger.warning(f"Model not loaded at startup: {e}")
        logger.info("API will run in retrieval-only mode. /chat will fail until model is available.")
    yield


app = FastAPI(
    title="IPL Analytics Assistant API",
    description="Hybrid RAG assistant for IPL cricket analytics — structured retrieval + insight search + fine-tuned LLM.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
# GET /health
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
def health():
    """System status: model loaded, data available, table counts."""
    model_loaded = False
    try:
        from app.services.generator import _MODEL
        model_loaded = _MODEL is not None
    except Exception:
        pass

    parquet_count = len(list(_DATA_DIR.glob("*.parquet"))) if _DATA_DIR.exists() else 0
    insights_path = _DATA_DIR / "insights.json"
    insights_count = 0
    if insights_path.exists():
        with open(insights_path) as f:
            insights_count = len(json.load(f))

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        adapter_exists=_ADAPTER_DIR.exists(),
        analytics_data_exists=_DATA_DIR.exists(),
        analytics_tables_count=parquet_count,
        insights_count=insights_count,
    )


# ═══════════════════════════════════════════════════════════════════════════
# POST /chat
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Full hybrid answer: route → retrieve → generate.
    Requires the LLM to be loaded.
    """
    start = time.time()

    try:
        result = answer_query(req.query, verbose=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    route_info = result.get("route", {})
    elapsed = time.time() - start
    logger.info(f"[chat] {elapsed:.1f}s | {route_info.get('route')}/{route_info.get('intent')} | {req.query[:60]}")

    return ChatResponse(
        query=req.query,
        route=route_info.get("route", "unknown"),
        intent=route_info.get("intent", "unknown"),
        answer=result["answer"],
        structured_count=result.get("structured_count", 0),
        insight_count=result.get("insight_count", 0),
    )


# ═══════════════════════════════════════════════════════════════════════════
# POST /debug/retrieve
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/debug/retrieve", response_model=RetrieveResponse)
def debug_retrieve(req: RetrieveRequest):
    """
    Retrieval-only endpoint (no LLM). Returns routing, entities,
    structured results, insights, and assembled context.
    Useful for debugging and front-end demos without GPU.
    """
    ql = req.query.lower()
    route_info = classify_query(req.query)
    intent = route_info["intent"]
    entities = route_info["entities"]
    route = route_info["route"]

    # Retrieve
    structured = []
    insights = []

    if route in ("structured", "mixed"):
        structured = _retrieve_structured(ql, route_info)

    if route == "insight":
        insights = retrieve_insights(req.query, top_k=4)
    elif route == "mixed":
        insights = retrieve_insights(req.query, top_k=3)
    elif route == "structured" and not structured:
        insights = retrieve_insights(req.query, top_k=3)

    # Build context
    context = build_context(
        structured_results=structured,
        insight_results=insights,
        intent=intent,
        entities=entities,
    )

    # Serialize structured results (convert numpy/pandas types to plain dicts)
    serializable_structured = []
    for r in structured:
        entry = {"type": r.get("type", "")}
        if "entity" in r:
            entry["entity"] = r["entity"]
        if "data" in r:
            entry["data"] = {k: _to_json_safe(v) for k, v in r["data"].items()}
        if "rank" in r:
            entry["rank"] = r["rank"]
        serializable_structured.append(entry)

    serializable_insights = []
    for ins in insights:
        serializable_insights.append({
            "insight": ins.get("insight", ""),
            "category": ins.get("category", ""),
            "score": float(ins.get("score", 0)),
        })

    return RetrieveResponse(
        query=req.query,
        route=route,
        intent=intent,
        entities=EntitiesResponse(**entities),
        structured_count=len(structured),
        insight_count=len(insights),
        context=context,
        structured_results=serializable_structured,
        insight_results=serializable_insights,
    )


# ── Helpers ──────────────────────────────────────────────────────────────

# Need to import here (after schemas) to avoid circular issues
from app.api.schemas import EntitiesResponse


def _to_json_safe(val):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    import numpy as np
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if hasattr(val, 'item'):
        return val.item()
    return val
