"""
API Schemas — Request and response models for the IPL Analytics API.
"""

from pydantic import BaseModel, Field


# ── Request ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, examples=["Compare Bumrah and Chahal"])


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, examples=["Who are the top IPL run scorers?"])


# ── Response ─────────────────────────────────────────────────────────────

class EntitiesResponse(BaseModel):
    batters: list[str] = []
    bowlers: list[str] = []
    seasons: list[int] = []
    venues: list[str] = []
    phases: list[str] = []


class ChatResponse(BaseModel):
    query: str
    route: str
    intent: str
    answer: str
    structured_count: int = 0
    insight_count: int = 0


class RetrieveResponse(BaseModel):
    query: str
    route: str
    intent: str
    entities: EntitiesResponse
    structured_count: int = 0
    insight_count: int = 0
    context: str = ""
    structured_results: list[dict] = []
    insight_results: list[dict] = []


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    adapter_exists: bool
    analytics_data_exists: bool
    analytics_tables_count: int
    insights_count: int
