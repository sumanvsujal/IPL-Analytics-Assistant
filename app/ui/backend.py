"""Use an optional HTTP backend, or run the same API functions in-process."""
import os
import requests


def call_api(path: str, payload: dict | None = None) -> dict:
    url = os.getenv("API_URL", "").rstrip("/")
    if url:
        response = (requests.get(url + path, timeout=5) if payload is None
                    else requests.post(url + path, json=payload, timeout=120))
        response.raise_for_status()
        return response.json()

    from app.api.main import health, debug_retrieve, chat
    from app.api.schemas import RetrieveRequest, ChatRequest
    if path == "/health":
        result = health()
    elif path == "/debug/retrieve":
        result = debug_retrieve(RetrieveRequest(**payload))
    elif path == "/chat":
        result = chat(ChatRequest(**payload))
    else:
        raise ValueError(f"Unsupported local API path: {path}")
    return result.model_dump()
