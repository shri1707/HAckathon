"""Web UI server for the BIS Standards Recommendation Engine."""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent import BISRecommendationAgent
from src.config import Settings


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=3)
    id: str = "UI-QUERY"
    top_k: int = Field(5, ge=1, le=10)


class RecommendResponse(BaseModel):
    id: str
    query: str
    retrieved_standards: list[str]
    latency_seconds: float
    mode: str
    out_of_scope: bool = False
    message: str = ""


ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"

app = FastAPI(title="BIS Standards Recommendation Engine")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

_agent_lock = threading.Lock()
_agent_cache: dict[int, BISRecommendationAgent] = {}


def get_agent(top_k: int) -> BISRecommendationAgent:
    with _agent_lock:
        agent = _agent_cache.get(top_k)
        if agent is None:
            agent = BISRecommendationAgent(settings=Settings.from_env(), top_k=top_k)
            _agent_cache[top_k] = agent
        return agent


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    settings = Settings.from_env()
    return {
        "ok": True,
        "dataset_path": str(settings.dataset_path),
        "index_dir": str(settings.index_dir),
        "embedding_model": settings.embedding_model,
        "gemini_model": settings.gemini_model,
        "gemini_enabled": bool(settings.use_gemini and settings.gemini_api_key),
    }


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    agent = get_agent(request.top_k)
    result = agent.recommend(query_id=request.id, query=query)
    if result.get("out_of_scope"):
        mode = "out-of-scope"
    else:
        mode = "gemini" if agent.reranker is not None else "local-embedding"
    return RecommendResponse(
        id=result["id"],
        query=query,
        retrieved_standards=result["retrieved_standards"],
        latency_seconds=result["latency_seconds"],
        mode=mode,
        out_of_scope=result.get("out_of_scope", False),
        message=result.get("message", ""),
    )
