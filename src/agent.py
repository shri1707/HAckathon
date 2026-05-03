"""LangGraph agent for BIS standard recommendations."""

from __future__ import annotations

import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .config import Settings
from .index import VectorIndex
from .llm import GeminiReranker
from .retriever import Candidate, Retriever, looks_like_domain_query


class AgentState(TypedDict, total=False):
    id: str
    query: str
    candidates: list[Candidate]
    retrieved_standards: list[str]
    latency_seconds: float
    started_at: float
    out_of_scope: bool
    message: str


class BISRecommendationAgent:
    def __init__(self, settings: Settings, top_k: int = 5) -> None:
        self.settings = settings
        self.top_k = top_k
        self.index = VectorIndex(
            model_name=settings.embedding_model,
            index_dir=settings.index_dir,
            dataset_path=settings.dataset_path,
        )
        self.index.load_or_build()
        self.retriever = Retriever(self.index)
        self.reranker = (
            GeminiReranker(settings.gemini_api_key, settings.gemini_model)
            if settings.use_gemini and settings.gemini_api_key
            else None
        )
        self.graph = self._build_graph()

    def recommend(self, query_id: str, query: str) -> dict:
        state = self.graph.invoke({"id": query_id, "query": query, "started_at": time.perf_counter()})
        return {
            "id": state["id"],
            "retrieved_standards": state["retrieved_standards"],
            "latency_seconds": state["latency_seconds"],
            "out_of_scope": state.get("out_of_scope", False),
            "message": state.get("message", ""),
        }

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("retrieve_candidates", self._retrieve_candidates)
        graph.add_node("check_scope", self._check_scope)
        graph.add_node("rank_with_gemini", self._rank_with_gemini)
        graph.add_node("finish", self._finish)
        graph.set_entry_point("retrieve_candidates")
        graph.add_edge("retrieve_candidates", "check_scope")
        graph.add_edge("check_scope", "rank_with_gemini")
        graph.add_edge("rank_with_gemini", "finish")
        graph.add_edge("finish", END)
        return graph.compile()

    def _retrieve_candidates(self, state: AgentState) -> AgentState:
        if not looks_like_domain_query(state["query"]):
            return {
                "candidates": [],
                "out_of_scope": True,
                "message": (
                    "This query is outside the BIS building-material standards context."
                ),
            }
        candidates = self.retriever.retrieve(state["query"])
        if not self.retriever.is_relevant(candidates):
            return {
                "candidates": candidates,
                "out_of_scope": True,
                "message": (
                    "This query does not appear to describe a building material "
                    "or BIS standard from the provided dataset."
                ),
            }
        return {"candidates": candidates, "out_of_scope": False, "message": ""}

    def _check_scope(self, state: AgentState) -> AgentState:
        if state.get("out_of_scope") or self.reranker is None:
            return {}
        try:
            in_scope, reason = self.reranker.is_query_in_scope(state["query"], state["candidates"])
        except Exception:
            return {}
        if in_scope:
            return {}
        return {
            "out_of_scope": True,
            "message": reason
            or "This query is outside the BIS building-material standards dataset.",
        }

    def _rank_with_gemini(self, state: AgentState) -> AgentState:
        if state.get("out_of_scope"):
            return {"retrieved_standards": []}
        candidates = state["candidates"]
        fallback = [candidate.standard_id for candidate in candidates[: self.top_k]]
        if self.reranker is None:
            return {"retrieved_standards": fallback}
        try:
            ranked = self.reranker.rerank(state["query"], candidates, self.top_k)
            return {"retrieved_standards": ranked}
        except Exception:
            return {"retrieved_standards": fallback}

    def _finish(self, state: AgentState) -> AgentState:
        latency = time.perf_counter() - state["started_at"]
        return {"latency_seconds": round(latency, 3)}
