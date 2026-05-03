"""Gemini reranking client."""

from __future__ import annotations

import json
import re

from google import genai

from .retriever import Candidate


class GeminiReranker:
    def __init__(self, api_key: str, model_name: str) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def is_query_in_scope(self, query: str, candidates: list[Candidate]) -> tuple[bool, str]:
        candidate_payload = [
            {
                "standard_id": candidate.standard_id,
                "score": round(candidate.score, 4),
                "evidence": candidate.evidence,
            }
            for candidate in candidates[:5]
        ]
        prompt = (
            "Classify whether the user query is asking for a Bureau of Indian Standards "
            "(BIS) standard related to building materials, construction materials, cement, "
            "concrete, aggregates, steel, pipes, blocks, sheets, fittings, or similar products "
            "covered by the provided dataset candidates. "
            "Return strict JSON with keys in_scope and reason. "
            "Set in_scope to false for general knowledge, politics, weather, sky color, people, "
            "places, or anything not asking for a BIS/building-material standard. "
            "Do not classify as in scope merely because some candidate text has weak word overlap.\n\n"
            f"Query: {query}\n\n"
            f"Top retrieved candidates:\n{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}"
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        )
        data = self._parse_json(response.text or "")
        return bool(data.get("in_scope")), str(data.get("reason", ""))

    def rerank(self, query: str, candidates: list[Candidate], top_k: int) -> list[str]:
        candidate_payload = [
            {
                "standard_id": candidate.standard_id,
                "score": round(candidate.score, 4),
                "page": candidate.page_number,
                "evidence": candidate.evidence,
            }
            for candidate in candidates
        ]
        prompt = (
            "You are selecting Bureau of Indian Standards (BIS) standards for a product query. "
            "Choose only from the provided candidate standard_id values. "
            "Return strict JSON with one key named retrieved_standards, whose value is an array "
            f"of exactly {top_k} standard_id strings ranked from most to least relevant.\n\n"
            f"Query: {query}\n\n"
            f"Candidates:\n{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        )
        text = response.text or ""
        data = self._parse_json(text)
        allowed = [candidate.standard_id for candidate in candidates]
        allowed_set = set(allowed)
        chosen = [
            item for item in data.get("retrieved_standards", [])
            if isinstance(item, str) and item in allowed_set
        ]
        for standard_id in allowed:
            if len(chosen) >= top_k:
                break
            if standard_id not in chosen:
                chosen.append(standard_id)
        return chosen[:top_k]

    @staticmethod
    def _parse_json(text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise
