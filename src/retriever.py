"""Candidate retrieval and score aggregation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .index import VectorIndex
from .text_utils import compact_for_prompt


@dataclass
class Candidate:
    standard_id: str
    score: float
    evidence: str
    page_number: int


DOMAIN_TERMS = {
    "aggregate",
    "aggregates",
    "asbestos",
    "bitumen",
    "block",
    "blocks",
    "brick",
    "building",
    "cement",
    "cladding",
    "concrete",
    "construction",
    "fine",
    "fitting",
    "fittings",
    "flooring",
    "gypsum",
    "lime",
    "masonry",
    "material",
    "materials",
    "mortar",
    "pipe",
    "pipes",
    "plaster",
    "portland",
    "precast",
    "reinforcement",
    "roof",
    "roofing",
    "sand",
    "sheet",
    "sheets",
    "slag",
    "standard",
    "standards",
    "steel",
    "stone",
    "structural",
    "timber",
    "waterproofing",
    "wood",
}

GENERAL_KNOWLEDGE_TERMS = {
    "capital",
    "colour",
    "color",
    "india",
    "minister",
    "president",
    "prime",
    "sky",
    "weather",
}


def query_terms(query: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", query.lower()))


def looks_like_domain_query(query: str) -> bool:
    terms = query_terms(query)
    if not terms:
        return False
    if terms & DOMAIN_TERMS:
        return True
    if terms & GENERAL_KNOWLEDGE_TERMS:
        return False
    return len(terms) >= 5


def keyword_overlap(query: str, evidence: str) -> float:
    terms = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
    if not terms:
        return 0.0
    evidence_terms = set(re.findall(r"[a-z0-9]{3,}", evidence.lower()))
    return len(terms & evidence_terms) / math.sqrt(len(terms))


class Retriever:
    MIN_RELEVANCE_SCORE = 0.45

    def __init__(self, index: VectorIndex) -> None:
        self.index = index

    def retrieve(self, query: str, candidate_count: int = 12) -> list[Candidate]:
        hits = self.index.search(query, top_n=36)
        by_standard: dict[str, Candidate] = {}

        for chunk, vector_score in hits:
            overlap = keyword_overlap(query, chunk.text)
            combined_score = vector_score + (0.08 * overlap)
            evidence = compact_for_prompt(chunk.text)
            for standard_id in chunk.standard_ids:
                existing = by_standard.get(standard_id)
                if existing is None or combined_score > existing.score:
                    by_standard[standard_id] = Candidate(
                        standard_id=standard_id,
                        score=combined_score,
                        evidence=evidence,
                        page_number=chunk.page_number,
                    )

        return sorted(by_standard.values(), key=lambda item: item.score, reverse=True)[:candidate_count]

    def is_relevant(self, candidates: list[Candidate]) -> bool:
        if not candidates:
            return False
        return candidates[0].score >= self.MIN_RELEVANCE_SCORE
