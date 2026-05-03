"""Judge entrypoint for the BIS Standards Recommendation Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent import BISRecommendationAgent
from src.config import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BIS standards inference.")
    parser.add_argument("--input", required=True, help="Path to input JSON file.")
    parser.add_argument("--output", required=True, help="Path to output JSON file.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of standards to return for each query. Default: 5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as file:
        items = json.load(file)

    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list of query objects.")

    settings = Settings.from_env()
    agent = BISRecommendationAgent(settings=settings, top_k=args.top_k)

    results = []
    for item in items:
        query_id = item.get("id")
        query = item.get("query")
        if not query_id or not query:
            raise ValueError("Every input item must contain 'id' and 'query'.")

        result = agent.recommend(query_id=str(query_id), query=str(query))
        output_item = {
            "id": result["id"],
            "query": str(query),
        }
        if "expected_standards" in item:
            output_item["expected_standards"] = item["expected_standards"]
        output_item["retrieved_standards"] = result["retrieved_standards"]
        output_item["latency_seconds"] = result["latency_seconds"]
        if result.get("out_of_scope"):
            output_item["out_of_scope"] = True
            output_item["message"] = result.get("message", "")
        results.append(output_item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
