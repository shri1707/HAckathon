"""Small local evaluator for public test-set style files.

Usage:
    python eval_script.py --expected public_test_set.json --predictions team_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BIS recommendation output.")
    parser.add_argument("--expected", required=True, help="JSON file with expected_standards.")
    parser.add_argument("--predictions", required=True, help="JSON output from inference.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))

    expected_by_id = {item["id"]: item.get("expected_standards", []) for item in expected}
    pred_by_id = {item["id"]: item for item in predictions}

    hits_at_3 = 0
    reciprocal_sum = 0.0
    latencies = []

    for query_id, expected_standards in expected_by_id.items():
        predicted = pred_by_id.get(query_id, {})
        retrieved = predicted.get("retrieved_standards", [])
        latencies.append(float(predicted.get("latency_seconds", 0.0)))

        expected_set = set(expected_standards)
        if any(standard in expected_set for standard in retrieved[:3]):
            hits_at_3 += 1

        rank = next(
            (index + 1 for index, standard in enumerate(retrieved[:5]) if standard in expected_set),
            None,
        )
        if rank:
            reciprocal_sum += 1.0 / rank

    total = max(len(expected_by_id), 1)
    print(f"Hit Rate @3: {(hits_at_3 / total) * 100:.2f}%")
    print(f"MRR @5: {reciprocal_sum / total:.4f}")
    print(f"Avg Latency: {sum(latencies) / max(len(latencies), 1):.3f}s")


if __name__ == "__main__":
    main()
