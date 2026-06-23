#!/usr/bin/env python3
"""Summarize pipeline JSONL outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.results_jsonl)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    router_status = Counter()
    router_classes = Counter()
    crack_models = Counter()
    crack_grades = Counter()
    warnings = Counter()
    elapsed = []
    errors = 0
    for row in rows:
        if row.get("error"):
            errors += 1
        if row.get("elapsed_ms") is not None:
            elapsed.append(float(row["elapsed_ms"]))
        router = row.get("router") or {}
        router_status[router.get("route_decision", {}).get("status", "error")] += 1
        for det in router.get("detections", []):
            router_classes[det.get("class_name", "unknown")] += 1
        for det in row.get("crack_detections", []):
            crack_models[det.get("source_model", "unknown")] += 1
            crack_grades[det.get("damage_grade", "unknown")] += 1
        warnings.update(row.get("warnings", []))

    summary = {
        "results_jsonl": str(path),
        "images": len(rows),
        "errors": errors,
        "router_status": dict(router_status),
        "router_classes": dict(router_classes),
        "crack_models": dict(crack_models),
        "crack_grades": dict(crack_grades),
        "warnings": dict(warnings),
        "elapsed_ms_avg": round(mean(elapsed), 3) if elapsed else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

