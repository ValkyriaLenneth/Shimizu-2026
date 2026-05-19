#!/usr/bin/env python3
"""Batch smoke test that does not require real crack labels."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    python = repo / "coarse_router_yolov9" / ".venv" / "bin" / "python"
    if not python.exists():
        python = Path(sys.executable)

    source = repo / "data" / "unzip" / "3_RC壁" / "obj_train_data"
    out = repo / "outputs" / "pipeline" / "batch_smoke_mock"
    run(
        [
            str(python),
            "-m",
            "router_crack_pipeline.pipeline.run_full_pipeline",
            "--config",
            "router_crack_pipeline/configs/pipeline.default.yaml",
            "--source",
            str(source),
            "--output-dir",
            str(out),
            "--device",
            "cpu",
            "--mock-crack",
            "--limit",
            "3",
            "--skip-visualization",
        ],
        repo,
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["images"] == 3, summary
    assert summary["error_count"] == 0, summary
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    run([str(python), "router_crack_pipeline/scripts/summarize_pipeline_results.py", str(out / "results.jsonl")], repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

