#!/usr/bin/env python3
"""Smoke tests for the router + crack pipeline."""

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

    sample = repo / "data" / "rfdetr_rc_wall_all_non_legacy_test_v1" / "test" / "images" / "data_add100__3-B-00009.jpg"
    if not sample.exists():
        raise FileNotFoundError(sample)

    out = repo / "outputs" / "rfdetr_prod_pipeline" / "smoke_test_script"
    run(
        [
            str(python),
            "-m",
            "rfdetr_prod_pipeline.pipeline.run_full_pipeline",
            "--config",
            "rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.local.yaml",
            "--source",
            str(sample),
            "--output-dir",
            str(out),
            "--device",
            "cpu",
            "--mock-crack",
        ],
        repo,
    )

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["images"] == 1, summary
    assert summary["router_status_counts"].get("ok", 0) == 1, summary
    assert summary["crack_detections"] >= 1, summary
    assert (out / "visualizations" / f"{sample.stem}_pipeline.jpg").exists()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
