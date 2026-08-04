#!/usr/bin/env python3
"""Pick epochs by recall, not mAP, then score the per-class threshold grid.

The per-category configs state the rule plainly:

    primary_metric: test/recall
    Selection must follow the established downstream procedure, because the
    automatic checkpoint_best_total.pth is chosen by mAP, not recall-first.

On 2026-08-04 every experiment was scored after picking epochs by
``val/mAP_50``, which is exactly the metric the config warns against. The two
orderings disagree substantially - on the ctrl80 run the best-recall epochs are
56/55/29 while the best-mAP epochs are 50/51/66, and the recall at those epochs
differs by more than the noise band. Any comparison against the delivered
baseline, which was selected recall-first, was therefore biased against the new
runs by an unknown amount.

This script applies the documented procedure end to end so the mistake is not
repeatable: read ``val/recall`` per epoch, take the top-k, score each on the
frozen test split over the per-class threshold grid at match IoU 0.229, and
report the best operating point subject to a precision floor.

Note on the protocol's own optimism: ``valid`` is a byte-identical copy of
``test`` (lock file ``valid_mirrors_test: true``), so epoch selection, checkpoint
selection and threshold selection all happen on the same 72-box split. The
delivered numbers carry the same optimism, which keeps like-for-like comparisons
valid, but no number produced here is a generalisation estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="training output dir with metrics.csv")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--metric", default="val/recall",
                   help="per-epoch column to rank by; the config mandates recall")
    p.add_argument("--precision-floor", type=float, default=0.60)
    p.add_argument("--threshold-grid", default="0.20,0.25,0.30,0.35,0.40")
    p.add_argument("--iou-threshold", type=float, default=0.229)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--skip-existing", action="store_true", default=True)
    return p.parse_args()


def top_epochs(run: Path, metric: str, k: int) -> list[tuple[int, float]]:
    rows: list[tuple[float, int]] = []
    mfile = run / "metrics.csv"
    if not mfile.exists():
        return []
    for r in csv.DictReader(mfile.open(encoding="utf-8")):
        raw = (r.get(metric) or "").strip()
        if not raw:
            continue
        try:
            rows.append((float(raw), int(float(r["epoch"]))))
        except (ValueError, KeyError):
            continue
    seen, out = set(), []
    for v, e in sorted(rows, reverse=True):
        if e in seen:
            continue
        seen.add(e)
        out.append((e, v))
        if len(out) >= k:
            break
    return out


def score(csv_path: Path, floor: float) -> dict | None:
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    ok = [r for r in rows if float(r["precision"]) >= floor]
    if not ok:
        best_f1 = max(rows, key=lambda r: float(r["f1"])) if rows else None
        return {"feasible": False,
                "best_f1": float(best_f1["f1"]) if best_f1 else None,
                "best_f1_recall": float(best_f1["recall"]) if best_f1 else None,
                "best_f1_precision": float(best_f1["precision"]) if best_f1 else None}
    b = max(ok, key=lambda r: float(r["recall"]))
    return {"feasible": True, "recall": float(b["recall"]), "precision": float(b["precision"]),
            "f1": float(b["f1"]), "thresholds": b["thresholds"]}


def main() -> int:
    args = parse_args()
    run = Path(args.run_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or run.name

    eps = top_epochs(run, args.metric, args.topk)
    if not eps:
        print(f"{label}: no usable '{args.metric}' column in {run}/metrics.csv")
        return 1
    print(f"{label}: top epochs by {args.metric} -> " +
          ", ".join(f"ep{e:03d}({v:.3f})" for e, v in eps))

    results = []
    for ep, val in eps:
        ck = run / "epoch_pth" / f"checkpoint_epoch_{ep:03d}.pth"
        if not ck.exists():
            print(f"  ep{ep:03d}: checkpoint missing")
            continue
        csv_path = out / f"{label}_ep{ep:03d}.csv"
        if not (args.skip_existing and csv_path.exists()):
            cmd = [sys.executable, str(REPO / "scripts" / "evaluate_rfdetr_class_threshold_grid.py"),
                   "--checkpoint", str(ck), "--dataset-dir", args.dataset_dir, "--split", "test",
                   "--threshold-grid", args.threshold_grid,
                   "--iou-threshold", str(args.iou_threshold), "--num-classes", "3",
                   "--device", args.device, "--output-csv", str(csv_path)]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
            if r.returncode != 0:
                print(f"  ep{ep:03d}: eval failed ({r.stderr.strip().splitlines()[-1] if r.stderr else '?'})")
                continue
        s = score(csv_path, args.precision_floor)
        if s is None:
            continue
        s.update({"epoch": ep, f"selection_{args.metric}": val})
        results.append(s)
        if s["feasible"]:
            print(f"  ep{ep:03d}: R={s['recall']:.3f} P={s['precision']:.3f} "
                  f"F1={s['f1']:.3f} @ thr {s['thresholds']}")
        else:
            print(f"  ep{ep:03d}: no point with P>={args.precision_floor}; "
                  f"best F1 {s['best_f1']:.3f} (R={s['best_f1_recall']:.3f}/P={s['best_f1_precision']:.3f})")

    feas = [r for r in results if r["feasible"]]
    summary = {"label": label, "run_dir": str(run), "metric": args.metric,
               "precision_floor": args.precision_floor, "results": results}
    if feas:
        best = max(feas, key=lambda r: r["recall"])
        summary["best"] = best
        print(f"  ★ {label}: R={best['recall']:.3f} P={best['precision']:.3f} @ ep{best['epoch']:03d}")
    else:
        print(f"  ★ {label}: no operating point at P>={args.precision_floor}")
    (out / f"{label}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
