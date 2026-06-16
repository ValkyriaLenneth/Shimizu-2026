#!/usr/bin/env python3
"""Analyze RF-DETR router per-query class ambiguity against GT."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from rfdetr_prod_pipeline.pipeline.rfdetr_backend import RfdetrBackend
from rfdetr_prod_pipeline.pipeline.result_merge import iou_xyxy


CLASS_NAMES = {0: "天井", 1: "壁类", 2: "RC柱"}
COMPONENT_TO_ROUTER = {"tenjo": "天井", "inner_wall": "壁类", "rc_wall": "壁类", "rc_column": "RC柱"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", default="data/pipeline_eval_official_plus_20260615/split.json")
    parser.add_argument("--checkpoint", default="final_release_20260615/models/rfdetr/router/checkpoint_epoch_023.pth")
    parser.add_argument("--output-dir", default="outputs/rfdetr_prod_pipeline/router_per_query_ambiguity_20260616")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--top2-threshold", type=float, default=0.25)
    parser.add_argument("--margin-threshold", type=float, default=0.10)
    parser.add_argument("--ratio-threshold", type=float, default=0.85)
    parser.add_argument("--gt-iou-threshold", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = json.loads(Path(args.split_json).read_text(encoding="utf-8"))["samples"]
    backend = RfdetrBackend(args.checkpoint, CLASS_NAMES, device=args.device)
    model = backend.model
    model.model.model.to(model.model.device)
    model.model.model.eval()

    per_query_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        image_path = Path(sample["eval_image"])
        label_path = Path(sample["eval_label"])
        component = sample["component"]
        expected_class = COMPONENT_TO_ROUTER[component]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        h, w = image_bgr.shape[:2]
        gt_boxes = load_gt_boxes(label_path, w, h)
        queries = raw_router_queries(model, image_bgr)
        candidate_rows = []
        for query_index, query in enumerate(queries):
            top1_id = int(query["top_ids"][0])
            top2_id = int(query["top_ids"][1])
            top1_score = float(query["top_scores"][0])
            top2_score = float(query["top_scores"][1])
            if top1_score < args.score_threshold:
                continue
            box = query["xyxy"]
            best_gt_iou = max((iou_xyxy(tuple(gt), tuple(box)) for gt in gt_boxes), default=0.0)
            expected_score = float(query["scores"][router_id(expected_class)])
            expected_rank = rank_of(query["scores"], router_id(expected_class))
            ambiguous = (
                top2_score >= args.top2_threshold
                and (
                    top1_score - top2_score <= args.margin_threshold
                    or top2_score / max(top1_score, 1e-9) >= args.ratio_threshold
                )
            )
            expected_in_top2 = router_id(expected_class) in {top1_id, top2_id}
            row = {
                "image": str(image_path),
                "component": component,
                "expected_class": expected_class,
                "query_index": query_index,
                "bbox_xyxy": json.dumps([round(v, 3) for v in box]),
                "top1_class": CLASS_NAMES[top1_id],
                "top1_score": round(top1_score, 6),
                "top2_class": CLASS_NAMES[top2_id],
                "top2_score": round(top2_score, 6),
                "margin": round(top1_score - top2_score, 6),
                "ratio": round(top2_score / max(top1_score, 1e-9), 6),
                "expected_score": round(expected_score, 6),
                "expected_rank": expected_rank,
                "expected_in_top2": int(expected_in_top2),
                "ambiguous": int(ambiguous),
                "best_gt_iou": round(best_gt_iou, 6),
                "gt_hit": int(best_gt_iou >= args.gt_iou_threshold),
            }
            per_query_rows.append(row)
            candidate_rows.append(row)

        top1_hits = [r for r in candidate_rows if r["gt_hit"] and r["top1_class"] == expected_class]
        top2_rescue_hits = [
            r
            for r in candidate_rows
            if r["gt_hit"]
            and r["top1_class"] != expected_class
            and r["top2_class"] == expected_class
            and r["ambiguous"]
        ]
        expected_any_hits = [
            r
            for r in candidate_rows
            if r["gt_hit"] and r["expected_score"] >= args.top2_threshold and r["expected_rank"] <= 2
        ]
        ambiguous_queries = [r for r in candidate_rows if r["ambiguous"]]
        per_image_rows.append(
            {
                "image": str(image_path),
                "component": component,
                "expected_class": expected_class,
                "queries_kept": len(candidate_rows),
                "ambiguous_queries": len(ambiguous_queries),
                "has_ambiguous_query": int(bool(ambiguous_queries)),
                "top1_gt_hit": int(bool(top1_hits)),
                "top2_rescue_gt_hit": int(bool(top2_rescue_hits)),
                "expected_top2_gt_hit": int(bool(expected_any_hits)),
                "best_gt_iou_top1": max((float(r["best_gt_iou"]) for r in top1_hits), default=0.0),
                "best_gt_iou_top2_rescue": max((float(r["best_gt_iou"]) for r in top2_rescue_hits), default=0.0),
            }
        )
        if index % 25 == 0:
            print(f"processed {index}/{len(samples)}")

    write_csv(out_dir / "per_query_ambiguity.csv", per_query_rows)
    write_csv(out_dir / "per_image_ambiguity.csv", per_image_rows)
    summary = summarize(per_query_rows, per_image_rows, args)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def raw_router_queries(model: Any, image_bgr: np.ndarray) -> list[dict[str, Any]]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb)
    img = TF.to_tensor(pil)
    h, w = img.shape[1:]
    img = img.to(model.model.device)
    img = TF.resize(img, [model.model.resolution, model.model.resolution])
    img = TF.normalize(img, model.means, model.stds)
    batch = torch.stack([img])
    with torch.no_grad():
        outputs = model.model.model(batch)
    if isinstance(outputs, tuple):
        outputs = {"pred_boxes": outputs[0], "pred_logits": outputs[1]}
    logits = outputs["pred_logits"][0]
    boxes = outputs["pred_boxes"][0]
    scores = logits.sigmoid()[:, : len(CLASS_NAMES)]
    xyxy = cxcywh_to_xyxy(boxes).detach().cpu().numpy()
    xyxy[:, [0, 2]] *= w
    xyxy[:, [1, 3]] *= h
    scores_np = scores.detach().cpu().numpy()
    top_ids = np.argsort(-scores_np, axis=1)[:, :2]
    out = []
    for i in range(scores_np.shape[0]):
        out.append(
            {
                "xyxy": [float(v) for v in xyxy[i]],
                "scores": [float(v) for v in scores_np[i]],
                "top_ids": [int(v) for v in top_ids[i]],
                "top_scores": [float(scores_np[i, top_ids[i, 0]]), float(scores_np[i, top_ids[i, 1]])],
            }
        )
    return out


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, bw, bh = boxes.unbind(-1)
    return torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)


def load_gt_boxes(path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = parts[:5]
        cx_f, cy_f, bw_f, bh_f = map(float, (cx, cy, bw, bh))
        boxes.append(
            [
                (cx_f - bw_f / 2) * width,
                (cy_f - bh_f / 2) * height,
                (cx_f + bw_f / 2) * width,
                (cy_f + bh_f / 2) * height,
            ]
        )
    return boxes


def router_id(name: str) -> int:
    for idx, class_name in CLASS_NAMES.items():
        if class_name == name:
            return idx
    raise KeyError(name)


def rank_of(scores: list[float], class_id: int) -> int:
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    return order.index(class_id) + 1


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(per_query_rows: list[dict[str, Any]], per_image_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    query_count = len(per_query_rows)
    ambiguous = [r for r in per_query_rows if int(r["ambiguous"])]
    image_count = len(per_image_rows)
    by_component = defaultdict(lambda: Counter())
    for row in per_image_rows:
        c = by_component[row["component"]]
        c["images"] += 1
        c["has_ambiguous_query"] += int(row["has_ambiguous_query"])
        c["top1_gt_hit"] += int(row["top1_gt_hit"])
        c["top2_rescue_gt_hit"] += int(row["top2_rescue_gt_hit"])
        c["expected_top2_gt_hit"] += int(row["expected_top2_gt_hit"])
    return {
        "params": {
            "score_threshold": args.score_threshold,
            "top2_threshold": args.top2_threshold,
            "margin_threshold": args.margin_threshold,
            "ratio_threshold": args.ratio_threshold,
            "gt_iou_threshold": args.gt_iou_threshold,
        },
        "images": image_count,
        "queries_kept": query_count,
        "ambiguous_queries": len(ambiguous),
        "images_with_ambiguous_query": sum(int(r["has_ambiguous_query"]) for r in per_image_rows),
        "images_top1_gt_hit": sum(int(r["top1_gt_hit"]) for r in per_image_rows),
        "images_top2_rescue_gt_hit": sum(int(r["top2_rescue_gt_hit"]) for r in per_image_rows),
        "images_expected_top2_gt_hit": sum(int(r["expected_top2_gt_hit"]) for r in per_image_rows),
        "by_component": {k: dict(v) for k, v in by_component.items()},
    }


if __name__ == "__main__":
    main()
