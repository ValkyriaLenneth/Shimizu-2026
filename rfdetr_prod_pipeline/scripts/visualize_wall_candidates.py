#!/usr/bin/env python3
"""Create a Japanese HTML review page for wall subtype candidates."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any

import cv2


STATUS_LABELS = {
    "same_grade_merged": "同一レベルのため1件に統合",
    "grade_conflict": "レベル差あり：候補を併記",
    "single_model": "片方のモデルのみ検出",
}

GRADE_LABELS = {
    "B": "B",
    "C": "C",
    "D": "D",
}

GT_CLASS_TO_GRADE = {
    0: "B",
    1: "C",
    2: "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_jsonl")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_path = Path(args.results_jsonl).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else results_path.parent / "wall_candidate_review"
    asset_dir = out_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cards = []
    status_counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        groups = (row.get("wall_candidate_display") or {}).get("groups") or []
        if not groups:
            continue
        for group in groups:
            status = str(group.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1

        image_path = Path(row["image"])
        image_info = prepare_asset(image_path, asset_dir, index)
        if image_info is None:
            continue
        cards.append(render_card(row, image_info))

    html_text = render_page(results_path, len(rows), len(cards), status_counts, cards)
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")
    summary = {
        "results_jsonl": str(results_path),
        "images_total": len(rows),
        "images_with_wall_candidates": len(cards),
        "status_counts": status_counts,
        "html": str(out_dir / "index.html"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def prepare_asset(image_path: Path, asset_dir: Path, index: int) -> dict[str, Any] | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    h, w = image.shape[:2]
    copied = asset_dir / f"{index:04d}_{safe_name(image_path.name)}"
    shutil.copy2(image_path, copied)
    return {
        "path": image_path,
        "relative": Path("assets") / copied.name,
        "width": w,
        "height": h,
    }


def render_card(row: dict[str, Any], image_info: dict[str, Any]) -> str:
    groups = (row.get("wall_candidate_display") or {}).get("groups") or []
    status = "grade_conflict" if any(g.get("status") == "grade_conflict" for g in groups) else str(groups[0].get("status", "single_model"))
    overlays = render_overlays(row, image_info)
    group_blocks = "".join(render_group(group) for group in groups)
    filename = Path(row.get("image", "")).name
    return f"""
    <article class="card {html.escape(status)}" data-status="{html.escape(status)}">
      <div class="media">
        <div class="image-stage" style="aspect-ratio: {image_info['width']} / {image_info['height']};">
          <img src="{html.escape(str(image_info['relative']))}" loading="lazy" alt="">
          {overlays}
        </div>
      </div>
      <div class="body">
        <div class="topline">
          <span class="badge">{html.escape(STATUS_LABELS.get(status, status))}</span>
          <span class="count-label">1枚の写真内で候補を表示</span>
        </div>
        <h2>{html.escape(filename)}</h2>
        <p class="path">{html.escape(row.get("image", ""))}</p>
        <div class="groups">{group_blocks}</div>
      </div>
    </article>
    """


def render_overlays(row: dict[str, Any], image_info: dict[str, Any]) -> str:
    w, h = float(image_info["width"]), float(image_info["height"])
    layers = [
        ("auto-layer", render_auto_boxes(row, w, h)),
        ("result-layer", render_result_boxes(row, w, h)),
        ("gt-layer", render_gt_boxes(Path(row["image"]), w, h)),
    ]
    return "".join(f'<div class="overlay {name}">{content}</div>' for name, content in layers)


def render_auto_boxes(row: dict[str, Any], w: float, h: float) -> str:
    primary_class = str(((row.get("router") or {}).get("route_decision") or {}).get("primary_class") or "")
    detections = (row.get("router") or {}).get("detections", [])
    if primary_class:
        primary_detections = [det for det in detections if str(det.get("class_name", "")) == primary_class]
        detections = primary_detections or detections[:1]
    boxes = []
    for det in detections:
        class_name = str(det.get("class_name", ""))
        label = f"自動識別 {class_name} {float(det.get('confidence') or 0.0):.2f}"
        boxes.append(render_box(det.get("bbox_xyxy", [0, 0, 0, 0]), w, h, f"auto-box auto-{auto_class_key(class_name)}", label))
    return "".join(boxes)


def render_result_boxes(row: dict[str, Any], w: float, h: float) -> str:
    boxes = []
    for det in row.get("display_crack_detections", []):
        if det.get("source_router_class") != "壁类":
            continue
        grade = str(det.get("damage_grade", ""))
        structure = str(det.get("structure_type", "壁類"))
        prefix = "候補" if det.get("status") == "grade_conflict" else "判定"
        label = f"{prefix} {structure} {GRADE_LABELS.get(grade, grade)} {float(det.get('confidence') or 0.0):.2f}"
        boxes.append(render_box(det.get("bbox_xyxy", [0, 0, 0, 0]), w, h, f"result-box result-{result_class_key(structure)} grade-{grade}", label))
    return "".join(boxes)


def render_gt_boxes(image_path: Path, w: float, h: float) -> str:
    label_path = infer_label_path(image_path)
    if label_path is None or not label_path.exists():
        return ""
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xc, yc, bw, bh = [float(v) for v in parts]
        x1 = (xc - bw / 2.0) * w
        y1 = (yc - bh / 2.0) * h
        x2 = (xc + bw / 2.0) * w
        y2 = (yc + bh / 2.0) * h
        grade = GT_CLASS_TO_GRADE.get(int(cls), str(int(cls)))
        boxes.append(render_box([x1, y1, x2, y2], w, h, "gt-box", f"正解 {grade}"))
    return "".join(boxes)


def infer_label_path(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    if "images" not in parts:
        return None
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    label_path = Path(*parts).with_suffix(".txt")
    return label_path


def auto_class_key(class_name: str) -> str:
    if class_name == "天井":
        return "ceiling"
    if class_name == "RC柱":
        return "column"
    return "wall"


def result_class_key(structure_type: str) -> str:
    if structure_type == "RC壁":
        return "rc-wall"
    if structure_type == "内壁":
        return "inner-wall"
    return "wall"


def render_box(box: list[Any], w: float, h: float, css_class: str, label: str) -> str:
    x1, y1, x2, y2 = [float(v) for v in box]
    left = clamp_percent(x1 / w * 100.0)
    top = clamp_percent(y1 / h * 100.0)
    width = clamp_percent((x2 - x1) / w * 100.0)
    height = clamp_percent((y2 - y1) / h * 100.0)
    return (
        f'<div class="box {html.escape(css_class)}" '
        f'style="left:{left:.4f}%;top:{top:.4f}%;width:{width:.4f}%;height:{height:.4f}%;">'
        f'<span>{html.escape(label)}</span></div>'
    )


def render_group(group: dict[str, Any]) -> str:
    status = str(group.get("status", ""))
    chips = []
    for candidate in group.get("candidates") or []:
        chips.append(
            "<span class=\"candidate\">"
            f"{html.escape(str(candidate.get('structure_type', '')))} "
            f"<strong>{html.escape(str(candidate.get('damage_grade', '')))}</strong> "
            f"{float(candidate.get('confidence') or 0.0):.2f}"
            "</span>"
        )
    reason = japanese_reason(status)
    return f"""
      <section class="group">
        <div class="group-head">
          <span>{html.escape(STATUS_LABELS.get(status, status))}</span>
          <small>候補領域 {group.get("group_index", "")}</small>
        </div>
        <div class="candidate-row">{''.join(chips)}</div>
        <p>{html.escape(reason)}</p>
      </section>
    """


def japanese_reason(status: str) -> str:
    if status == "same_grade_merged":
        return "内壁モデルとRC壁モデルが同じ損傷レベルを出したため、確認画面では1件に統合しています。"
    if status == "grade_conflict":
        return "内壁モデルとRC壁モデルで損傷レベルが異なるため、同じ写真内に2つの候補として表示しています。"
    if status == "single_model":
        return "片方の壁モデルのみが該当箇所を検出したため、単独候補として表示しています。"
    return "壁類の候補判定です。"


def render_page(results_path: Path, total: int, card_count: int, status_counts: dict[str, int], cards: list[str]) -> str:
    counts = " / ".join(f"{STATUS_LABELS.get(k, k)}: {v}" for k, v in sorted(status_counts.items()))
    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>壁類候補レビュー</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --text: #18202a;
      --muted: #647080;
      --line: #d9dee6;
      --panel: #ffffff;
      --conflict: #b42318;
      --merged: #13795b;
      --single: #3d5a80;
      --auto: #00a884;
      --auto-ceiling: #8b5cf6;
      --auto-wall: #00a884;
      --auto-column: #f59e0b;
      --inner-wall: #2563eb;
      --rc-wall: #e11d48;
      --gt: #7c3aed;
      --b: #198754;
      --c: #0d6efd;
      --d: #dc3545;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: system-ui, -apple-system, BlinkMacSystemFont, "Hiragino Sans", "Yu Gothic", "Meiryo", "Noto Sans CJK JP", sans-serif; }}
    header {{ position: sticky; top: 0; z-index: 10; background: rgba(246,247,249,.95); border-bottom: 1px solid var(--line); backdrop-filter: blur(10px); }}
    .header-inner {{ max-width: 1440px; margin: 0 auto; padding: 18px 24px; display: grid; grid-template-columns: 1fr auto; gap: 18px; align-items: center; }}
    h1 {{ margin: 0; font-size: 22px; letter-spacing: 0; }}
    .summary {{ color: var(--muted); font-size: 14px; margin-top: 4px; }}
    .controls {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    button, label.toggle {{ border: 1px solid var(--line); background: var(--panel); color: var(--text); border-radius: 6px; padding: 8px 12px; cursor: pointer; font-size: 14px; }}
    button.active {{ border-color: #1f6feb; color: #1f6feb; }}
    label.toggle {{ display: inline-flex; align-items: center; gap: 6px; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 22px 24px 48px; display: grid; gap: 18px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-left-width: 5px; border-radius: 8px; display: grid; grid-template-columns: minmax(420px, 58%) 1fr; overflow: hidden; box-shadow: 0 10px 24px rgba(24,32,42,.06); }}
    .card.grade_conflict {{ border-left-color: var(--conflict); }}
    .card.same_grade_merged {{ border-left-color: var(--merged); }}
    .card.single_model {{ border-left-color: var(--single); }}
    .media {{ background: #151b23; min-height: 360px; padding: 12px; display: flex; align-items: center; justify-content: center; }}
    .image-stage {{ position: relative; width: 100%; max-height: 78vh; }}
    .image-stage img {{ width: 100%; height: 100%; object-fit: contain; display: block; }}
    .overlay {{ position: absolute; inset: 0; pointer-events: none; }}
    .box {{ position: absolute; border: 3px solid; border-radius: 2px; min-width: 12px; min-height: 12px; }}
    .box span {{ position: absolute; left: -3px; top: -24px; color: white; font-size: 11px; line-height: 1.25; font-weight: 700; padding: 3px 6px; border-radius: 4px; white-space: nowrap; text-shadow: none; transition: transform .12s ease; }}
    .auto-box {{ border-style: dashed; }}
    .auto-ceiling {{ border-color: var(--auto-ceiling); }}
    .auto-ceiling span {{ background: var(--auto-ceiling); }}
    .auto-wall {{ border-color: var(--auto-wall); }}
    .auto-wall span {{ background: var(--auto-wall); }}
    .auto-column {{ border-color: var(--auto-column); }}
    .auto-column span {{ background: var(--auto-column); }}
    .result-box {{ border-width: 4px; }}
    .result-box span {{ top: auto; bottom: -25px; }}
    .result-inner-wall {{ border-style: solid; }}
    .result-rc-wall {{ border-style: double; }}
    .result-inner-wall.grade-B {{ border-color: #198754; }}
    .result-inner-wall.grade-B span {{ background: #198754; }}
    .result-inner-wall.grade-C {{ border-color: #2563eb; }}
    .result-inner-wall.grade-C span {{ background: #2563eb; }}
    .result-inner-wall.grade-D {{ border-color: #1d4ed8; }}
    .result-inner-wall.grade-D span {{ background: #1d4ed8; }}
    .result-rc-wall.grade-B {{ border-color: #f97316; }}
    .result-rc-wall.grade-B span {{ background: #f97316; }}
    .result-rc-wall.grade-C {{ border-color: #e11d48; }}
    .result-rc-wall.grade-C span {{ background: #e11d48; }}
    .result-rc-wall.grade-D {{ border-color: #991b1b; }}
    .result-rc-wall.grade-D span {{ background: #991b1b; }}
    .gt-box {{ border-color: var(--gt); border-style: dotted; }}
    .gt-box span {{ background: var(--gt); }}
    body.hide-auto .auto-layer, body.hide-result .result-layer, body.hide-gt .gt-layer {{ display: none; }}
    .body {{ padding: 18px; min-width: 0; }}
    .topline {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
    .badge {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 5px 10px; background: #eef2f7; font-size: 13px; font-weight: 700; }}
    .count-label {{ color: var(--muted); font-size: 13px; }}
    h2 {{ margin: 12px 0 4px; font-size: 18px; overflow-wrap: anywhere; }}
    .path {{ margin: 0 0 14px; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    .groups {{ display: grid; gap: 12px; }}
    .group {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; }}
    .group-head {{ display: flex; justify-content: space-between; gap: 12px; font-weight: 700; }}
    .group-head small {{ color: var(--muted); font-weight: 500; }}
    .candidate-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }}
    .candidate {{ border: 1px solid var(--line); border-radius: 6px; padding: 7px 9px; background: #fafbfc; }}
    .group p {{ margin: 0; color: var(--muted); line-height: 1.5; }}
    @media (max-width: 960px) {{
      .header-inner {{ grid-template-columns: 1fr; }}
      .controls {{ justify-content: flex-start; }}
      .card {{ grid-template-columns: 1fr; }}
      .media {{ min-height: 260px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <div>
        <h1>壁類候補レビュー</h1>
        <div class="summary">対象画像: {total} / 壁類候補あり: {card_count} / {html.escape(counts)}</div>
      </div>
      <div class="controls">
        <button class="active" data-filter="all">すべて</button>
        <button data-filter="grade_conflict">レベル差あり</button>
        <button data-filter="same_grade_merged">同一レベル</button>
        <button data-filter="single_model">片方のみ</button>
        <label class="toggle"><input type="checkbox" data-layer="auto" checked> 自動識別</label>
        <label class="toggle"><input type="checkbox" data-layer="result" checked> 判定結果</label>
        <label class="toggle"><input type="checkbox" data-layer="gt" checked> 正解ラベル</label>
      </div>
    </div>
  </header>
  <main>{''.join(cards) if cards else '<p>壁類候補はありません。</p>'}</main>
  <script>
    const buttons = document.querySelectorAll('button[data-filter]');
    const cards = document.querySelectorAll('.card');
    const labelPadding = 4;
    function overlaps(a, b) {{
      return !(a.right + labelPadding < b.left || a.left > b.right + labelPadding || a.bottom + labelPadding < b.top || a.top > b.bottom + labelPadding);
    }}
    function layoutLabels() {{
      document.querySelectorAll('.image-stage').forEach((stage) => {{
        const stageRect = stage.getBoundingClientRect();
        const labels = Array.from(stage.querySelectorAll('.box span')).filter((label) => {{
          const box = label.closest('.box');
          return box && getComputedStyle(box).display !== 'none' && getComputedStyle(label).display !== 'none';
        }});
        labels.forEach((label) => label.style.transform = '');
        const ordered = labels.sort((a, b) => {{
          const ar = a.getBoundingClientRect();
          const br = b.getBoundingClientRect();
          return (ar.top - br.top) || (ar.left - br.left);
        }});
        const placed = [];
        ordered.forEach((label) => {{
          let dy = 0;
          for (let attempt = 0; attempt < 18; attempt += 1) {{
            label.style.transform = `translateY(${{dy}}px)`;
            const rect = label.getBoundingClientRect();
            const hit = placed.some((other) => overlaps(rect, other));
            if (!hit && rect.bottom <= stageRect.bottom + 24) {{
              placed.push(rect);
              return;
            }}
            dy += 18;
          }}
          label.style.transform = `translateY(${{dy}}px)`;
          placed.push(label.getBoundingClientRect());
        }});
      }});
    }}
    buttons.forEach((button) => button.addEventListener('click', () => {{
      buttons.forEach((item) => item.classList.remove('active'));
      button.classList.add('active');
      const filter = button.dataset.filter;
      cards.forEach((card) => {{
        card.style.display = filter === 'all' || card.dataset.status === filter ? '' : 'none';
      }});
      requestAnimationFrame(layoutLabels);
    }}));
    document.querySelectorAll('input[data-layer]').forEach((input) => input.addEventListener('change', () => {{
      document.body.classList.toggle('hide-' + input.dataset.layer, !input.checked);
      requestAnimationFrame(layoutLabels);
    }}));
    window.addEventListener('load', layoutLabels);
    window.addEventListener('resize', () => requestAnimationFrame(layoutLabels));
    document.querySelectorAll('.image-stage img').forEach((img) => img.addEventListener('load', layoutLabels));
  </script>
</body>
</html>
"""


def clamp_percent(value: float) -> float:
    return max(0.0, min(100.0, value))


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


if __name__ == "__main__":
    raise SystemExit(main())
