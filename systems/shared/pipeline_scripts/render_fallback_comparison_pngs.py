#!/usr/bin/env python3
"""Render customer-facing PNG comparisons for fallback rescue cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO = Path(__file__).resolve().parents[2]
FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]

CLASS_JA = {
    "rc_wall": "RC壁",
    "inner_wall": "内壁",
    "rc_column": "RC柱",
    "tenjo": "天井",
}

ROUTER_CLASS_JA = {
    "壁类": "壁類",
    "RC柱": "RC柱",
    "天井": "天井",
}

MODEL_TO_STRUCTURE = {
    "rc_wall": "RC壁",
    "inner_wall": "内壁",
    "rc_column": "RC柱",
    "ceiling": "天井",
}

GT_GRADE = {0: "B", 1: "C", 2: "D"}

COLORS = {
    "bg": "#f3f5f7",
    "panel": "#ffffff",
    "panel_border": "#d8dee6",
    "text": "#18212b",
    "muted": "#5a6773",
    "before": "#b42318",
    "after": "#027a48",
    "gt": "#ffffff",
    "router": "#f79009",
    "pred": "#16a34a",
    "pred_fallback": "#0ea5e9",
    "none": "#9aa4b2",
    "legend_bg": "#101828",
    "chip_bg": "#eef2f6",
    "chip_border": "#d0d7de",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default="outputs/e2e_old_d900_baseline_v2")
    parser.add_argument("--after-dir", default="outputs/e2e_old_d900_fallback_per20")
    parser.add_argument("--output-dir", default="docs/report_assets_20260526/fallback_compare_png")
    parser.add_argument(
        "--samples",
        nargs="*",
        default=["3-C-00039.jpg", "d-173.jpg", "d-40044.JPG", "4-C-00022.jpg"],
        help="Image filename suffixes to render. Defaults to the 4 rescued examples used in section 2.4.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_dir = resolve(args.baseline_dir)
    after_dir = resolve(args.after_dir)
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fonts = load_fonts()
    baseline_results = load_results(baseline_dir)
    after_results = load_results(after_dir)
    baseline_eval = load_eval(baseline_dir)
    after_eval = load_eval(after_dir)

    rendered: list[tuple[Image.Image, str]] = []
    summary: list[dict[str, Any]] = []
    for idx, sample in enumerate(args.samples, start=1):
        before = find_by_suffix(baseline_results, sample)
        after = find_by_suffix(after_results, sample)
        before_eval = find_by_suffix(baseline_eval, sample, key="image")
        after_eval_row = find_by_suffix(after_eval, sample, key="image")
        if before is None or after is None or before_eval is None or after_eval_row is None:
            continue
        image = render_case(idx, sample, before, after, before_eval, after_eval_row, fonts)
        stem = safe_stem(f"{idx:02d}_{Path(sample).stem}_comparison")
        out_path = out_dir / f"{stem}.png"
        image.save(out_path)
        rendered.append((image, out_path.name))
        summary.append(
            {
                "sample": sample,
                "png": str(out_path.relative_to(REPO)),
                "baseline_fn": int(before_eval["main_false_negative"]),
                "after_fn": int(after_eval_row["main_false_negative"]),
                "baseline_main_predictions": int(before_eval["main_predictions"]),
                "after_main_predictions": int(after_eval_row["main_predictions"]),
                "fallback_rescued_matches": int(after_eval_row["fallback_rescued_matches"]),
            }
        )

    if rendered:
        contact = make_contact_sheet(rendered, fonts)
        contact_path = out_dir / "00_contact_sheet.png"
        contact.save(contact_path)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "images": [s["png"] for s in summary]}, ensure_ascii=False, indent=2))
    return 0


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO / p)


def load_fonts() -> dict[str, ImageFont.FreeTypeFont]:
    path = next((Path(p) for p in FONT_PATHS if Path(p).exists()), None)
    if path is None:
        raise FileNotFoundError("Japanese font not found. Expected Noto Sans CJK.")
    return {
        "title": ImageFont.truetype(str(path), 44),
        "subtitle": ImageFont.truetype(str(path), 26),
        "body": ImageFont.truetype(str(path), 24),
        "small": ImageFont.truetype(str(path), 20),
        "chip": ImageFont.truetype(str(path), 18),
        "box": ImageFont.truetype(str(path), 18),
    }


def load_results(e2e_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (e2e_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def load_eval(e2e_dir: Path) -> list[dict[str, str]]:
    return list(csv.DictReader((e2e_dir / "eval_by_image.csv").open(encoding="utf-8")))


def find_by_suffix(rows: list[Any], suffix: str, key: str = "image") -> Any | None:
    for row in rows:
        value = row[key] if isinstance(row, dict) else None
        if value and str(value).endswith(suffix):
            return row
    return None


def render_case(
    index: int,
    sample_name: str,
    before: dict[str, Any],
    after: dict[str, Any],
    before_eval: dict[str, str],
    after_eval: dict[str, str],
    fonts: dict[str, ImageFont.FreeTypeFont],
) -> Image.Image:
    image_path = Path(before["image"])
    label_path = resolve(after["eval_meta"]["label"])
    original = Image.open(image_path).convert("RGB")
    gt = load_gt(label_path, original.width, original.height)

    canvas_w = 2400
    canvas_h = 1520
    canvas = Image.new("RGB", (canvas_w, canvas_h), COLORS["bg"])
    draw = ImageDraw.Draw(canvas)

    title = f"改善事例 {index}: {sample_name}"
    structure = CLASS_JA.get(after_eval["class_key"], after_eval["class_key"])
    subtitle = f"対象構造: {structure}"
    draw.text((60, 44), title, font=fonts["title"], fill=COLORS["text"])
    draw.text((60, 102), subtitle, font=fonts["subtitle"], fill=COLORS["muted"])

    before_panel = render_panel(
        original=original,
        gt=gt,
        result=before,
        eval_row=before_eval,
        panel_title="改善前",
        title_color=COLORS["before"],
        fonts=fonts,
        show_fallback=False,
    )
    after_panel = render_panel(
        original=original,
        gt=gt,
        result=after,
        eval_row=after_eval,
        panel_title="改善後",
        title_color=COLORS["after"],
        fonts=fonts,
        show_fallback=True,
    )

    canvas.paste(before_panel, (60, 180))
    canvas.paste(after_panel, (1230, 180))
    draw_legend(canvas, fonts)
    return canvas


def render_panel(
    *,
    original: Image.Image,
    gt: list[dict[str, Any]],
    result: dict[str, Any],
    eval_row: dict[str, str],
    panel_title: str,
    title_color: str,
    fonts: dict[str, ImageFont.FreeTypeFont],
    show_fallback: bool,
) -> Image.Image:
    panel_w = 1110
    panel_h = 1230
    image_area_h = 760
    pad = 36
    panel = Image.new("RGB", (panel_w, panel_h), COLORS["panel"])
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((1, 1, panel_w - 2, panel_h - 2), radius=24, outline=COLORS["panel_border"], width=2)
    draw.rounded_rectangle((24, 22, 220, 84), radius=18, fill=title_color)
    draw.text((54, 38), panel_title, font=fonts["subtitle"], fill="#ffffff")

    framed = Image.new("RGB", (panel_w - pad * 2, image_area_h), "#111827")
    contained = ImageOps.contain(original, (framed.width, framed.height))
    xoff = (framed.width - contained.width) // 2
    yoff = (framed.height - contained.height) // 2
    framed.paste(contained, (xoff, yoff))

    overlay = ImageDraw.Draw(framed)
    scale_x = contained.width / original.width
    scale_y = contained.height / original.height
    scale = min(scale_x, scale_y)
    ox = xoff
    oy = yoff

    placed_labels: list[tuple[int, int, int, int]] = []
    for i, box in enumerate(gt, start=1):
        label = f"正解 {i}: {box['grade']}"
        draw_box(overlay, box["bbox_xyxy"], scale, ox, oy, COLORS["gt"], label, fonts["box"], placed_labels)

    for i, det in enumerate((result.get("router") or {}).get("detections", []), start=1):
        class_name = ROUTER_CLASS_JA.get(str(det.get("class_name", "")), str(det.get("class_name", "")))
        label = f"自動識別 {i}: {class_name} {float(det.get('confidence') or 0.0):.2f}"
        draw_box(overlay, det["bbox_xyxy"], scale, ox, oy, COLORS["router"], label, fonts["box"], placed_labels)

    main_model = str((result.get("eval_meta") or {}).get("main_model", ""))
    main_dets = [d for d in result.get("raw_crack_detections", []) if str(d.get("source_model", "")) == main_model]
    for i, det in enumerate(main_dets, start=1):
        structure = MODEL_TO_STRUCTURE.get(main_model, main_model)
        grade = grade_from_text(str(det.get("damage_grade", "")))
        prefix = "再確認" if show_fallback and bool(det.get("is_fallback")) else "判定"
        label = f"{prefix} {i}: {structure} {grade} {float(det.get('confidence') or 0.0):.2f}"
        color = COLORS["pred_fallback"] if bool(det.get("is_fallback")) else COLORS["pred"]
        draw_box(overlay, det["bbox_xyxy"], scale, ox, oy, color, label, fonts["box"], placed_labels)

    panel.paste(framed, (pad, 112))
    draw_panel_summary(panel, fonts, eval_row, result, main_dets, show_fallback)
    return panel


def draw_panel_summary(
    panel: Image.Image,
    fonts: dict[str, ImageFont.FreeTypeFont],
    eval_row: dict[str, str],
    result: dict[str, Any],
    main_dets: list[dict[str, Any]],
    show_fallback: bool,
) -> None:
    draw = ImageDraw.Draw(panel)
    y = 906
    chip_h = 42
    chips = [
        f"正解 {eval_row['gt_boxes']}件",
        f"判定 {eval_row['main_predictions']}件",
        f"漏れ {eval_row['main_false_negative']}件",
    ]
    if show_fallback:
        chips.append(f"救済 {eval_row['fallback_rescued_matches']}件")
    x = 36
    for chip in chips:
        w = text_width(fonts["chip"], chip) + 28
        draw.rounded_rectangle((x, y, x + w, y + chip_h), radius=18, fill=COLORS["chip_bg"], outline=COLORS["chip_border"])
        draw.text((x + 14, y + 9), chip, font=fonts["chip"], fill=COLORS["text"])
        x += w + 12

    router_classes = [ROUTER_CLASS_JA.get(p, p) for p in eval_row["router_classes"].split("|") if p]
    reason_lines = [
        f"自動識別の出力: {' / '.join(router_classes) if router_classes else 'なし'}",
        summary_line(result, main_dets, show_fallback),
    ]
    if show_fallback:
        reason_lines.append(f"再確認理由: {human_fallback_reason(main_dets)}")
    else:
        reason_lines.append("再確認なし: 正しい判定モデルに到達できず、主判定は0件")

    ty = 978
    for line in reason_lines:
        draw.text((36, ty), line, font=fonts["body"], fill=COLORS["text"])
        ty += 44


def summary_line(result: dict[str, Any], main_dets: list[dict[str, Any]], show_fallback: bool) -> str:
    if not main_dets:
        return "判定結果: 正しい判定モデルの検出なし"
    labels = []
    for det in main_dets:
        structure = MODEL_TO_STRUCTURE.get(str(det.get("source_model", "")), str(det.get("source_model", "")))
        grade = grade_from_text(str(det.get("damage_grade", "")))
        labels.append(f"{structure} {grade}")
    prefix = "判定結果" if show_fallback else "主判定"
    return f"{prefix}: " + " / ".join(labels)


def human_fallback_reason(main_dets: list[dict[str, Any]]) -> str:
    families = []
    for det in main_dets:
        for reason in det.get("fallback_reasons", []) or []:
            if reason.startswith("pair:壁类_present_RC柱_missing"):
                families.append("壁類として認識されたため、RC柱モデルを追加実行")
            elif reason.startswith("pair:RC柱_present_壁类_missing"):
                families.append("RC柱として認識されたため、壁モデルを追加実行")
            elif reason.startswith("main_dropout"):
                families.append("主判定の検出が弱かったため、関連モデルを追加実行")
            else:
                families.append(reason)
    return " / ".join(dict.fromkeys(families)) if families else "自動再確認を実行"


def draw_box(
    draw: ImageDraw.ImageDraw,
    box_xyxy: list[float],
    scale: float,
    ox: int,
    oy: int,
    color: str,
    label: str,
    font: ImageFont.FreeTypeFont,
    placed_labels: list[tuple[int, int, int, int]],
) -> None:
    x1, y1, x2, y2 = [int(round(v * scale)) for v in box_xyxy]
    x1 += ox
    x2 += ox
    y1 += oy
    y2 += oy
    draw.rectangle((x1, y1, x2, y2), outline=color, width=4)

    label_w = text_width(font, label) + 18
    label_h = font.size + 12
    lx = max(8, min(x1, draw.im.size[0] - label_w - 8))
    ly = max(8, y1 - label_h - 8)
    while any(overlap((lx, ly, lx + label_w, ly + label_h), rect) for rect in placed_labels):
        ly += label_h + 4
        if ly + label_h > draw.im.size[1] - 8:
            ly = max(8, y1 - label_h - 8)
            lx = min(draw.im.size[0] - label_w - 8, lx + 24)
            break
    draw.rounded_rectangle((lx, ly, lx + label_w, ly + label_h), radius=10, fill=color)
    draw.text((lx + 9, ly + 5), label, font=font, fill="#ffffff")
    placed_labels.append((lx, ly, lx + label_w, ly + label_h))


def overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def load_gt(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xc, yc, bw, bh = [float(v) for v in parts]
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        out.append({"bbox_xyxy": [x1, y1, x2, y2], "grade": GT_GRADE.get(int(cls), str(int(cls)))})
    return out


def grade_from_text(text: str) -> str:
    match = re.search(r"程度([BCD])", text)
    return match.group(1) if match else "?"


def draw_legend(canvas: Image.Image, fonts: dict[str, ImageFont.FreeTypeFont]) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((60, 1432, 2340, 1492), radius=18, fill=COLORS["legend_bg"])
    items = [
        ("正解", COLORS["gt"], "#111827"),
        ("自動識別", COLORS["router"], "#ffffff"),
        ("判定結果", COLORS["pred"], "#ffffff"),
        ("再確認で追加された判定", COLORS["pred_fallback"], "#ffffff"),
    ]
    x = 92
    for label, color, text_fill in items:
        draw.rounded_rectangle((x, 1444, x + 26, 1470), radius=8, fill=color)
        if text_fill != "#ffffff":
            draw.rectangle((x + 5, 1449, x + 21, 1465), outline=text_fill, width=2)
        draw.text((x + 40, 1440), label, font=fonts["small"], fill="#ffffff")
        x += 360


def text_width(font: ImageFont.FreeTypeFont, text: str) -> int:
    box = font.getbbox(text)
    return box[2] - box[0]


def make_contact_sheet(rendered: list[tuple[Image.Image, str]], fonts: dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    thumbs = []
    for image, name in rendered:
        thumb = image.copy()
        thumb.thumbnail((1120, 710))
        thumbs.append((thumb, name))

    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    margin = 40
    cell_w = 1120
    cell_h = 820
    canvas = Image.new("RGB", (cols * cell_w + (cols + 1) * margin, rows * cell_h + (rows + 1) * margin + 80), COLORS["bg"])
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 24), "改善事例一覧", font=fonts["title"], fill=COLORS["text"])
    for i, (thumb, name) in enumerate(thumbs):
        row = i // cols
        col = i % cols
        x = margin + col * (cell_w + margin)
        y = 100 + row * (cell_h + margin)
        draw.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=24, fill=COLORS["panel"], outline=COLORS["panel_border"], width=2)
        tx = x + (cell_w - thumb.width) // 2
        ty = y + 26
        canvas.paste(thumb, (tx, ty))
        draw.text((x + 28, y + cell_h - 54), name, font=fonts["small"], fill=COLORS["muted"])
    return canvas


def safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


if __name__ == "__main__":
    raise SystemExit(main())
