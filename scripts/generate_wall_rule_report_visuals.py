#!/usr/bin/env python3
"""Generate focused wall-rule visuals for the RF-DETR meeting report."""

from __future__ import annotations

import json
from pathlib import Path
import textwrap
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "outputs/rfdetr_prod_pipeline/report_wall_rule_rc_wall_batch/results.jsonl"
OUT_DIR = REPO / "docs/report_assets_20260609_downstream"

FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

COLORS = {
    "bg": "#f7f8fa",
    "panel": "#ffffff",
    "text": "#1f2933",
    "muted": "#65758b",
    "inner": "#2563eb",
    "rc": "#d97706",
    "final": "#dc2626",
    "router": "#16a34a",
    "line": "#d7dde8",
}


CASES = [
    {
        "image": "data_add100__3-B-00069.jpg",
        "out": "rfdetr_wall_rule_case_cb_to_b.jpg",
        "title": "壁類表示ルール: 内壁 C x RC壁 B",
        "note": "RC壁判定を優先し、PC上は 壁-B として1件表示",
        "group_index": 0,
    },
    {
        "image": "data_add100__3-C-00021.jpg",
        "out": "rfdetr_wall_rule_case_dc_to_d.jpg",
        "title": "壁類表示ルール: 内壁 D x RC壁 C",
        "note": "指定ルールにより、PC上は 壁-D として1件表示",
        "group_index": 0,
    },
]


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = FONT_PATHS[:]
    if bold:
        candidates = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"] + candidates
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_rows() -> dict[str, dict[str, Any]]:
    rows = {}
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[Path(row["image"]).name] = row
    return rows


def draw_box(draw: ImageDraw.ImageDraw, box: list[float], scale: float, offset: tuple[int, int], color: str, width: int = 5) -> None:
    ox, oy = offset
    xy = [int(box[0] * scale + ox), int(box[1] * scale + oy), int(box[2] * scale + ox), int(box[3] * scale + oy)]
    draw.rectangle(xy, outline=color, width=width)


def fit_image(image: Image.Image, max_w: int, max_h: int) -> tuple[Image.Image, float, tuple[int, int]]:
    scale = min(max_w / image.width, max_h / image.height)
    size = (int(image.width * scale), int(image.height * scale))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    offset = ((max_w - size[0]) // 2, (max_h - size[1]) // 2)
    return resized, scale, offset


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int, fill: str = COLORS["text"], bold: bool = False) -> None:
    draw.text(xy, value, font=font(size, bold=bold), fill=fill)


def wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    size: int,
    width: int,
    fill: str = COLORS["text"],
) -> None:
    y = xy[1]
    for line in textwrap.wrap(value, width=width):
        text(draw, (xy[0], y), line, size, fill)
        y += int(size * 1.55)


def row_card(draw: ImageDraw.ImageDraw, y: int, label: str, value: str, color: str) -> None:
    x = 875
    draw.rounded_rectangle((x, y, 1330, y + 74), radius=10, fill="#ffffff", outline=COLORS["line"], width=2)
    draw.rounded_rectangle((x + 16, y + 18, x + 42, y + 44), radius=6, fill=color)
    text(draw, (x + 58, y + 14), label, 24, COLORS["muted"])
    text(draw, (x + 300, y + 12), value, 30, COLORS["text"], bold=True)


def generate_case(case: dict[str, Any], row: dict[str, Any]) -> None:
    group = row["wall_candidate_display"]["groups"][case["group_index"]]
    display = group["display_detections"][0]
    candidates = {item["source_model"]: item for item in group["candidates"]}

    image = Image.open(row["image"]).convert("RGB")
    canvas = Image.new("RGB", (1400, 820), COLORS["bg"])
    draw = ImageDraw.Draw(canvas)

    text(draw, (44, 32), case["title"], 34, COLORS["text"], bold=True)
    text(draw, (44, 78), case["note"], 23, COLORS["muted"])

    panel_x, panel_y, panel_w, panel_h = 44, 128, 780, 640
    draw.rounded_rectangle((panel_x, panel_y, panel_x + panel_w, panel_y + panel_h), radius=14, fill=COLORS["panel"], outline=COLORS["line"], width=2)
    fitted, scale, offset = fit_image(image, panel_w - 36, panel_h - 36)
    img_x = panel_x + 18 + offset[0]
    img_y = panel_y + 18 + offset[1]
    canvas.paste(fitted, (img_x, img_y))
    box_offset = (img_x, img_y)

    if "inner_wall" in candidates:
        draw_box(draw, candidates["inner_wall"]["bbox_xyxy"], scale, box_offset, COLORS["inner"], width=4)
    if "rc_wall" in candidates:
        draw_box(draw, candidates["rc_wall"]["bbox_xyxy"], scale, box_offset, COLORS["rc"], width=4)
    draw_box(draw, display["bbox_xyxy"], scale, box_offset, COLORS["final"], width=7)

    draw.rounded_rectangle((852, 128, 1356, 768), radius=14, fill=COLORS["panel"], outline=COLORS["line"], width=2)
    text(draw, (884, 164), "内部候補", 28, COLORS["text"], bold=True)
    inner_grade = candidates.get("inner_wall", {}).get("damage_grade", "-")
    rc_grade = candidates.get("rc_wall", {}).get("damage_grade", "-")
    row_card(draw, 218, "内壁モデル", str(inner_grade), COLORS["inner"])
    row_card(draw, 312, "RC壁モデル", str(rc_grade), COLORS["rc"])

    text(draw, (884, 432), "PC上の表示", 28, COLORS["text"], bold=True)
    draw.rounded_rectangle((884, 486, 1324, 596), radius=14, fill="#fff7f7", outline=COLORS["final"], width=3)
    text(draw, (916, 510), str(display["damage_grade"]), 48, COLORS["final"], bold=True)
    text(draw, (884, 632), "表示は1件のみ。候補詳細はJSONに保持。", 22, COLORS["muted"])
    wrapped_text(draw, (884, 672), str(display["reason"]), 21, 30, COLORS["muted"])

    legend_y = 782
    for x, label, color in [(62, "内壁候補", COLORS["inner"]), (210, "RC壁候補", COLORS["rc"]), (360, "最終表示", COLORS["final"])]:
        draw.rounded_rectangle((x, legend_y, x + 24, legend_y + 24), radius=5, fill=color)
        text(draw, (x + 34, legend_y - 4), label, 22, COLORS["muted"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_DIR / case["out"], quality=92)


def main() -> int:
    rows = load_rows()
    for case in CASES:
        generate_case(case, rows[case["image"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
