from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "天井": (0, 170, 255),
    "内壁": (0, 200, 120),
    "RC壁": (255, 145, 0),
    "RC柱": (220, 70, 255),
}
ASCII_LABELS = {
    "天井": "ceiling",
    "内壁": "inner_wall",
    "RC壁": "rc_wall",
    "RC柱": "rc_column",
}


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def scale_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    ymin, xmin, ymax, xmax = bbox
    left = round(max(0, min(1000, xmin)) / 1000 * width)
    top = round(max(0, min(1000, ymin)) / 1000 * height)
    right = round(max(0, min(1000, xmax)) / 1000 * width)
    bottom = round(max(0, min(1000, ymax)) / 1000 * height)
    return left, top, right, bottom


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: tuple[int, int, int], font) -> None:
    x, y = xy
    margin = 5
    bbox = draw.textbbox((x, y), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    y = max(0, y - th - margin * 2)
    draw.rectangle((x, y, x + tw + margin * 2, y + th + margin * 2), fill=color)
    draw.text((x + margin, y + margin), text, fill=(0, 0, 0), font=font)


def visualize_row(row: dict, out_path: Path, max_side: int) -> dict:
    image_path = Path(row["image_path"])
    parsed = row["response"]["parsed"] if row.get("ok") else {"elements": []}
    elements = parsed.get("elements", [])
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        scale = min(1.0, max_side / max(orig_w, orig_h))
        if scale < 1:
            img = img.resize((round(orig_w * scale), round(orig_h * scale)), Image.Resampling.LANCZOS)
        width, height = img.size
        draw = ImageDraw.Draw(img)
        font = load_font(max(18, round(max(width, height) / 55)))
        line_width = max(4, round(max(width, height) / 180))

        for element in elements:
            label = element.get("label", "unknown")
            color = COLORS.get(label, (255, 255, 0))
            bbox = element.get("bbox_2d")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            left, top, right, bottom = scale_bbox(bbox, width, height)
            draw.rectangle((left, top, right, bottom), outline=color, width=line_width)
            conf = element.get("confidence")
            conf_text = f" {conf:.2f}" if isinstance(conf, (int, float)) else ""
            text = f"{ASCII_LABELS.get(label, label)}{conf_text}"
            draw_label(draw, (left, top), text, color, font)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, quality=90)

    return {
        "rel": out_path.name,
        "orig_size": [orig_w, orig_h],
        "viz_size": [width, height],
        "elements": elements,
        "image_level_labels": parsed.get("image_level_labels", []),
        "notes": parsed.get("notes", ""),
    }


def write_html(cards: list[dict], out_dir: Path, title: str) -> None:
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;background:#f6f7f9;color:#17202a}",
        "h1{font-size:22px;margin:0 0 16px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}",
        ".card{background:white;border:1px solid #d9dee7;border-radius:8px;padding:12px}",
        ".meta{font-size:13px;color:#4c5968;line-height:1.45;margin-bottom:8px}",
        "img{width:100%;height:auto;border:1px solid #edf0f3;border-radius:6px;background:#111}",
        ".chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}",
        ".chip{font-size:12px;border-radius:999px;background:#edf3ff;padding:2px 8px}",
        "pre{white-space:pre-wrap;font-size:12px;background:#f3f5f7;padding:8px;border-radius:6px;overflow:auto}",
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        "<div class='grid'>",
    ]
    for card in cards:
        labels = card["viz"].get("image_level_labels", [])
        chips = "".join(f"<span class='chip'>{html.escape(str(label))}</span>" for label in labels)
        elements = json.dumps(card["viz"].get("elements", []), ensure_ascii=False, indent=2)
        parts.extend(
            [
                "<div class='card'>",
                "<div class='meta'>",
                f"<strong>{html.escape(card['image_rel_path'])}</strong><br>",
                f"expected: {html.escape(card['expected_label'])} | predicted labels: {html.escape(', '.join(labels))}",
                "</div>",
                f"<img src='visualizations/{html.escape(card['viz']['rel'])}' loading='lazy'>",
                f"<div class='chips'>{chips}</div>",
                f"<pre>{html.escape(elements)}</pre>",
                "</div>",
            ]
        )
    parts.extend(["</div></body></html>"])
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/gemini_coarse_3_1_pro_50x4/results.jsonl")
    parser.add_argument("--out-dir", default="outputs/gemini_coarse_3_1_pro_50x4")
    parser.add_argument("--max-side", type=int, default=1600)
    args = parser.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.out_dir)
    viz_dir = out_dir / "visualizations"
    rows = read_rows(results_path)
    cards = []
    for idx, row in enumerate(rows, start=1):
        stem = f"{idx:03d}_{Path(row['image_path']).stem}.jpg"
        viz = visualize_row(row, viz_dir / stem, args.max_side)
        cards.append({**row, "viz": viz})
        print(f"{idx}/{len(rows)} {row['image_rel_path']} -> {viz_dir / stem}")

    summary = {
        "results": str(results_path),
        "count": len(rows),
        "visualization_dir": str(viz_dir),
        "html": str(out_dir / "index.html"),
    }
    (out_dir / "visualization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_html(cards, out_dir, "Gemini coarse annotation preview")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
