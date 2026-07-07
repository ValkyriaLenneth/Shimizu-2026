#!/usr/bin/env python3
"""Local review app for Gemini router annotations on new classes."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import tempfile
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "outputs/gemini_new_router_classes_20260630/results.jsonl"
REVIEW_DIR = ROOT / "outputs/gemini_new_router_classes_20260630/manual_review"
REVIEW_PATH = REVIEW_DIR / "review_annotations.json"
EXPORT_DIR = REVIEW_DIR / "export_yolo_labels"
DEDUP_ITEMS_PATH: Path | None = None

CLASS_TO_ID = {"天井": 0, "壁类": 1, "RC柱": 2, "ブレース": 3, "柱脚": 4}
LABEL_ALIASES = {"内壁": "壁类", "RC壁": "壁类"}
CLASS_NAMES = ["天井", "壁类", "RC柱", "ブレース", "柱脚"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SEARCH_ROOTS = [
    ROOT / "data/raw_new_classes_20260630/extracted",
    ROOT / ".local_artifacts/data/old_20260603_flat",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                row = json.loads(line)
                row["_line"] = line_number
                rows.append(row)
    return rows


def normalize_label(label: str | None) -> str | None:
    if label is None:
        return None
    return LABEL_ALIASES.get(label, label)


def clamp_box(box: list | tuple | None) -> list[float] | None:
    if not box or len(box) != 4:
        return None
    try:
        y1, x1, y2, x2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    y1 = max(0.0, min(1000.0, y1))
    x1 = max(0.0, min(1000.0, x1))
    y2 = max(0.0, min(1000.0, y2))
    x2 = max(0.0, min(1000.0, x2))
    if y2 <= y1 or x2 <= x1:
        return None
    return [round(y1, 3), round(x1, 3), round(y2, 3), round(x2, 3)]


def build_file_index() -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                parts = set(path.parts)
                label = None
                if "ブレース" in parts:
                    label = "ブレース"
                elif "柱脚" in parts:
                    label = "柱脚"
                elif path.name.startswith("brace_"):
                    label = "ブレース"
                elif path.name.startswith("columnbase_"):
                    label = "柱脚"
                if label:
                    index.setdefault((label, path.name), path)
    return index


def risk_flags(row: dict, boxes: list[dict]) -> list[str]:
    expected = row.get("expected_label")
    flags = []
    if not row.get("ok"):
        flags.append("api_not_ok")
    if row.get("error"):
        flags.append("error_field")
    expected_boxes = [b for b in boxes if b.get("label") == expected]
    if not expected_boxes:
        flags.append("no_expected_box")
    if len(expected_boxes) > 1:
        flags.append("multi_expected_boxes")
    for box in expected_boxes:
        y1, x1, y2, x2 = box["bbox"]
        area = (y2 - y1) * (x2 - x1) / 1_000_000
        if area >= 0.85:
            flags.append("full_image_box")
        if area <= 0.01:
            flags.append("tiny_box")
    return sorted(set(flags))


def row_to_item(row: dict, image_index: dict[tuple[str, str], Path]) -> dict:
    expected = row.get("expected_label")
    original_path = Path(str(row.get("image_path", "")))
    image_path = ROOT / original_path if not original_path.is_absolute() else original_path
    if not image_path.exists():
        image_path = image_index.get((expected, original_path.name), image_path)

    parsed = ((row.get("response") or {}).get("parsed") or {})
    boxes = []
    for element_index, element in enumerate(parsed.get("elements") or []):
        if not isinstance(element, dict):
            continue
        label = normalize_label(element.get("label"))
        bbox = clamp_box(element.get("bbox_2d"))
        if label not in CLASS_TO_ID or bbox is None:
            continue
        boxes.append(
            {
                "id": f"g{element_index}",
                "label": label,
                "bbox": bbox,
                "confidence": element.get("confidence"),
                "source_label": element.get("label"),
                "reason": element.get("reason") or "",
            }
        )
    item_id = str(row["_line"] - 1)
    item = {
        "id": item_id,
        "line": row["_line"],
        "expected_label": expected,
        "file_name": original_path.name,
        "image_rel_path": row.get("image_rel_path") or str(original_path),
        "image_path": str(image_path),
        "image_exists": image_path.exists(),
        "ok": bool(row.get("ok")),
        "error": row.get("error"),
        "notes": parsed.get("notes") or "",
        "image_level_labels": parsed.get("image_level_labels") or [],
        "boxes": boxes,
    }
    item["risk_flags"] = risk_flags(row, boxes)
    return item


def load_review() -> dict:
    if not REVIEW_PATH.exists():
        return {}
    return json.loads(REVIEW_PATH.read_text(encoding="utf-8"))


def save_review(review: dict) -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="review_annotations.", suffix=".json", dir=REVIEW_DIR)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(review, f, ensure_ascii=False, indent=2)
        f.write("\n")
    Path(tmp_name).replace(REVIEW_PATH)


def current_boxes(item: dict, review: dict) -> list[dict]:
    saved = review.get(item["id"])
    if saved and "boxes" in saved:
        return saved["boxes"]
    return item["boxes"]


def make_state() -> dict:
    if DEDUP_ITEMS_PATH is not None:
        items = json.loads(DEDUP_ITEMS_PATH.read_text(encoding="utf-8"))
    else:
        rows = load_jsonl(RESULTS_PATH)
        image_index = build_file_index()
        items = [row_to_item(row, image_index) for row in rows]
    review = load_review()
    return {"items": items, "review": review}


def to_yolo_line(box: dict) -> str | None:
    label = box.get("label")
    if label not in CLASS_TO_ID:
        return None
    bbox = clamp_box(box.get("bbox"))
    if bbox is None:
        return None
    y1, x1, y2, x2 = bbox
    cls = CLASS_TO_ID[label]
    xc = ((x1 + x2) / 2.0) / 1000.0
    yc = ((y1 + y2) / 2.0) / 1000.0
    w = (x2 - x1) / 1000.0
    h = (y2 - y1) / 1000.0
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def export_yolo(items: list[dict], review: dict) -> dict:
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
    (EXPORT_DIR / "images").mkdir(parents=True)
    (EXPORT_DIR / "labels").mkdir(parents=True)
    exported = skipped = 0
    for item in items:
        saved = review.get(item["id"], {})
        if saved.get("status") == "rejected":
            skipped += 1
            continue
        boxes = current_boxes(item, review)
        lines = [line for box in boxes if (line := to_yolo_line(box))]
        if not lines or not item["image_exists"]:
            skipped += 1
            continue
        stem = f"reviewed_{int(item['id']):05d}_{Path(item['file_name']).stem}"
        src = Path(item["image_path"])
        dst_image = EXPORT_DIR / "images" / f"{stem}{src.suffix.lower()}"
        dst_label = EXPORT_DIR / "labels" / f"{stem}.txt"
        os.link(src, dst_image)
        dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        exported += 1
    manifest = {
        "exported_images": exported,
        "skipped_items": skipped,
        "class_map": {str(i): name for i, name in enumerate(CLASS_NAMES)},
        "labels_dir": str((EXPORT_DIR / "labels").relative_to(ROOT)),
        "images_dir": str((EXPORT_DIR / "images").relative_to(ROOT)),
    }
    (EXPORT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Gemini Router 标注审核</title>
  <style>
    :root { --bg:#f7f7f4; --ink:#202124; --muted:#686b70; --line:#d8d6cf; --accent:#0f766e; --danger:#b42318; --blue:#2563eb; --amber:#b45309; }
    * { box-sizing:border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }
    header { height:56px; display:flex; align-items:center; gap:14px; padding:0 18px; border-bottom:1px solid var(--line); background:#fff; }
    header h1 { margin:0; font-size:17px; font-weight:700; }
    button, select, input, textarea { font:inherit; }
    button { border:1px solid var(--line); background:#fff; color:var(--ink); height:32px; padding:0 10px; border-radius:6px; cursor:pointer; }
    button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
    button.danger { color:var(--danger); }
    .app { display:grid; grid-template-columns: 320px 1fr 340px; height:calc(100vh - 56px); min-height:620px; }
    aside, .panel { overflow:auto; border-right:1px solid var(--line); background:#fff; }
    .right { border-left:1px solid var(--line); border-right:0; }
    .filters { padding:12px; display:grid; gap:8px; border-bottom:1px solid var(--line); }
    .filters .row { display:flex; gap:8px; }
    select, input, textarea { width:100%; border:1px solid var(--line); border-radius:6px; padding:7px 8px; background:#fff; }
    .stats { display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:12px; color:var(--muted); }
    .item { padding:10px 12px; border-bottom:1px solid #eceae4; cursor:pointer; }
    .item.active { background:#e8f3f1; }
    .item strong { display:block; font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .item span { display:block; margin-top:3px; font-size:12px; color:var(--muted); }
    .chips { display:flex; flex-wrap:wrap; gap:4px; margin-top:6px; }
    .chip { font-size:11px; padding:2px 6px; border-radius:999px; border:1px solid var(--line); background:#fafafa; color:var(--muted); }
    .chip.risk { color:var(--amber); border-color:#f0c36d; background:#fff7e6; }
    main { display:grid; grid-template-rows: auto 1fr; min-width:0; }
    .toolbar { padding:10px 12px; display:flex; gap:8px; align-items:center; border-bottom:1px solid var(--line); background:#fff; }
    .canvas-wrap { position:relative; overflow:auto; display:grid; place-items:center; background:#e9e7df; }
    #stage { position:relative; margin:18px; line-height:0; box-shadow:0 2px 18px rgba(0,0,0,.16); background:#111; }
    #image { display:block; max-width:calc(100vw - 730px); max-height:calc(100vh - 150px); width:auto; height:auto; }
    #overlay { position:absolute; inset:0; cursor:crosshair; }
    .right { padding:12px; display:grid; align-content:start; gap:12px; background:#fff; }
    .section { border:1px solid var(--line); border-radius:8px; padding:10px; }
    .section h2 { margin:0 0 8px; font-size:14px; }
    .meta { font-size:12px; color:var(--muted); display:grid; gap:5px; }
    .box-row { display:grid; grid-template-columns:1fr auto; gap:8px; align-items:center; padding:8px 0; border-top:1px solid #eceae4; }
    .box-row:first-child { border-top:0; }
    .box-row.active { background:#f0f9ff; margin:0 -6px; padding-left:6px; padding-right:6px; border-radius:6px; }
    .box-meta { font-size:12px; color:var(--muted); margin-top:3px; }
    .empty { color:var(--muted); font-size:13px; padding:12px; }
    .status { display:grid; grid-template-columns:1fr 1fr 1fr; gap:6px; }
    .kbd { color:var(--muted); font-size:12px; margin-left:auto; }
  </style>
</head>
<body>
  <header>
    <h1>Gemini Router 标注审核</h1>
    <button id="save" class="primary">保存</button>
    <button id="export">导出 YOLO</button>
    <span class="kbd">拖拽画框，点击选框，Delete 删除，←/→ 切换</span>
  </header>
  <div class="app">
    <aside>
      <div class="filters">
        <div class="row">
          <select id="labelFilter"><option value="">全部类别</option><option>ブレース</option><option>柱脚</option></select>
          <select id="riskFilter"><option value="">全部样本</option><option value="risk">只看风险</option><option value="missing">缺原图</option><option value="reviewed">已审核</option><option value="unreviewed">未审核</option></select>
        </div>
        <input id="search" placeholder="文件名 / 风险 / 标签" />
        <div class="stats" id="stats"></div>
      </div>
      <div id="list"></div>
    </aside>
    <main>
      <div class="toolbar">
        <button id="prev">←</button><button id="next">→</button>
        <select id="drawLabel" style="width:130px"></select>
        <button id="addExpected">整图期望类</button>
        <button id="deleteBox" class="danger">删除选框</button>
        <button id="reset">重置为 Gemini</button>
      </div>
      <div class="canvas-wrap">
        <div id="stage"><img id="image" alt=""><canvas id="overlay"></canvas></div>
      </div>
    </main>
    <aside class="right">
      <div class="section">
        <h2>当前图片</h2>
        <div class="meta" id="meta"></div>
      </div>
      <div class="section">
        <h2>审核状态</h2>
        <div class="status">
          <button data-status="accepted">通过</button>
          <button data-status="needs_fix">需修正</button>
          <button data-status="rejected">排除</button>
        </div>
        <textarea id="reviewNotes" rows="3" placeholder="人工备注"></textarea>
      </div>
      <div class="section">
        <h2>标注框</h2>
        <div id="boxes"></div>
      </div>
    </aside>
  </div>
<script>
const labels = ["天井", "壁类", "RC柱", "ブレース", "柱脚"];
const colors = {"天井":"#2563eb","壁类":"#16a34a","RC柱":"#7c3aed","ブレース":"#dc2626","柱脚":"#d97706"};
let state = {items: [], filtered: [], index: 0, item: null, boxes: [], selected: -1, review: {}, dirty: false};
const $ = id => document.getElementById(id);
labels.forEach(l => $("drawLabel").append(new Option(l, l)));

async function api(path, opts={}) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
function itemReview(item) { return state.review[item.id] || {}; }
function currentStatus(item) { return itemReview(item).status || "unreviewed"; }
function hasRisk(item) { return item.risk_flags && item.risk_flags.length; }
function filterItems() {
  const label = $("labelFilter").value, risk = $("riskFilter").value, q = $("search").value.toLowerCase();
  state.filtered = state.items.filter(it => {
    if (label && it.expected_label !== label) return false;
    if (risk === "risk" && !hasRisk(it)) return false;
    if (risk === "missing" && it.image_exists) return false;
    if (risk === "reviewed" && currentStatus(it) === "unreviewed") return false;
    if (risk === "unreviewed" && currentStatus(it) !== "unreviewed") return false;
    if (q) {
      const hay = [it.file_name, it.expected_label, currentStatus(it), ...(it.risk_flags||[])].join(" ").toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  state.index = Math.min(state.index, Math.max(0, state.filtered.length - 1));
  renderList();
  renderStats();
}
function renderStats() {
  const total = state.items.length, reviewed = state.items.filter(it => currentStatus(it) !== "unreviewed").length;
  const risk = state.items.filter(hasRisk).length, missing = state.items.filter(it => !it.image_exists).length;
  $("stats").innerHTML = `<div>总数 ${total}</div><div>当前 ${state.filtered.length}</div><div>风险 ${risk}</div><div>已审 ${reviewed}</div><div>缺图 ${missing}</div>`;
}
function renderList() {
  $("list").innerHTML = state.filtered.map((it, i) => {
    const chips = [`<span class="chip">${currentStatus(it)}</span>`, ...(it.risk_flags||[]).map(r => `<span class="chip risk">${r}</span>`)].join("");
    return `<div class="item ${i===state.index?'active':''}" data-i="${i}"><strong>${it.file_name}</strong><span>${it.expected_label} · line ${it.line}</span><div class="chips">${chips}</div></div>`;
  }).join("") || `<div class="empty">没有匹配样本</div>`;
  document.querySelectorAll(".item").forEach(el => el.onclick = () => loadByIndex(Number(el.dataset.i)));
}
async function loadByIndex(i) {
  if (!state.filtered.length) return;
  if (state.dirty) await saveCurrent(false);
  state.index = Math.max(0, Math.min(i, state.filtered.length - 1));
  const brief = state.filtered[state.index];
  state.item = await api(`/api/item/${brief.id}`);
  state.review[state.item.id] = state.item.review || state.review[state.item.id] || {};
  state.boxes = JSON.parse(JSON.stringify(state.item.current_boxes));
  state.selected = state.boxes.length ? 0 : -1;
  $("drawLabel").value = state.item.expected_label;
  $("image").src = state.item.image_exists ? `/image/${state.item.id}` : "";
  $("image").onload = resizeCanvas;
  renderAll();
}
function renderAll() { renderList(); renderMeta(); renderBoxes(); draw(); }
function renderMeta() {
  const it = state.item;
  if (!it) return;
  $("meta").innerHTML = [
    `期望类别：${it.expected_label}`,
    `文件：${it.file_name}`,
    `原图：${it.image_exists ? "已找到" : "缺失"}`,
    `Gemini labels：${(it.image_level_labels||[]).join(", ") || "-"}`,
    `风险：${(it.risk_flags||[]).join(", ") || "无"}`,
    `备注：${it.notes || "-"}`
  ].map(x => `<div>${escapeHtml(x)}</div>`).join("");
  $("reviewNotes").value = itemReview(it).notes || "";
  document.querySelectorAll("[data-status]").forEach(b => {
    b.className = (itemReview(it).status === b.dataset.status) ? "primary" : "";
  });
}
function renderBoxes() {
  $("boxes").innerHTML = state.boxes.map((b, i) => {
    const bb = b.bbox.map(v => Math.round(v)).join(", ");
    return `<div class="box-row ${i===state.selected?'active':''}" data-i="${i}">
      <div><select data-label="${i}">${labels.map(l => `<option ${b.label===l?'selected':''}>${l}</option>`).join("")}</select>
      <div class="box-meta">${bb} · conf ${b.confidence ?? "-"}</div></div><button class="danger" data-del="${i}">删</button></div>`;
  }).join("") || `<div class="empty">暂无框，拖拽可新增</div>`;
  document.querySelectorAll("[data-label]").forEach(el => el.onchange = () => { state.boxes[Number(el.dataset.label)].label = el.value; state.dirty = true; draw(); renderBoxes(); });
  document.querySelectorAll("[data-del]").forEach(el => el.onclick = () => deleteBox(Number(el.dataset.del)));
  document.querySelectorAll(".box-row").forEach(el => el.onclick = e => { if (!e.target.dataset.del) { state.selected = Number(el.dataset.i); renderBoxes(); draw(); }});
}
function resizeCanvas() {
  const img = $("image"), canvas = $("overlay");
  canvas.width = img.clientWidth; canvas.height = img.clientHeight;
  canvas.style.width = img.clientWidth + "px"; canvas.style.height = img.clientHeight + "px";
  draw();
}
function draw() {
  const canvas = $("overlay"), ctx = canvas.getContext("2d");
  ctx.clearRect(0,0,canvas.width,canvas.height);
  state.boxes.forEach((b,i) => {
    const [y1,x1,y2,x2] = b.bbox;
    const x = x1/1000*canvas.width, y = y1/1000*canvas.height, w = (x2-x1)/1000*canvas.width, h = (y2-y1)/1000*canvas.height;
    ctx.strokeStyle = colors[b.label] || "#fff"; ctx.lineWidth = i===state.selected ? 4 : 2; ctx.strokeRect(x,y,w,h);
    ctx.fillStyle = colors[b.label] || "#111"; ctx.fillRect(x,y,ctx.measureText(b.label).width+16,22);
    ctx.fillStyle = "#fff"; ctx.font = "13px sans-serif"; ctx.fillText(b.label, x+6, y+15);
  });
}
function pointTo1000(e) {
  const r = $("overlay").getBoundingClientRect();
  return [Math.max(0, Math.min(1000, (e.clientX-r.left)/r.width*1000)), Math.max(0, Math.min(1000, (e.clientY-r.top)/r.height*1000))];
}
let drag = null;
$("overlay").addEventListener("mousedown", e => { drag = pointTo1000(e); });
$("overlay").addEventListener("mouseup", e => {
  if (!drag) return;
  const [x2,y2] = pointTo1000(e), [x1,y1] = drag; drag = null;
  if (Math.abs(x2-x1) < 5 || Math.abs(y2-y1) < 5) return selectAt(x2,y2);
  state.boxes.push({id:"m"+Date.now(), label:$("drawLabel").value, bbox:[Math.min(y1,y2),Math.min(x1,x2),Math.max(y1,y2),Math.max(x1,x2)], confidence:null, reason:"manual"});
  state.selected = state.boxes.length - 1; state.dirty = true; renderBoxes(); draw();
});
function selectAt(x,y) {
  state.selected = state.boxes.findIndex(b => y>=b.bbox[0] && x>=b.bbox[1] && y<=b.bbox[2] && x<=b.bbox[3]);
  renderBoxes(); draw();
}
function deleteBox(i=state.selected) {
  if (i < 0) return;
  state.boxes.splice(i,1); state.selected = Math.min(i, state.boxes.length-1); state.dirty = true; renderBoxes(); draw();
}
async function saveCurrent(show=true) {
  if (!state.item) return;
  const body = {boxes: state.boxes, status: itemReview(state.item).status || "needs_fix", notes: $("reviewNotes").value};
  const res = await api(`/api/item/${state.item.id}`, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  state.review[state.item.id] = res.review; state.dirty = false; if (show) alert("已保存");
  filterItems();
}
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
$("save").onclick = () => saveCurrent(true);
$("export").onclick = async () => { await saveCurrent(false); const r = await api("/api/export", {method:"POST"}); alert(`导出完成：${r.exported_images} 张`); };
$("prev").onclick = () => loadByIndex(state.index-1); $("next").onclick = () => loadByIndex(state.index+1);
$("deleteBox").onclick = () => deleteBox();
$("reset").onclick = () => { state.boxes = JSON.parse(JSON.stringify(state.item.boxes)); state.dirty = true; state.selected = state.boxes.length?0:-1; renderBoxes(); draw(); };
$("addExpected").onclick = () => { state.boxes.push({id:"m"+Date.now(), label:state.item.expected_label, bbox:[0,0,1000,1000], confidence:null, reason:"manual_full_image"}); state.selected=state.boxes.length-1; state.dirty=true; renderBoxes(); draw(); };
document.querySelectorAll("[data-status]").forEach(b => b.onclick = () => { if (!state.item) return; state.review[state.item.id] = {...itemReview(state.item), status:b.dataset.status}; state.dirty = true; renderMeta(); });
["labelFilter","riskFilter","search"].forEach(id => $(id).oninput = filterItems);
document.addEventListener("keydown", e => { if(e.key==="Delete") deleteBox(); if(e.key==="ArrowRight") loadByIndex(state.index+1); if(e.key==="ArrowLeft") loadByIndex(state.index-1); });
window.onresize = resizeCanvas;
(async function init(){ const data = await api("/api/items"); state.items=data.items; state.review=data.review; filterItems(); await loadByIndex(0); })();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "GeminiReview/1.0"

    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str = "text/html; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/":
                self.send_text(INDEX_HTML)
            elif path == "/api/items":
                state = make_state()
                brief = []
                review = state["review"]
                for item in state["items"]:
                    brief.append({k: item[k] for k in ["id", "line", "expected_label", "file_name", "image_exists", "risk_flags"]})
                self.send_json({"items": brief, "review": review})
            elif match := re.fullmatch(r"/api/item/(\d+)", path):
                state = make_state()
                item = state["items"][int(match.group(1))]
                review = state["review"].get(item["id"], {})
                item["review"] = review
                item["current_boxes"] = current_boxes(item, state["review"])
                self.send_json(item)
            elif match := re.fullmatch(r"/image/(\d+)", path):
                state = make_state()
                item = state["items"][int(match.group(1))]
                image_path = Path(item["image_path"])
                if not image_path.exists():
                    self.send_error(404)
                    return
                content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
                data = image_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/summary":
                state = make_state()
                counts = Counter(item["expected_label"] for item in state["items"])
                risks = Counter(flag for item in state["items"] for flag in item["risk_flags"])
                missing = sum(1 for item in state["items"] if not item["image_exists"])
                self.send_json({"counts": dict(counts), "risks": dict(risks), "missing_images": missing})
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}") if length else {}
        try:
            if match := re.fullmatch(r"/api/item/(\d+)", path):
                state = make_state()
                item = state["items"][int(match.group(1))]
                boxes = []
                for index, box in enumerate(payload.get("boxes") or []):
                    label = normalize_label(box.get("label"))
                    bbox = clamp_box(box.get("bbox"))
                    if label in CLASS_TO_ID and bbox:
                        boxes.append({"id": box.get("id") or f"m{index}", "label": label, "bbox": bbox, "confidence": box.get("confidence"), "reason": box.get("reason") or "manual"})
                review = state["review"]
                review[item["id"]] = {
                    "status": payload.get("status") or "needs_fix",
                    "notes": payload.get("notes") or "",
                    "boxes": boxes,
                    "file_name": item["file_name"],
                    "expected_label": item["expected_label"],
                }
                save_review(review)
                self.send_json({"review": review[item["id"]]})
            elif path == "/api/export":
                state = make_state()
                manifest = export_yolo(state["items"], state["review"])
                self.send_json(manifest)
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--dedup", action="store_true", help="Use the deduplicated review queue.")
    args = parser.parse_args()
    global REVIEW_DIR, REVIEW_PATH, EXPORT_DIR, DEDUP_ITEMS_PATH
    if args.dedup:
        REVIEW_DIR = ROOT / "outputs/gemini_new_router_classes_20260630/manual_review_dedup"
        REVIEW_PATH = REVIEW_DIR / "review_annotations.json"
        EXPORT_DIR = REVIEW_DIR / "export_yolo_labels"
        DEDUP_ITEMS_PATH = REVIEW_DIR / "dedup_items.json"
        if not DEDUP_ITEMS_PATH.exists():
            raise FileNotFoundError(f"{DEDUP_ITEMS_PATH}; run dedup_review.py first")
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(RESULTS_PATH)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving Gemini annotation review app at http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
