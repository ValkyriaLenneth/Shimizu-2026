#!/usr/bin/env python3
"""Build the final crack-detection dataset folder from approved sources."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "final_crack_yolo_20260519"
DATA_ADD100 = REPO / "additional_data_2026-05-19" / "unpacked" / "data_add100"
LABELS_20251107 = REPO / "additional_data_2026-05-19" / "unpacked" / "labels_20251107"
DETECT_CVAT = REPO / "data" / "unzip"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CLASSES = [
    {
        "key": "tenjo",
        "display": "1_天井",
        "data_add_dir": "1_天井",
        "label_dir": "tenjo",
        "cvat_dir": "1_天井",
        "names": ["天井の損傷程度B", "天井の損傷程度C", "天井の損傷程度D"],
    },
    {
        "key": "inner_wall",
        "display": "2_内壁",
        "data_add_dir": "2_内壁",
        "label_dir": "inner_wall",
        "cvat_dir": "2_内壁",
        "names": ["内壁の損傷程度B", "内壁の損傷程度C", "内壁の損傷程度D"],
    },
    {
        "key": "rc_wall",
        "display": "3_RC壁",
        "data_add_dir": "3_RC壁",
        "label_dir": "rc_wall",
        "cvat_dir": "3_RC壁",
        "names": ["耐震壁の損傷程度B", "耐震壁の損傷程度C", "耐震壁の損傷程度D"],
    },
    {
        "key": "rc_column",
        "display": "4_RC柱",
        "data_add_dir": "4_RC柱",
        "label_dir": "rc_column",
        "cvat_dir": "4_RC柱",
        "names": ["RC柱の損傷程度B", "RC柱の損傷程度C", "RC柱の損傷程度D"],
    },
]


@dataclass(frozen=True)
class Sample:
    class_key: str
    class_display: str
    source: str
    source_split: str
    image_path: Path
    label_path: Path
    original_stem: str
    output_stem: str


def main() -> int:
    ensure_inputs()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "all").mkdir(parents=True)
    (OUT / "split").mkdir(parents=True)
    (OUT / "raw_sources").mkdir(parents=True)

    samples: list[Sample] = []
    for cfg in CLASSES:
        samples.extend(collect_data_add100(cfg))
        samples.extend(collect_20251107(cfg))

    rows = []
    summary: dict[str, object] = {
        "dataset": str(OUT.relative_to(REPO)),
        "classes": {},
        "totals": {"samples": 0, "boxes": 0, "invalid_label_files": 0},
        "sources": {
            "data_add100": str(DATA_ADD100.relative_to(REPO)),
            "labels_20251107": str(LABELS_20251107.relative_to(REPO)),
            "detect_dataset_cvat_images": str(DETECT_CVAT.relative_to(REPO)),
        },
    }

    for sample in samples:
        split = choose_split(sample)
        box_count, label_errors = validate_label(sample.label_path)
        copy_sample(sample, split)
        rows.append(
            {
                "class_key": sample.class_key,
                "class_display": sample.class_display,
                "source": sample.source,
                "source_split": sample.source_split,
                "final_split": split,
                "original_stem": sample.original_stem,
                "output_stem": sample.output_stem,
                "image": str(sample.image_path.relative_to(REPO)),
                "label": str(sample.label_path.relative_to(REPO)),
                "box_count": box_count,
                "label_errors": ";".join(label_errors),
            }
        )
        class_summary = summary["classes"].setdefault(
            sample.class_key,
            {
                "display": sample.class_display,
                "samples": 0,
                "boxes": 0,
                "sources": {},
                "splits": {"train": 0, "valid": 0, "test": 0},
                "invalid_label_files": 0,
            },
        )
        class_summary["samples"] += 1
        class_summary["boxes"] += box_count
        class_summary["sources"][sample.source] = class_summary["sources"].get(sample.source, 0) + 1
        class_summary["splits"][split] += 1
        if label_errors:
            class_summary["invalid_label_files"] += 1
            summary["totals"]["invalid_label_files"] += 1
        summary["totals"]["samples"] += 1
        summary["totals"]["boxes"] += box_count

    write_manifest(rows)
    write_data_yamls()
    copy_raw_source_metadata()
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def ensure_inputs() -> None:
    for path in [DATA_ADD100, LABELS_20251107, DETECT_CVAT]:
        if not path.exists():
            raise FileNotFoundError(path)


def collect_data_add100(cfg: dict[str, object]) -> list[Sample]:
    class_dir = DATA_ADD100 / str(cfg["data_add_dir"])
    samples = []
    for split in ["train", "valid", "test"]:
        image_dir = class_dir / split / "images"
        label_dir = class_dir / split / "labels"
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = find_image_by_stem(image_dir, label_path.stem)
            if image_path is None:
                raise FileNotFoundError(f"missing data_add100 image for {label_path}")
            samples.append(
                Sample(
                    class_key=str(cfg["key"]),
                    class_display=str(cfg["display"]),
                    source="data_add100",
                    source_split=split,
                    image_path=image_path,
                    label_path=label_path,
                    original_stem=label_path.stem,
                    output_stem=f"data_add100__{label_path.stem}",
                )
            )
    return samples


def collect_20251107(cfg: dict[str, object]) -> list[Sample]:
    label_root = find_obj_train_data(LABELS_20251107 / str(cfg["label_dir"]))
    image_root = DETECT_CVAT / str(cfg["cvat_dir"]) / "obj_train_data"
    image_index = {p.stem.lower(): p for p in image_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    samples = []
    for label_path in sorted(label_root.glob("*.txt")):
        image_path = image_index.get(label_path.stem.lower())
        if image_path is None:
            raise FileNotFoundError(f"missing detect_dataset-cvat image for {label_path}")
        samples.append(
            Sample(
                class_key=str(cfg["key"]),
                class_display=str(cfg["display"]),
                source="labels_20251107",
                source_split="none",
                image_path=image_path,
                label_path=label_path,
                original_stem=label_path.stem,
                output_stem=f"labels_20251107__{label_path.stem}",
            )
        )
    return samples


def find_obj_train_data(root: Path) -> Path:
    matches = [p for p in root.rglob("obj_train_data") if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"obj_train_data not found under {root}")
    if len(matches) > 1:
        raise RuntimeError(f"multiple obj_train_data directories under {root}: {matches}")
    return matches[0]


def find_image_by_stem(root: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        for candidate in [root / f"{stem}{suffix}", root / f"{stem}{suffix.upper()}"]:
            if candidate.exists():
                return candidate
    matches = [p for p in root.iterdir() if p.is_file() and p.stem.lower() == stem.lower() and p.suffix.lower() in IMAGE_SUFFIXES]
    return matches[0] if matches else None


def choose_split(sample: Sample) -> str:
    value = int(hashlib.sha1(f"{sample.class_key}/{sample.output_stem}".encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "valid"
    return "test"


def validate_label(path: Path) -> tuple[int, list[str]]:
    errors = []
    boxes = 0
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            errors.append(f"line{line_no}:field_count={len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"line{line_no}:non_numeric")
            continue
        if cls not in {0, 1, 2}:
            errors.append(f"line{line_no}:class={cls}")
        if any(v < 0 or v > 1 for v in coords):
            errors.append(f"line{line_no}:coord_out_of_range")
        boxes += 1
    return boxes, errors


def copy_sample(sample: Sample, split: str) -> None:
    for root in [OUT / "all" / sample.class_key, OUT / "split" / sample.class_key / split]:
        image_out = root / "images" / f"{sample.output_stem}{sample.image_path.suffix.lower()}"
        label_out = root / "labels" / f"{sample.output_stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image_path, image_out)
        shutil.copy2(sample.label_path, label_out)


def write_manifest(rows: list[dict[str, object]]) -> None:
    path = OUT / "manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_data_yamls() -> None:
    for cfg in CLASSES:
        names = "\n".join(f"- {name}" for name in cfg["names"])
        all_yaml = f"""path: {OUT / 'all' / str(cfg['key'])}
train: images
val: images
test: images
nc: 3
names:
{names}
"""
        split_yaml = f"""path: {OUT / 'split' / str(cfg['key'])}
train: train/images
val: valid/images
test: test/images
nc: 3
names:
{names}
"""
        (OUT / "all" / str(cfg["key"]) / "data.yaml").write_text(all_yaml, encoding="utf-8")
        (OUT / "split" / str(cfg["key"]) / "data.yaml").write_text(split_yaml, encoding="utf-8")


def copy_raw_source_metadata() -> None:
    for cfg in CLASSES:
        dst = OUT / "raw_sources" / str(cfg["key"])
        dst.mkdir(parents=True, exist_ok=True)
        data_yaml = DATA_ADD100 / str(cfg["data_add_dir"]) / "data.yaml"
        if data_yaml.exists():
            shutil.copy2(data_yaml, dst / "data_add100.data.yaml")
        label_root = LABELS_20251107 / str(cfg["label_dir"])
        for name in ["obj.names", "obj.data", "train.txt"]:
            for path in label_root.rglob(name):
                shutil.copy2(path, dst / f"labels_20251107.{name}")


def write_readme(summary: dict[str, object]) -> None:
    lines = [
        "# Final Crack YOLO Dataset 20260519",
        "",
        "本目录由 `scripts/build_final_crack_dataset_20260519.py` 生成。",
        "",
        "## 来源",
        "",
        "- `data_add100`: 每类 301 张图片 + label。",
        "- `labels_20251107`: 四类最后追加 label。",
        "- `detect_dataset-cvat`: 为 `labels_20251107` 提供对应图片，当前位于 `data/unzip`。",
        "",
        "## 目录",
        "",
        "- `all/<class>/images|labels`: 每类全部样本，文件名带来源前缀。",
        "- `split/<class>/<train|valid|test>/images|labels`: 对全部样本按稳定 hash 重新划分。",
        "- `manifest.csv`: 每个样本的来源、最终 split、原始路径、box 数。",
        "- `summary.json`: 聚合统计。",
        "- `raw_sources/<class>`: 每类来源侧 metadata 副本。",
        "",
        "## 汇总",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
