from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import ensure_dir, load_yaml


def class_code_from_path(path: Path, class_map: dict[str, str]) -> str | None:
    for part in path.parts:
        if len(part) >= 2 and part[1] == "." and part[0] in class_map:
            return part[0]
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_image_file(path: Path, extensions: set[str]) -> bool:
    return path.suffix.lower() in extensions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    raw_root = Path(cfg["raw_root"])
    audit_dir = ensure_dir(cfg["audit_dir"])
    extensions = {ext.lower() for ext in cfg["image_extensions"]}
    class_map = cfg["class_map"]

    all_files = [p for p in raw_root.rglob("*") if p.is_file()]
    image_files = [p for p in all_files if is_image_file(p, extensions)]

    valid_rows = []
    invalid_rows = []
    image_error_rows = []
    hash_to_paths: dict[str, list[str]] = defaultdict(list)

    for path in tqdm(image_files, desc="Auditing images"):
        code = class_code_from_path(path.relative_to(raw_root), class_map)
        if code is None:
            invalid_rows.append({"path": str(path), "reason": "missing_class_directory"})
            continue

        try:
            with Image.open(path) as img:
                img.verify()
            digest = sha256_file(path)
        except Exception as exc:
            image_error_rows.append({"path": str(path), "error": repr(exc)})
            continue

        hash_to_paths[digest].append(str(path))
        valid_rows.append(
            {
                "src_path": str(path),
                "class_code": code,
                "class_name": class_map[code],
                "sha256": digest,
            }
        )

    duplicate_rows = []
    for digest, paths in hash_to_paths.items():
        if len(paths) > 1:
            for path in paths:
                duplicate_rows.append({"sha256": digest, "path": path, "count": len(paths)})

    pd.DataFrame(valid_rows).to_csv(audit_dir / "valid_images.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(invalid_rows).to_csv(audit_dir / "invalid_files.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(image_error_rows).to_csv(audit_dir / "image_errors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(duplicate_rows).to_csv(audit_dir / "duplicate_files.csv", index=False, encoding="utf-8-sig")

    counts = pd.DataFrame(valid_rows).groupby(["class_code", "class_name"]).size().reset_index(name="count")
    counts.to_csv(audit_dir / "class_counts.csv", index=False, encoding="utf-8-sig")
    print(counts.to_string(index=False))
    print(f"valid={len(valid_rows)} invalid={len(invalid_rows)} image_errors={len(image_error_rows)} duplicates={len(duplicate_rows)}")


if __name__ == "__main__":
    main()
