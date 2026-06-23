from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from audit_dataset import class_code_from_path, is_image_file, sha256_file
from common import ensure_dir, load_yaml, save_json


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported copy_mode: {mode}")


def build_manifest(cfg: dict) -> pd.DataFrame:
    raw_root = Path(cfg["raw_root"])
    extensions = {ext.lower() for ext in cfg["image_extensions"]}
    class_map = cfg["class_map"]
    rows = []
    for path in raw_root.rglob("*"):
        if not path.is_file() or not is_image_file(path, extensions):
            continue
        rel = path.relative_to(raw_root)
        code = class_code_from_path(rel, class_map)
        if code is None:
            continue
        rows.append(
            {
                "src_path": str(path),
                "raw_rel_path": str(rel),
                "class_code": code,
                "class_name": class_map[code],
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows).sort_values(["class_code", "src_path"]).reset_index(drop=True)


def assign_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    seed = int(cfg["seed"])
    split_cfg = cfg["split"]
    train_size = float(split_cfg["train"])
    val_size = float(split_cfg["val"])
    test_size = float(split_cfg["test"])
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    cross_class_hashes = df.groupby("sha256")["class_name"].nunique()
    cross_class_hashes = cross_class_hashes[cross_class_hashes > 1]
    if not cross_class_hashes.empty:
        bad_hashes = set(cross_class_hashes.index)
        conflict_rows = df[df["sha256"].isin(bad_hashes)].sort_values(["sha256", "class_name", "src_path"])
        audit_dir = ensure_dir(cfg["audit_dir"])
        conflict_rows.to_csv(audit_dir / "cross_class_duplicates.csv", index=False, encoding="utf-8-sig")
        if cfg.get("exclude_cross_class_duplicates", True):
            df = df[~df["sha256"].isin(bad_hashes)].copy()
            print(f"Excluded {len(conflict_rows)} cross-class duplicate files. See {audit_dir / 'cross_class_duplicates.csv'}")
        else:
            raise ValueError(
                "Some duplicate images appear under multiple classes. "
                "Resolve these before splitting to avoid ambiguous labels."
            )

    # Split by unique image hash so duplicate files cannot leak across train/val/test.
    group_df = df.drop_duplicates("sha256")[["sha256", "class_name"]].reset_index(drop=True)
    train_df, temp_df = train_test_split(
        group_df,
        train_size=train_size,
        random_state=seed,
        stratify=group_df["class_name"],
    )
    val_ratio_in_temp = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_in_temp,
        random_state=seed,
        stratify=temp_df["class_name"],
    )
    split_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    )[["sha256", "split"]]
    return df.merge(split_df, on="sha256", how="left").sort_values(["split", "class_name", "src_path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    processed_root = Path(cfg["processed_root"])
    manifest_dir = ensure_dir(cfg["manifest_dir"])

    if processed_root.exists() and (args.force or cfg.get("overwrite_processed", False)):
        shutil.rmtree(processed_root)
    elif processed_root.exists() and any(processed_root.iterdir()):
        raise RuntimeError(f"{processed_root} already exists and is not empty. Use --force to rebuild.")

    df = build_manifest(cfg)
    if df.empty:
        raise RuntimeError("No valid images found.")
    df = assign_splits(df, cfg)

    class_names = list(dict.fromkeys(cfg["class_map"].values()))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    df["class_idx"] = df["class_name"].map(class_to_idx)

    counters: dict[tuple[str, str], int] = {}
    rel_paths = []
    for row in df.itertuples(index=False):
        key = (row.split, row.class_name)
        counters[key] = counters.get(key, 0) + 1
        src = Path(row.src_path)
        suffix = src.suffix.lower()
        dst_rel = Path(row.split) / row.class_name / f"{row.class_name}_{counters[key]:05d}{suffix}"
        link_or_copy(src, processed_root / dst_rel, cfg.get("copy_mode", "hardlink"))
        rel_paths.append(str(dst_rel))

    df["processed_rel_path"] = rel_paths
    df.to_csv(manifest_dir / f'{cfg["dataset_name"]}_all.csv', index=False, encoding="utf-8-sig")
    for split in ["train", "val", "test"]:
        df[df["split"] == split].to_csv(
            manifest_dir / f'{cfg["dataset_name"]}_{split}.csv',
            index=False,
            encoding="utf-8-sig",
        )

    save_json(class_to_idx, manifest_dir / "class_to_idx.json")
    save_json(
        {
            "dataset_name": cfg["dataset_name"],
            "seed": cfg["seed"],
            "split": cfg["split"],
            "class_to_idx": class_to_idx,
        },
        manifest_dir / "split_config.json",
    )
    print(df.groupby(["split", "class_name"]).size().to_string())
    print(f"Prepared dataset at {processed_root}")


if __name__ == "__main__":
    main()
