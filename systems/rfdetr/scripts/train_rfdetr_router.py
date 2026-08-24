#!/usr/bin/env python3
"""Train an RF-DETR model for the 3-class building-element router."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rfdetr_router_base_aug_v2.yaml")
    # Any key defined under `experiments:` in the config is selectable. The old
    # hardcoded choices=["small", "medium"] made every other key unreachable,
    # including `large` and the `alt` experiment in the rc_wall report finetune
    # config. The name is validated against the config instead, in
    # build_train_options, so a typo still fails loudly.
    parser.add_argument("--experiment", default="small")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--model-size", choices=sorted(MODEL_CLASSES), default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", default="", help="positive integer or 'auto'")
    parser.add_argument("--grad-accum-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    # RF-DETR carries a separate backbone learning rate (TrainConfig.lr_encoder,
    # default 1.5e-4) which --lr does NOT touch. Every "low learning rate"
    # experiment before 2026-07-25 therefore lowered only the decoder while the
    # DINOv2 encoder kept training at 1.5e-4, five times faster. Exposed here so
    # the two can be set together, or the encoder frozen outright.
    parser.add_argument("--lr-encoder", type=float, default=0.0)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--devices", type=int, default=0,
                        help="GPUs for data-parallel training. batch-size is per device, so "
                             "2 devices at batch 14 reproduce the delivered effective 28 that "
                             "does not fit on one 32 GB card")
    parser.add_argument("--strategy", default="", help="PTL strategy, e.g. ddp")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="recompute backbone activations instead of storing them; "
                             "same gradients, less memory, so batch 28 fits on a 32 GB card")
    # Regularisation. Measured 2026-08-04: train recall 0.995 vs test 0.590 on
    # brace (394 boxes, only 2 missed, zero misses in the smallest quartile) is
    # extreme memorisation, yet the delivered runs used drop_path=0.0 - stochastic
    # depth entirely off - and weight_decay 1e-4. Neither was ever exposed here, so
    # none of the six training-side interventions in the 2026-07-26 table touched
    # them. Stochastic depth is the regulariser reported to gain most on small
    # corpora, which is the regime this project is in.
    # Background Recalibration: 27% of column_base training images carry damage
    # the annotators did not box, and focal loss pushes exactly those - the ones
    # the model is most confident about - hardest toward background. See
    # systems/rfdetr/scripts/brl_patch.py.
    parser.add_argument("--brl-threshold", type=float, default=0.0,
                        help="ignore unmatched queries above this confidence; 0 disables")
    parser.add_argument("--brl-mode", default="ignore", choices=["ignore", "flip", "pu"],
                        help="how to treat confident unmatched queries")
    parser.add_argument("--brl-pu-keep", type=float, default=0.25,
                        help="residual background weight in pu mode")
    parser.add_argument("--drop-path", type=float, default=-1.0,
                        help="stochastic depth rate; RF-DETR default is 0.0 (off)")
    parser.add_argument("--weight-decay", type=float, default=-1.0,
                        help="AdamW weight decay; RF-DETR default is 1e-4")
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--resolution", type=int, default=0)
    parser.add_argument("--num-queries", type=int, default=0)
    parser.add_argument("--focal-alpha", type=float, default=-1.0)
    parser.add_argument("--set-cost-class", type=float, default=-1.0)
    parser.add_argument("--set-cost-bbox", type=float, default=-1.0)
    parser.add_argument("--set-cost-giou", type=float, default=-1.0)
    parser.add_argument("--use-varifocal-loss", action="store_true")
    parser.add_argument("--checkpoint", default="", help="RF-DETR .pth checkpoint to initialize from")
    parser.add_argument(
        "--train-class-rows",
        default="",
        help="freeze all parameters except selected rows of every class_embed head, e.g. 3,4",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--trainer-precision", default="")
    parser.add_argument("--aug-config", default="", help="augmentation preset name or YAML/JSON file")
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--run-test", dest="run_test", action="store_true", default=None)
    parser.add_argument("--no-run-test", dest="run_test", action="store_false")
    parser.add_argument("--test-each-epoch", dest="test_each_epoch", action="store_true", default=None)
    parser.add_argument("--no-test-each-epoch", dest="test_each_epoch", action="store_false")
    parser.add_argument("--external-eval-profiles", dest="external_eval_profiles", action="store_true", default=None)
    parser.add_argument("--no-external-eval-profiles", dest="external_eval_profiles", action="store_false")
    parser.add_argument("--save-epoch-pth", dest="save_epoch_pth", action="store_true", default=None)
    parser.add_argument("--no-save-epoch-pth", dest="save_epoch_pth", action="store_false")
    parser.add_argument("--tensorboard", dest="tensorboard", action="store_true", default=None)
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def resolve_path(value: str | Path, repo: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo / path).resolve()


def build_train_options(args: argparse.Namespace, repo: Path) -> dict[str, Any]:
    cfg = load_config(resolve_path(args.config, repo))
    defaults = cfg.get("training_defaults", {}) or {}
    experiments = cfg.get("experiments", {}) or {}
    if experiments and args.experiment not in experiments:
        raise ValueError(
            f"experiment {args.experiment!r} is not defined in {args.config}; "
            f"available: {', '.join(sorted(experiments))}"
        )
    exp = experiments.get(args.experiment, {})
    dataset_cfg = cfg.get("dataset", {}) or {}

    dataset_dir = args.dataset_dir or dataset_cfg.get("dir")
    if not dataset_dir:
        raise ValueError("dataset dir is required")
    output_dir = args.output_dir or exp.get("output_dir") or f"outputs/rfdetr_router/{args.experiment}"
    model_size = args.model_size or exp.get("model_size") or args.experiment
    batch_size: int | str
    if args.batch_size:
        batch_size = "auto" if args.batch_size == "auto" else int(args.batch_size)
    else:
        batch_size = exp.get("batch_size", 4)

    options: dict[str, Any] = {
        "model_size": model_size,
        "dataset_dir": str(resolve_path(dataset_dir, repo)),
        "output_dir": str(resolve_path(output_dir, repo)),
        "epochs": args.epochs or int(defaults.get("epochs", 50)),
        "batch_size": batch_size,
        "grad_accum_steps": args.grad_accum_steps or int(exp.get("grad_accum_steps", 4)),
        "lr": args.lr or float(exp.get("lr", 1e-4)),
        "num_workers": args.num_workers if args.num_workers >= 0 else int(defaults.get("num_workers", 8)),
        "device": args.device,
        "dataset_file": "yolo",
        "official_eval_dataset_dir": str(resolve_path(dataset_cfg.get("dir", dataset_dir), repo)),
        "eval_interval": int(defaults.get("eval_interval", 1)),
        "checkpoint_interval": (
            args.checkpoint_interval if args.checkpoint_interval > 0 else int(defaults.get("checkpoint_interval", 1))
        ),
        "tensorboard": bool(defaults.get("tensorboard", True)) if args.tensorboard is None else args.tensorboard,
        "wandb": bool(defaults.get("wandb", False)),
        "run_test": bool(defaults.get("run_test", True)) if args.run_test is None else args.run_test,
        "test_each_epoch": (
            bool(defaults.get("test_each_epoch", True)) if args.test_each_epoch is None else args.test_each_epoch
        ),
        "external_eval_profiles": (
            list(defaults.get("external_eval_profiles", []) or [])
            if args.external_eval_profiles is not False
            else []
        ),
        "save_epoch_pth": (
            bool(defaults.get("save_epoch_pth", True)) if args.save_epoch_pth is None else args.save_epoch_pth
        ),
        "trainer_precision": args.trainer_precision or str(defaults.get("trainer_precision", "")),
        "seed": args.seed,
        "notes": {
            "run_id": datetime.now(UTC).strftime("%Y%m%d_%H%M%S"),
            "task": "shimizu_router_rfdetr",
            "config": str(resolve_path(args.config, repo)),
            "experiment": args.experiment,
            "selection_policy": cfg.get("selection_policy", {}),
        },
    }
    if args.resolution:
        options["resolution"] = args.resolution
    if args.num_queries:
        options["num_queries"] = args.num_queries
    if args.lr_encoder > 0:
        options["lr_encoder"] = args.lr_encoder
    if args.drop_path >= 0:
        options["drop_path"] = args.drop_path
    if args.weight_decay >= 0:
        options["weight_decay"] = args.weight_decay
    if args.freeze_encoder:
        options["freeze_encoder"] = True
    if args.gradient_checkpointing:
        options["gradient_checkpointing"] = True
    if args.devices:
        options["devices"] = args.devices
    if args.strategy:
        options["strategy"] = args.strategy
    if args.focal_alpha >= 0:
        options["focal_alpha"] = args.focal_alpha
    if args.set_cost_class >= 0:
        options["set_cost_class"] = args.set_cost_class
    if args.set_cost_bbox >= 0:
        options["set_cost_bbox"] = args.set_cost_bbox
    if args.set_cost_giou >= 0:
        options["set_cost_giou"] = args.set_cost_giou
    if args.use_varifocal_loss:
        options["use_varifocal_loss"] = True
    if args.aug_config:
        options["aug_config"] = load_aug_config(args.aug_config, repo)
    if args.checkpoint:
        options["checkpoint"] = str(resolve_path(args.checkpoint, repo))
    return options


def load_aug_config(value: str, repo: Path) -> Any:
    presets = {
        "default": "AUG_CONFIG",
        "conservative": "AUG_CONSERVATIVE",
        "aggressive": "AUG_AGGRESSIVE",
        "aerial": "AUG_AERIAL",
        "industrial": "AUG_INDUSTRIAL",
    }
    key = value.lower()
    if key in presets:
        from rfdetr.datasets import aug_config

        return getattr(aug_config, presets[key])

    path = resolve_path(value, repo)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    return data


def validate_dataset(dataset_dir: Path) -> None:
    required = [
        dataset_dir / "data.yaml",
        dataset_dir / "train" / "images",
        dataset_dir / "train" / "labels",
        dataset_dir / "valid" / "images",
        dataset_dir / "valid" / "labels",
        dataset_dir / "test" / "images",
        dataset_dir / "test" / "labels",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("RF-DETR YOLO dataset view is incomplete: " + ", ".join(missing))


def build_model(model_size: str, checkpoint: str = "", model_kwargs: dict[str, Any] | None = None) -> Any:
    try:
        import rfdetr
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "RF-DETR is not installed. Install it with: "
            "python -m pip install -r requirements-rfdetr.txt"
        ) from exc

    model_kwargs = model_kwargs or {}
    if checkpoint:
        return rfdetr.from_checkpoint(checkpoint, **model_kwargs)

    class_name = MODEL_CLASSES[model_size]
    model_class = getattr(rfdetr, class_name)
    return model_class(**model_kwargs)


def configure_class_row_training(model: Any, class_ids: list[int]) -> None:
    """Freeze the model and mask classifier gradients to selected class rows."""
    inner = model
    while not hasattr(inner, "named_parameters") and hasattr(inner, "model"):
        inner = inner.model
    if not hasattr(inner, "named_parameters"):
        raise TypeError(f"could not locate torch module under {type(model)!r}")
    selected = set(class_ids)
    trainable = 0
    heads = 0
    for name, parameter in inner.named_parameters():
        parameter.requires_grad = False
        if "class_embed" not in name or parameter.ndim not in {1, 2}:
            continue
        if not selected or max(selected) >= parameter.shape[0]:
            raise ValueError(f"class ids {sorted(selected)} incompatible with {name} {tuple(parameter.shape)}")
        parameter.requires_grad = True
        mask = parameter.new_zeros(parameter.shape)
        for class_id in selected:
            mask[class_id] = 1
        parameter.register_hook(lambda gradient, row_mask=mask: gradient * row_mask)
        trainable += sum(parameter[class_id].numel() for class_id in selected)
        heads += 1
    if not heads:
        raise RuntimeError("no class_embed parameters found for row-only training")
    print(f"[class-row-training] classes={sorted(selected)} heads={heads} trainable_values={trainable}")


def main() -> int:
    args = parse_args()
    if getattr(args, "brl_threshold", 0.0) > 0:
        import brl_patch
        brl_patch.apply(args.brl_threshold, args.brl_mode, args.brl_pu_keep)
    repo = Path.cwd()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    options = build_train_options(args, repo)
    validate_dataset(Path(options["dataset_dir"]))

    Path(options["output_dir"]).mkdir(parents=True, exist_ok=True)
    option_path = Path(options["output_dir"]) / "train_options.json"
    option_path.write_text(json.dumps(options, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(options, ensure_ascii=False, indent=2))

    if args.dry_run:
        print(f"[DRY-RUN] wrote {option_path}")
        return 0

    test_each_epoch = bool(options.pop("test_each_epoch"))
    external_eval_profiles = list(options.pop("external_eval_profiles", []) or [])
    official_eval_dataset_dir = str(options.pop("official_eval_dataset_dir"))
    save_epoch_pth = bool(options.pop("save_epoch_pth"))
    trainer_precision = str(options.pop("trainer_precision") or "")
    if test_each_epoch or save_epoch_pth or trainer_precision or external_eval_profiles:
        from scripts.rfdetr_router_callbacks import install_router_trainer_patch

        install_router_trainer_patch(
            save_epoch_pth=save_epoch_pth,
            test_each_epoch=test_each_epoch,
            trainer_precision=trainer_precision or None,
            external_eval_profiles=external_eval_profiles,
            official_eval_dataset_dir=official_eval_dataset_dir,
            eval_device="cpu",
        )

    model_size = str(options.pop("model_size"))
    checkpoint = str(options.pop("checkpoint", ""))
    model_kwargs: dict[str, Any] = {}
    if "num_queries" in options:
        model_kwargs["num_queries"] = int(options["num_queries"])
    # freeze_encoder lives on ModelConfig, not TrainConfig, so it has to reach the
    # constructor rather than model.train().
    if options.pop("freeze_encoder", False):
        model_kwargs["freeze_encoder"] = True
    model = build_model(model_size, checkpoint, model_kwargs)
    if args.train_class_rows:
        configure_class_row_training(
            model, [int(value) for value in args.train_class_rows.split(",") if value.strip()]
        )
    model.train(**options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
