"""Router-specific RF-DETR training callbacks.

These callbacks are intentionally kept outside the installed ``rfdetr`` package.
``train_rfdetr_router.py`` monkey-patches RF-DETR's trainer factory at runtime so
the upstream package remains reproducible from pip.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer


class RouterPerEpochEvalCallback(Callback):
    """Save RF-DETR-loadable epoch checkpoints and optionally test each epoch."""

    def __init__(
        self,
        *,
        save_epoch_pth: bool = True,
        test_each_epoch: bool = True,
        metric_csv_name: str = "test_results.csv",
        external_eval_profiles: list[dict[str, Any]] | None = None,
        official_eval_dataset_dir: str = "",
        eval_device: str = "cpu",
    ) -> None:
        super().__init__()
        self.save_epoch_pth = save_epoch_pth
        self.test_each_epoch = test_each_epoch
        self.metric_csv_name = metric_csv_name
        self.external_eval_profiles = external_eval_profiles or []
        self.official_eval_dataset_dir = official_eval_dataset_dir
        self.eval_device = eval_device
        self._running_test = False

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking or self._running_test:
            return
        if "val/mAP_50_95" not in trainer.callback_metrics:
            return

        epoch = int(trainer.current_epoch)
        checkpoint_path: Path | None = None
        if self.save_epoch_pth:
            checkpoint_path = self._save_epoch_pth(trainer, pl_module, epoch)
        if self.test_each_epoch:
            self._run_test_and_record(trainer, pl_module, epoch)
        if self.external_eval_profiles and trainer.is_global_zero:
            if checkpoint_path is None:
                checkpoint_path = self._epoch_pth_path(pl_module, epoch)
            self._run_external_eval_profiles(pl_module, checkpoint_path, epoch)

    def _save_epoch_pth(self, trainer: Trainer, pl_module: LightningModule, epoch: int) -> Path | None:
        if not trainer.is_global_zero:
            return None

        from rfdetr.training.callbacks.best_model import BestModelCallback

        path = self._epoch_pth_path(pl_module, epoch)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_state_dict = self._get_preferred_state_dict(trainer, pl_module)
        train_config = self._train_config_with_class_names(trainer, pl_module)
        args_dict = train_config.model_dump() if hasattr(train_config, "model_dump") else train_config
        model_name = BestModelCallback._resolve_model_name(pl_module)
        payload = BestModelCallback._build_checkpoint_payload(
            model_state_dict,
            args_dict,
            trainer,
            model_name=model_name,
        )
        torch.save(payload, path)
        return path

    def _run_test_and_record(self, trainer: Trainer, pl_module: LightningModule, epoch: int) -> None:
        if not trainer.is_global_zero:
            return

        datamodule = trainer.datamodule
        if datamodule is None:
            return

        self._force_real_yolo_test_split(datamodule)
        self._running_test = True
        try:
            results = trainer.test(pl_module, datamodule=datamodule, verbose=False)
        finally:
            self._running_test = False

        metrics: dict[str, Any] = dict(results[0]) if results else {}
        metrics["epoch"] = epoch
        self._append_metrics(Path(pl_module.train_config.output_dir) / self.metric_csv_name, metrics)

    def _run_external_eval_profiles(
        self,
        pl_module: LightningModule,
        checkpoint_path: Path,
        epoch: int,
    ) -> None:
        if not checkpoint_path.exists():
            print(f"[external-eval] skip epoch {epoch}: missing checkpoint {checkpoint_path}", flush=True)
            return

        output_dir = Path(pl_module.train_config.output_dir)
        for profile in self.external_eval_profiles:
            name = str(profile.get("name", "")).strip()
            if not name:
                raise ValueError("external eval profile is missing a non-empty name")
            evaluator = str(profile.get("evaluator", "threshold_sweep")).strip()
            dataset_dir = str(profile.get("dataset_dir") or self.official_eval_dataset_dir).strip()
            if not dataset_dir:
                raise ValueError(f"external eval profile {name!r} is missing dataset_dir")
            split = str(profile.get("split", "test"))
            iou_threshold = str(profile.get("iou_threshold", 0.5))
            device = str(profile.get("device", self.eval_device))
            num_classes = str(profile.get("num_classes", 3))

            sweep_dir = output_dir / "external_eval" / name
            sweep_csv = sweep_dir / f"epoch_{epoch:03d}.csv"
            if evaluator == "class_threshold_grid":
                threshold_grid = str(profile.get("threshold_grid", profile.get("thresholds", ""))).strip()
                if not threshold_grid:
                    raise ValueError(f"external eval profile {name!r} is missing threshold_grid")
                cmd = [
                    sys.executable,
                    "scripts/evaluate_rfdetr_class_threshold_grid.py",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--dataset-dir",
                    dataset_dir,
                    "--split",
                    split,
                    "--threshold-grid",
                    threshold_grid,
                    "--iou-threshold",
                    iou_threshold,
                    "--num-classes",
                    num_classes,
                    "--output-csv",
                    str(sweep_csv),
                    "--device",
                    device,
                ]
                eval_desc = f"class_threshold_grid={threshold_grid}"
            elif evaluator == "threshold_sweep":
                thresholds = str(profile.get("thresholds", "")).strip()
                if not thresholds:
                    raise ValueError(f"external eval profile {name!r} is missing thresholds")
                cmd = [
                    sys.executable,
                    "scripts/evaluate_rfdetr_threshold_sweep.py",
                    "--checkpoint",
                    str(checkpoint_path),
                    "--dataset-dir",
                    dataset_dir,
                    "--split",
                    split,
                    "--thresholds",
                    thresholds,
                    "--iou-threshold",
                    iou_threshold,
                    "--num-classes",
                    num_classes,
                    "--output-csv",
                    str(sweep_csv),
                    "--device",
                    device,
                ]
                eval_desc = f"thresholds={thresholds}"
            else:
                raise ValueError(f"unknown external eval evaluator {evaluator!r}")
            print(
                f"[external-eval] epoch {epoch} profile={name} "
                f"{eval_desc} match_iou={iou_threshold}",
                flush=True,
            )
            env = os.environ.copy()
            if device == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            subprocess.run(cmd, check=True, env=env)
            self._append_profile_rows(
                output_dir / f"test_results_{name}.csv",
                sweep_csv,
                profile=profile,
                epoch=epoch,
                checkpoint_path=checkpoint_path,
            )
            self._append_profile_summary(
                output_dir / "test_results_profiles_summary.csv",
                sweep_csv,
                profile=profile,
                epoch=epoch,
                checkpoint_path=checkpoint_path,
            )

    @staticmethod
    def _epoch_pth_path(pl_module: LightningModule, epoch: int) -> Path:
        return Path(pl_module.train_config.output_dir) / "epoch_pth" / f"checkpoint_epoch_{epoch:03d}.pth"

    @staticmethod
    def _get_preferred_state_dict(trainer: Trainer, pl_module: LightningModule) -> dict[str, torch.Tensor]:
        for callback in trainer.callbacks:
            getter = getattr(callback, "get_ema_model_state_dict", None)
            if callable(getter):
                state_dict = getter()
                if state_dict is not None:
                    return state_dict

        raw = getattr(pl_module.model, "_orig_mod", None)
        if not isinstance(raw, torch.nn.Module):
            raw = pl_module.model
        return {k: v.detach().clone() for k, v in raw.state_dict().items()}

    @staticmethod
    def _train_config_with_class_names(trainer: Trainer, pl_module: LightningModule) -> Any:
        train_config = pl_module.train_config
        dataset_class_names = getattr(trainer.datamodule, "class_names", None)
        if (
            dataset_class_names is not None
            and hasattr(train_config, "model_copy")
            and getattr(train_config, "class_names", None) is None
        ):
            return train_config.model_copy(update={"class_names": dataset_class_names})
        return train_config

    @staticmethod
    def _force_real_yolo_test_split(datamodule: Any) -> None:
        """RF-DETR 1.7.1 maps YOLO test to val; replace it with real test."""

        train_config = getattr(datamodule, "train_config", None)
        model_config = getattr(datamodule, "model_config", None)
        if train_config is None or model_config is None:
            return
        if getattr(train_config, "dataset_file", None) != "yolo":
            return

        from rfdetr._namespace import _namespace_from_configs
        from rfdetr.datasets import build_dataset

        ns = _namespace_from_configs(model_config, train_config)
        datamodule._dataset_test = build_dataset("test", ns, model_config.resolution)

    @staticmethod
    def _append_metrics(path: Path, metrics: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {key: _to_plain_value(value) for key, value in metrics.items()}

        existing_fieldnames: list[str] = []
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                existing_fieldnames = next(reader, [])

        fieldnames = list(existing_fieldnames)
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

        rows: list[dict[str, Any]] = []
        if path.exists() and existing_fieldnames:
            with path.open("r", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if fieldnames != existing_fieldnames:
                for old_row in rows:
                    for key in fieldnames:
                        old_row.setdefault(key, "")
        rows.append(row)

        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @classmethod
    def _append_profile_rows(
        cls,
        path: Path,
        sweep_csv: Path,
        *,
        profile: dict[str, Any],
        epoch: int,
        checkpoint_path: Path,
    ) -> None:
        rows = cls._read_sweep_rows(sweep_csv)
        profile_name = str(profile["name"])
        match_iou = profile.get("iou_threshold", 0.5)
        for row in rows:
            row["epoch"] = epoch
            row["profile"] = profile_name
            row["match_iou_threshold"] = match_iou
            row["checkpoint"] = str(checkpoint_path)
            row["sweep_csv"] = str(sweep_csv)
            cls._append_metrics(path, row)

    @classmethod
    def _append_profile_summary(
        cls,
        path: Path,
        sweep_csv: Path,
        *,
        profile: dict[str, Any],
        epoch: int,
        checkpoint_path: Path,
    ) -> None:
        rows = cls._read_sweep_rows(sweep_csv)
        selected = cls._select_profile_row(rows, profile)
        selected["epoch"] = epoch
        selected["profile"] = str(profile["name"])
        selected["match_iou_threshold"] = profile.get("iou_threshold", 0.5)
        selected["selection"] = profile.get("selection", "best_f1")
        selected["checkpoint"] = str(checkpoint_path)
        selected["sweep_csv"] = str(sweep_csv)
        cls._append_metrics(path, selected)

    @staticmethod
    def _read_sweep_rows(path: Path) -> list[dict[str, Any]]:
        with path.open("r", newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    @staticmethod
    def _select_profile_row(rows: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
        if not rows:
            raise ValueError(f"external eval profile {profile.get('name')!r} produced no rows")

        selection = str(profile.get("selection", "best_f1"))
        if selection == "threshold":
            target = float(profile["selected_threshold"])
            return dict(
                min(
                    rows,
                    key=lambda row: abs(float(row.get("threshold", 0.0)) - target),
                )
            )
        if selection == "class_thresholds":
            targets = [float(item) for item in profile["selected_thresholds"]]
            if len(targets) != 3:
                raise ValueError("selected_thresholds must contain B/C/D thresholds")
            return dict(
                min(
                    rows,
                    key=lambda row: sum(
                        abs(float(row.get(f"threshold_class_{idx}", 0.0)) - target)
                        for idx, target in enumerate(targets)
                    ),
                )
            )
        if selection == "best_recall":
            return dict(
                max(
                    rows,
                    key=lambda row: (
                        float(row.get("recall", 0.0)),
                        float(row.get("precision", 0.0)),
                    ),
                )
            )
        if selection == "best_precision_at_min_recall":
            min_recall = float(profile["min_recall"])
            eligible = [row for row in rows if float(row.get("recall", 0.0)) >= min_recall]
            if not eligible:
                return dict(max(rows, key=lambda row: float(row.get("recall", 0.0))))
            return dict(max(eligible, key=lambda row: float(row.get("precision", 0.0))))
        return dict(max(rows, key=lambda row: float(row.get("f1", 0.0))))


def install_router_trainer_patch(
    *,
    save_epoch_pth: bool,
    test_each_epoch: bool,
    trainer_precision: str | None = None,
    external_eval_profiles: list[dict[str, Any]] | None = None,
    official_eval_dataset_dir: str = "",
    eval_device: str = "cpu",
) -> None:
    """Append router callbacks to RF-DETR's built Trainer at runtime."""

    import rfdetr.training as training_pkg
    import rfdetr.training.trainer as trainer_mod

    original_build_trainer = training_pkg.build_trainer
    if getattr(original_build_trainer, "_router_patch_installed", False):
        return

    def build_trainer_with_router_callbacks(*args: Any, **kwargs: Any) -> Trainer:
        if trainer_precision:
            kwargs["precision"] = trainer_precision
        trainer = original_build_trainer(*args, **kwargs)
        trainer.callbacks.append(
            RouterPerEpochEvalCallback(
                save_epoch_pth=save_epoch_pth,
                test_each_epoch=test_each_epoch,
                external_eval_profiles=external_eval_profiles,
                official_eval_dataset_dir=official_eval_dataset_dir,
                eval_device=eval_device,
            )
        )
        return trainer

    build_trainer_with_router_callbacks._router_patch_installed = True  # type: ignore[attr-defined]
    training_pkg.build_trainer = build_trainer_with_router_callbacks
    trainer_mod.build_trainer = build_trainer_with_router_callbacks


def _to_plain_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value
