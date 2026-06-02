"""Router-specific RF-DETR training callbacks.

These callbacks are intentionally kept outside the installed ``rfdetr`` package.
``train_rfdetr_router.py`` monkey-patches RF-DETR's trainer factory at runtime so
the upstream package remains reproducible from pip.
"""

from __future__ import annotations

import csv
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
    ) -> None:
        super().__init__()
        self.save_epoch_pth = save_epoch_pth
        self.test_each_epoch = test_each_epoch
        self.metric_csv_name = metric_csv_name
        self._running_test = False

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking or self._running_test:
            return
        if "val/mAP_50_95" not in trainer.callback_metrics:
            return

        epoch = int(trainer.current_epoch)
        if self.save_epoch_pth:
            self._save_epoch_pth(trainer, pl_module, epoch)
        if self.test_each_epoch:
            self._run_test_and_record(trainer, pl_module, epoch)

    def _save_epoch_pth(self, trainer: Trainer, pl_module: LightningModule, epoch: int) -> None:
        if not trainer.is_global_zero:
            return

        from rfdetr.training.callbacks.best_model import BestModelCallback

        output_dir = Path(pl_module.train_config.output_dir)
        epoch_dir = output_dir / "epoch_pth"
        epoch_dir.mkdir(parents=True, exist_ok=True)

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
        torch.save(payload, epoch_dir / f"checkpoint_epoch_{epoch:03d}.pth")

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


def install_router_trainer_patch(
    *,
    save_epoch_pth: bool,
    test_each_epoch: bool,
    trainer_precision: str | None = None,
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
