"""Load the provided EventTSign splits and three-channel event images."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from eventsign.paths import resolve_project_path


def build_transform(image_size: int):
    """Keep the main model's original resize and tensor conversion."""
    return transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )


class EventSignDataset(Dataset):
    """Read ``train_labels.csv`` or ``test_labels.csv`` without resplitting.

    Required columns are ``sample_id,event_path,split,lighting,lighting_id,
    class_id``. Released CSVs also retain ``rgb_path`` for paired RGB users;
    this event-only loader does not require RGB images. Image paths are relative
    to ``root``; a relative root is anchored at the release project directory.
    Absolute root/image paths remain supported. Numeric class IDs are used
    directly, with no remapping. Extra CSV columns are ignored.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int,
        transform=None,
    ):
        self.root = resolve_project_path(root)
        self.split = split
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split!r}; expected 'train' or 'test'.")
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        csv_path = self.root / f"{split}_labels.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Label file not found: {csv_path}")
        self.samples = self._load_csv(csv_path)
        self.transform = transform or build_transform(image_size)

    def _load_csv(self, csv_path: Path) -> List[Dict[str, Any]]:
        required = {
            "sample_id", "event_path", "split", "lighting", "lighting_id", "class_id"
        }
        samples: List[Dict[str, Any]] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path} is missing columns: {', '.join(sorted(missing))}")
            for line, row in enumerate(reader, start=2):
                if any(row.get(key) is None or not row[key].strip() for key in required):
                    raise ValueError(f"{csv_path}:{line} has empty required fields.")
                try:
                    label = int(row["class_id"])
                    lighting_id = int(row["lighting_id"])
                except ValueError as exc:
                    raise ValueError(
                        f"{csv_path}:{line} requires integer class_id and lighting_id."
                    ) from exc
                if lighting_id not in {0, 1, 2}:
                    raise ValueError(
                        f"{csv_path}:{line} has lighting_id={lighting_id}; "
                        "expected 0 (Low), 1 (Normal), or 2 (High)."
                    )
                samples.append(
                    {
                        "sample_id": row["sample_id"],
                        "event_path": self.root / row["event_path"],
                        "split": row["split"],
                        "lighting": row["lighting"],
                        "lighting_id": lighting_id,
                        "label": label,
                    }
                )
        if not samples:
            raise ValueError(f"Label file contains no samples: {csv_path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.samples[index]
        path = item["event_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Event image not found: {path}")
        with Image.open(path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return {
            "image": image_tensor,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "sample_id": item["sample_id"],
            "lighting_id": torch.tensor(item["lighting_id"], dtype=torch.long),
            "lighting": item["lighting"],
            "split": item["split"],
        }


def _cfg_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _build_dataset(cfg, split: str) -> EventSignDataset:
    data_cfg = cfg.data
    if str(_cfg_get(data_cfg, "dataset", "eventsign")).lower() != "eventsign":
        raise ValueError("This release supports data.dataset='eventsign'.")
    if str(_cfg_get(data_cfg, "input_mode", "event")).lower() != "event":
        raise ValueError("This release requires data.input_mode='event'.")
    if int(_cfg_get(data_cfg, "input_channels", 3)) != 3:
        raise ValueError("This release requires three-channel event images.")
    if str(_cfg_get(data_cfg, "representation", "event_image")).lower() not in {
        "event_image", "image"
    }:
        raise ValueError("This release loads rendered event images.")
    dataset = EventSignDataset(
        root=_cfg_get(data_cfg, "root"),
        split=split,
        image_size=int(_cfg_get(data_cfg, "image_size", 224)),
    )
    num_classes = int(_cfg_get(cfg.model, "num_classes", 58))
    if num_classes <= 0:
        raise ValueError("model.num_classes must be positive.")
    labels = [item["label"] for item in dataset.samples]
    if min(labels) < 0 or max(labels) >= num_classes:
        raise ValueError(
            f"EventTSign {split} class_id range [{min(labels)}, {max(labels)}] "
            f"is incompatible with model.num_classes={num_classes}."
        )
    return dataset


def _build_loader(cfg, dataset: EventSignDataset, shuffle: bool) -> DataLoader:
    batch_size = int(_cfg_get(cfg.train, "batch_size", 32))
    workers = int(_cfg_get(cfg.data, "workers", 4))
    if batch_size <= 0 or workers < 0:
        raise ValueError("train.batch_size must be positive and data.workers nonnegative.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    train_dataset = _build_dataset(cfg, "train")
    test_dataset = _build_dataset(cfg, "test")
    print(
        f"[Data] EVENTSIGN | root={train_dataset.root} | "
        f"classes={_cfg_get(cfg.model, 'num_classes', 58)} | "
        f"train={len(train_dataset)} | test={len(test_dataset)} | "
        "representation=event_image"
    )
    return (
        _build_loader(cfg, train_dataset, shuffle=True),
        _build_loader(cfg, test_dataset, shuffle=False),
    )


def build_test_loader(cfg) -> DataLoader:
    """Build evaluation data without requiring the training CSV."""
    return _build_loader(cfg, _build_dataset(cfg, "test"), shuffle=False)
