"""Train the main Hyper-TSign model with the original CE/AdamW protocol.

The original experiment evaluates the test split each epoch and chooses the
best checkpoint by test accuracy. This script preserves that selection rule.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from data.dataset import build_dataloaders
from eventsign.config import load_config
from eventsign.paths import resolve_project_path
from models import build_model


LIGHTING_ID_TO_NAME = {0: "Low", 1: "Normal", 2: "High"}
DEFAULT_CONFIG = "configs/hyper_tsign.yaml"
SELECTION_PROTOCOL = "Evaluate test each epoch; select best by test accuracy."


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def resolve_device(configured: str, override: str | None = None) -> torch.device:
    requested = override or configured
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        if override is not None:
            raise RuntimeError("CUDA was requested with --device but is unavailable.")
        print("CUDA is unavailable; using CPU.")
        requested = "cpu"
    return torch.device(requested)


class LightingAccuracyMeter:
    """Accuracies use the original CSV lighting IDs: Low/Normal/High = 0/1/2."""

    def __init__(self):
        self.correct = defaultdict(int)
        self.total = defaultdict(int)

    @torch.no_grad()
    def update(self, predictions, targets, lighting_ids):
        matches = predictions.detach().cpu() == targets.detach().cpu()
        lighting_ids = lighting_ids.detach().cpu()
        self.correct["All"] += matches.sum().item()
        self.total["All"] += targets.numel()
        for lighting_id, name in LIGHTING_ID_TO_NAME.items():
            mask = lighting_ids == lighting_id
            self.correct[name] += matches[mask].sum().item()
            self.total[name] += mask.sum().item()

    def compute(self):
        return {
            name: self.correct[name] / self.total[name] if self.total[name] else 0.0
            for name in ["All", *LIGHTING_ID_TO_NAME.values()]
        }


def build_optimizer(cfg, model):
    if str(getattr(cfg.train, "optimizer", "adamw")).lower() != "adamw":
        raise ValueError("The main-model training protocol requires train.optimizer='adamw'.")
    return AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        betas=tuple(cfg.train.betas),
    )


def build_scheduler(cfg, optimizer):
    if str(getattr(cfg.train, "scheduler", "cosine")).lower() != "cosine":
        raise ValueError("The main-model training protocol requires train.scheduler='cosine'.")
    warmup_epochs = max(int(cfg.train.warmup_epochs), 0)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(int(cfg.train.epochs) - warmup_epochs, 1),
        eta_min=float(cfg.train.min_lr),
    )
    if warmup_epochs == 0:
        return cosine
    warmup = LinearLR(
        optimizer,
        start_factor=float(cfg.train.warmup_start_factor),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def run_epoch(model, loader, device, optimizer=None):
    """Run one classification epoch; providing an optimizer enables training."""
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    lighting_meter = LightingAccuracyMeter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if training else "test", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (predictions == labels).sum().item()
            total += images.size(0)
            if "lighting_id" in batch:
                lighting_meter.update(predictions, labels, batch["lighting_id"])

    if total == 0:
        raise ValueError("The data loader yielded no samples.")
    metrics = {"loss": total_loss / total, "accuracy": total_correct / total, "count": total}
    if lighting_meter.total["All"]:
        metrics["lighting_accuracy"] = lighting_meter.compute()
        metrics["lighting_count"] = dict(lighting_meter.total)
    return metrics


def print_metrics(split, metrics):
    print(f"{split}: loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}")
    if "lighting_accuracy" in metrics:
        detail = " | ".join(
            f"{name}: {accuracy * 100:.2f}%"
            for name, accuracy in metrics["lighting_accuracy"].items()
        )
        print(f"[{split} lighting] {detail}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help="Configuration path, relative to the project directory or absolute.",
    )
    parser.add_argument(
        "--data-root", help="Override data.root; relative to the project directory or absolute."
    )
    parser.add_argument("--device", help="Override train.device, e.g. cpu or cuda:0.")
    parser.add_argument(
        "--no-pretrained", action="store_true", help="Train from random backbone weights."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    if args.no_pretrained:
        cfg.model.pretrained = False
    output_dir = resolve_project_path(cfg.output_dir) / cfg.experiment_name
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(
            f"Run output {output_dir.resolve()} already exists and is not an empty directory. "
            "Choose a new experiment_name or output_dir; resuming or overwriting a run "
            "is not supported."
        )
    device = resolve_device(cfg.train.device, args.device)
    cfg.train.device = str(device)
    set_seed(int(cfg.seed))
    train_loader, test_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = cfg.to_dict()
    (output_dir / "config.json").write_text(
        json.dumps(config_dict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"Protocol: {SELECTION_PROTOCOL}")
    print(f"Output: {output_dir.resolve()}")

    # Preserve the original strict comparison and initial value. A run whose
    # test accuracy is always zero produces last.pt but no best.pt.
    best_acc = 0.0
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as log_file:
        for epoch in range(int(cfg.train.epochs)):
            train_metrics = run_epoch(model, train_loader, device, optimizer)
            test_metrics = run_epoch(model, test_loader, device)
            scheduler.step()
            print(f"Epoch {epoch + 1:03d}/{cfg.train.epochs:03d}")
            print_metrics("Train", train_metrics)
            print_metrics("Test", test_metrics)
            record = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "test": test_metrics,
                "lr": optimizer.param_groups[0]["lr"],
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

            is_best = bool(cfg.train.save_best) and test_metrics["accuracy"] > best_acc
            if is_best:
                best_acc = test_metrics["accuracy"]
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "config": config_dict,
                "seed": int(cfg.seed),
                "num_classes": int(cfg.model.num_classes),
                "test_acc": test_metrics["accuracy"],
                "best_test_acc": best_acc,
                "metrics": record,
                "selection_protocol": SELECTION_PROTOCOL,
            }
            class_to_idx = getattr(train_loader.dataset, "class_to_idx", None)
            if class_to_idx is not None:
                checkpoint["class_to_idx"] = dict(class_to_idx)
            torch.save(checkpoint, output_dir / "last.pt")
            if is_best:
                torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
