"""Evaluate a main-model checkpoint on the EventSign test split."""

from __future__ import annotations

import argparse
import json

import torch

from data.dataset import build_test_loader
from eventsign.config import config_from_dict, load_config
from eventsign.paths import resolve_project_path
from models import build_model
from train import DEFAULT_CONFIG, print_metrics, resolve_device, run_epoch, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to last.pt or best.pt, relative to the project directory or absolute.",
    )
    parser.add_argument(
        "--config",
        help=(
            "Optional configuration override. Defaults to the checkpoint's saved "
            f"configuration, or {DEFAULT_CONFIG} for older checkpoints."
        ),
    )
    parser.add_argument(
        "--data-root", help="Override data.root; relative to the project directory or absolute."
    )
    parser.add_argument("--device", help="Override train.device, e.g. cpu or cuda:0.")
    parser.add_argument(
        "--output", help="Optional JSON destination, relative to the project directory or absolute."
    )
    args = parser.parse_args()

    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Expected a checkpoint dictionary containing a 'model' state dictionary.")
    if args.config:
        cfg = load_config(args.config)
    elif "config" in checkpoint:
        cfg = config_from_dict(checkpoint["config"])
    else:
        cfg = load_config(DEFAULT_CONFIG)
    if args.data_root:
        cfg.data.root = args.data_root
    device = resolve_device(cfg.train.device, args.device)
    set_seed(int(cfg.seed))

    # Evaluation gets every weight from the trained checkpoint. It does not
    # download or require an ImageNet/backbone initialization checkpoint.
    cfg.model.pretrained = False
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    test_loader = build_test_loader(cfg)
    metrics = run_epoch(model, test_loader, device)
    print_metrics("Test", metrics)
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "epoch": checkpoint.get("epoch"),
        "split": "test",
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        output = resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
