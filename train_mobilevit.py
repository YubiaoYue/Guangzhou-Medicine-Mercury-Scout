#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a MobileViT classifier (ImageFolder) + Early Stopping (by val_loss).

Expected dataset structure:
  dataset_root/
    train/
      class_a/
      class_b/
      ...
    val/
      class_a/
      class_b/
      ...

Example:
  python train_mobilevit.py \
    --dataset_root /path/to/dataset_root \
    --pretrained_weights /path/to/mobilevit_s.pt \
    --num_classes 7 \
    --epochs 100 \
    --patience 5 \
    --batch_size 16 \
    --lr 1e-4 \
    --output_dir outputs_mobilevit \
    --save_best_acc
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

# IMPORTANT:
# - Your project should provide `model.py` with `mobile_vit_small` factory.
# - Example: from model import mobile_vit_small as create_model
from model import mobile_vit_small as create_model


# ----------------------------
# Reproducibility helpers
# ----------------------------
def seed_everything(seed: int = 10) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Strict determinism (may reduce speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ----------------------------
# Data
# ----------------------------
def build_transforms() -> Dict[str, transforms.Compose]:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    return {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]),
    }


def build_dataloaders(
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    int,
    int,
    Dict[str, int]
]:
    tfm = build_transforms()

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Val folder not found: {val_dir}")

    train_ds = datasets.ImageFolder(root=str(train_dir), transform=tfm["train"])
    val_ds = datasets.ImageFolder(root=str(val_dir), transform=tfm["val"])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, len(train_ds), len(val_ds), train_ds.class_to_idx


def save_class_indices(class_to_idx: Dict[str, int], save_path: Path) -> None:
    """Save index->class mapping for reproducibility."""
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=4, ensure_ascii=False)


# ----------------------------
# Model
# ----------------------------
def build_model(
    num_classes: int,
    pretrained_weights: Path | None,
    in_features: int = 640,
) -> nn.Module:
    """
    Build MobileViT model and replace classification head.

    Notes:
    - Your original code used: net.classifier.fc = Linear(640 -> num_classes)
    - Here we keep the same head replacement to stay consistent.
    """
    model = create_model()

    if pretrained_weights is not None:
        if not pretrained_weights.exists():
            raise FileNotFoundError(f"Pretrained weights not found: {pretrained_weights}")
        state = torch.load(str(pretrained_weights), map_location="cpu")
        model.load_state_dict(state)

    # Replace classifier head (consistent with your code)
    if not hasattr(model, "classifier") or not hasattr(model.classifier, "fc"):
        raise AttributeError(
            "Expected model to have `model.classifier.fc`. "
            "Please adapt head replacement according to your MobileViT implementation."
        )

    model.classifier.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    return model


# ----------------------------
# Train / Eval
# ----------------------------
def train_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    epoch_idx: int,
    epochs: int,
) -> float:
    model.train()
    running_loss = 0.0
    steps = len(loader)

    pbar = tqdm(loader, desc=f"train epoch[{epoch_idx}/{epochs}]", file=sys.stdout, leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(steps, 1)


@torch.no_grad()
def evaluate_with_loss(
    model: nn.Module,
    loader,
    device: torch.device,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    """
    Returns:
        avg_val_loss, val_accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="validate", file=sys.stdout, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MobileViT ImageFolder Training + Early Stopping")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root folder containing train/ and val/ subfolders.")
    parser.add_argument("--pretrained_weights", type=str, default="",
                        help="Path to pretrained weights (.pt/.pth). Leave empty to train from scratch.")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes.")
    parser.add_argument("--head_in_features", type=int, default=640,
                        help="In-features of classifier head (default 640 to match your code).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda:0 | cpu")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Dataloader workers. 0 means auto heuristic.")
    parser.add_argument("--output_dir", type=str, default="outputs_mobilevit",
                        help="Directory to save weights/logs.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience based on validation loss.")
    parser.add_argument("--save_best_acc", action="store_true",
                        help="If set, also save the best model by validation accuracy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pretrained_weights = args.pretrained_weights.strip()
    pretrained_weights_path = Path(pretrained_weights).expanduser().resolve() if pretrained_weights else None

    # dataloaders
    num_workers = args.num_workers if args.num_workers > 0 else min(os.cpu_count() or 1, 8, args.batch_size)
    print(f"[INFO] num_workers: {num_workers}")

    train_loader, val_loader, n_train, n_val, class_to_idx = build_dataloaders(
        dataset_root=dataset_root,
        batch_size=args.batch_size,
        num_workers=num_workers,
    )

    print(f"[INFO] Train images: {n_train}, Val images: {n_val}")
    print(f"[INFO] Found {len(class_to_idx)} classes: {sorted(class_to_idx.keys())}")

    # Save mapping for reproducibility
    save_class_indices(class_to_idx, output_dir / "class_indices.json")

    # model
    model = build_model(
        num_classes=args.num_classes,
        pretrained_weights=pretrained_weights_path,
        in_features=args.head_in_features,
    )
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Early stopping states
    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0
    patience = max(1, int(args.patience))
    best_model_path = weights_dir / "best_val_loss.pth"

    history: List[Tuple[float, float, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch_idx=epoch,
            epochs=args.epochs,
        )

        avg_val_loss, val_acc = evaluate_with_loss(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
        )

        print(
            f"[EPOCH {epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        history.append((train_loss, avg_val_loss, val_acc))

        # ---------- Early Stopping 检查（按 val_loss） ----------
        if avg_val_loss < best_val_loss - 1e-4:  # 加一点小 margin 防抖
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # 保存当前最优模型（按 val_loss）
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] Best val_loss improved -> {best_val_loss:.6f}. Saved: {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"[INFO] No val_loss improvement for {epochs_no_improve}/{patience} epoch(s). "
                  f"(best_val_loss={best_val_loss:.6f})")
            if epochs_no_improve >= patience:
                print(f"[EARLY STOPPING] Triggered at epoch {epoch}. Best val_loss={best_val_loss:.6f}")
                break

        # （可选）同时保存 best-val-acc
        if args.save_best_acc and (val_acc > best_val_acc):
            best_val_acc = val_acc
            best_acc_path = weights_dir / "best_val_acc.pth"
            torch.save(model.state_dict(), best_acc_path)
            print(f"[INFO] Best val_acc improved -> {best_val_acc:.6f}. Saved: {best_acc_path}")

    # save training log
    df = pd.DataFrame(history, columns=["train_loss", "val_loss", "val_acc"])
    df.to_csv(output_dir / "training_process.csv", index=False)
    print(f"[INFO] Training log saved: {output_dir / 'training_process.csv'}")
    print("[INFO] Finished Training.")


if __name__ == "__main__":
    main()
