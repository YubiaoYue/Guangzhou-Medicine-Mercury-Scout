#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MobileViT single-image inference script (load best_val_acc.pth).

Example:
  python predict_mobilevit.py \
    --image_path test.jpg \
    --weights outputs_mobilevit/weights/best_val_acc.pth \
    --class_indices outputs_mobilevit/class_indices.json \
    --num_classes 7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Must match training
from model import mobile_vit_small as create_model


# ----------------------------
# Utilities
# ----------------------------
def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_class_indices(json_path: Path) -> Dict[int, str]:
    """Load index -> class_name mapping."""
    with json_path.open("r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    return {int(k): v for k, v in idx_to_class.items()}


def build_transform() -> transforms.Compose:
    """Must be identical to validation transform used in training."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])


def build_model(
    num_classes: int,
    weights_path: Path,
    head_in_features: int = 640,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Build MobileViT model and load best-accuracy weights."""
    model = create_model()

    # Replace classification head (must match training)
    model.classifier.fc = nn.Linear(
        in_features=head_in_features,
        out_features=num_classes,
        bias=True
    )

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


# ----------------------------
# Inference
# ----------------------------
@torch.no_grad()
def predict_image(
    image_path: Path,
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    idx_to_class: Dict[int, str],
    topk: int = 5,
):
    """Predict class probabilities for a single image."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    topk = min(topk, probs.numel())
    top_probs, top_indices = torch.topk(probs, k=topk)

    results = []
    for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
        results.append({
            "class_index": idx,
            "class_name": idx_to_class.get(idx, "Unknown"),
            "probability": p
        })

    return probs.cpu().tolist(), results


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MobileViT Image Classification Inference")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to input image.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to best_val_acc.pth (or best_acc.pth).")
    parser.add_argument("--class_indices", type=str, required=True,
                        help="Path to class_indices.json generated during training.")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of classes.")
    parser.add_argument("--head_in_features", type=int, default=640,
                        help="Classifier head input features (default: 640).")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto | cuda:0 | cpu")
    parser.add_argument("--topk", type=int, default=5,
                        help="Top-K predictions to display.")
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    weights_path = Path(args.weights)
    class_indices_path = Path(args.class_indices)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not class_indices_path.exists():
        raise FileNotFoundError(f"class_indices.json not found: {class_indices_path}")

    device = get_device(args.device)
    print(f"[INFO] Using device: {device}")

    idx_to_class = load_class_indices(class_indices_path)
    transform = build_transform()

    model = build_model(
        num_classes=args.num_classes,
        weights_path=weights_path,
        head_in_features=args.head_in_features,
        device=device,
    )

    probs, topk_results = predict_image(
        image_path=image_path,
        model=model,
        transform=transform,
        device=device,
        idx_to_class=idx_to_class,
        topk=args.topk,
    )

    print("\n===== Prediction Results (best_val_acc) =====")
    for item in topk_results:
        print(f"Class: {item['class_name']:>20s} | "
              f"Prob: {item['probability']:.4f}")

    print("\n----- Full probability distribution -----")
    for idx, p in enumerate(probs):
        class_name = idx_to_class.get(idx, "Unknown")
        print(f"[{idx:02d}] {class_name:>20s}: {p:.4f}")


if __name__ == "__main__":
    main()
