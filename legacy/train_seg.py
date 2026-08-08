from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


def print_metric_group(results: Any, attr_name: str, label: str) -> None:
    metrics = getattr(results, attr_name, None)
    if metrics is None:
        return
    print(f"{label}:")
    if getattr(metrics, "map50", None) is not None:
        print(f"  mAP@0.5: {metrics.map50:.4f}")
    if getattr(metrics, "map", None) is not None:
        print(f"  mAP@0.5:0.95: {metrics.map:.4f}")
    if getattr(metrics, "p", None) is not None:
        print(f"  Precision: {metrics.p:.4f}")
    if getattr(metrics, "r", None) is not None:
        print(f"  Recall: {metrics.r:.4f}")


def validate_checkpoint(model_path: str, data_yaml: str, imgsz: int, batch_size: int, device: str) -> None:
    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=imgsz, batch=batch_size, device=device)
    print_metric_group(results, "box", "Box metrics")
    print_metric_group(results, "seg", "Seg metrics")


def main(args: argparse.Namespace) -> None:
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")

    save_period = max(1, math.ceil(args.epochs / max(args.num_checkpoints, 1)))
    print(f"Checkpoint period: every {save_period} epochs")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project=args.output_dir,
        name=args.exp_name,
        exist_ok=True,
        pretrained=True,
        optimizer=args.optimizer,
        patience=args.patience,
        lr0=args.learning_rate,
        save_period=save_period,
        close_mosaic=args.close_mosaic,
    )

    if not args.test_checkpoints:
        return

    weights_dir = output_dir / args.exp_name / "weights"
    if not weights_dir.exists():
        print(f"Weights dir not found: {weights_dir}")
        return

    for checkpoint in sorted(weights_dir.glob("epoch*.pt")):
        print(f"\nValidating checkpoint: {checkpoint.name}")
        validate_checkpoint(str(checkpoint), args.data, args.imgsz, args.batch_size, args.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO segmentation training helper.")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt", help="model name or checkpoint path")
    parser.add_argument("--data", type=str, default="dataset_custom_absolute/data.yaml", help="dataset yaml path")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--imgsz", type=int, default=960, help="input image size")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--device", type=str, default="0", help="training device, e.g. 0 or cpu")
    parser.add_argument("--workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--output-dir", type=str, default="runs", help="output directory")
    parser.add_argument("--exp-name", type=str, default="custom_absolute_seg", help="experiment name")
    parser.add_argument("--optimizer", type=str, default="auto", help="optimizer name")
    parser.add_argument("--patience", type=int, default=30, help="early stopping patience")
    parser.add_argument("--learning-rate", type=float, default=0.005, help="initial learning rate")
    parser.add_argument("--num-checkpoints", type=int, default=10, help="number of periodic checkpoints")
    parser.add_argument("--close-mosaic", type=int, default=10, help="disable mosaic in final epochs")
    parser.add_argument("--test-checkpoints", action="store_true", help="validate saved checkpoints after training")
    main(parser.parse_args())
