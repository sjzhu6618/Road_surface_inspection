from __future__ import annotations

import argparse

from ultralytics import YOLO


def print_metric_group(results, attr_name: str, label: str) -> None:
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


def main(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    print(f"Model: {args.model}")
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        split=args.split,
    )
    print_metric_group(results, "box", "Box metrics")
    print_metric_group(results, "seg", "Seg metrics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO segmentation validation helper.")
    parser.add_argument("--model", type=str, required=True, help="model checkpoint path")
    parser.add_argument("--data", type=str, default="dataset_custom_absolute/data.yaml", help="dataset yaml path")
    parser.add_argument("--imgsz", type=int, default=960, help="input image size")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--device", type=str, default="", help="device, e.g. 0 or cpu")
    parser.add_argument("--split", type=str, default="val", help="dataset split")
    main(parser.parse_args())
