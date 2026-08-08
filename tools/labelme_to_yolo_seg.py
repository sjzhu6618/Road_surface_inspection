from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 Labelme 多边形标注转换为 YOLO segmentation 标签。"
    )
    parser.add_argument(
        "--labelme-dir",
        type=Path,
        required=True,
        help="Labelme 导出目录，包含 json 和对应图片",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("dataset_custom_absolute/images/raw"),
        help="目标图片目录，默认写入 dataset_custom_absolute/images/raw",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("dataset_custom_absolute/labels/raw"),
        help="目标标签目录，默认写入 dataset_custom_absolute/labels/raw",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["pothole"],
        help="类别列表，例如 pothole longitudinal_crack",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="如果 Labelme 目录中的图片不在目标目录，则自动复制过去",
    )
    return parser.parse_args()


def iter_labelme_jsons(labelme_dir: Path) -> Iterable[Path]:
    return sorted(labelme_dir.glob("*.json"))


def normalize_points(
    points: Sequence[Sequence[float]],
    image_width: int,
    image_height: int,
) -> List[Tuple[float, float]]:
    normalized: List[Tuple[float, float]] = []
    for point in points:
        if len(point) != 2:
            continue
        x = min(max(float(point[0]) / max(image_width, 1), 0.0), 1.0)
        y = min(max(float(point[1]) / max(image_height, 1), 0.0), 1.0)
        normalized.append((x, y))
    return normalized


def resolve_image_path(labelme_dir: Path, data: Dict[str, object], json_path: Path) -> Path:
    image_name = str(data.get("imagePath") or "")
    if image_name:
        candidate = (json_path.parent / image_name).resolve()
        if candidate.exists():
            return candidate
    stem = json_path.stem
    for suffix in IMAGE_SUFFIXES:
        candidate = labelme_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"未找到与 {json_path.name} 对应的图片文件。")


def main() -> None:
    args = parse_args()
    class_to_id = {name: idx for idx, name in enumerate(args.class_names)}

    args.images_dir.mkdir(parents=True, exist_ok=True)
    args.labels_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    missing_images = 0

    for json_path in iter_labelme_jsons(args.labelme_dir):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_width = int(data.get("imageWidth") or 0)
        image_height = int(data.get("imageHeight") or 0)
        if image_width <= 0 or image_height <= 0:
            skipped += 1
            continue

        try:
            image_path = resolve_image_path(args.labelme_dir, data, json_path)
        except FileNotFoundError:
            missing_images += 1
            continue

        rows: List[str] = []
        for shape in data.get("shapes", []):
            if not isinstance(shape, dict):
                continue
            label = str(shape.get("label") or "").strip()
            shape_type = str(shape.get("shape_type") or "polygon").strip().lower()
            points = shape.get("points") or []

            if label not in class_to_id:
                continue
            if shape_type not in {"polygon", "linestrip"}:
                continue

            norm_points = normalize_points(points, image_width, image_height)
            if len(norm_points) < 3:
                continue

            coords: List[str] = []
            for x, y in norm_points:
                coords.append(f"{x:.6f}")
                coords.append(f"{y:.6f}")
            rows.append(f"{class_to_id[label]} " + " ".join(coords))

        if args.copy_images:
            dst_image = args.images_dir / image_path.name
            if not dst_image.exists():
                shutil.copy2(image_path, dst_image)

        label_path = args.labels_dir / f"{image_path.stem}.txt"
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8", newline="\n")
        converted += 1

    print(f"Converted json files: {converted}")
    print(f"Skipped json files: {skipped}")
    print(f"Missing images: {missing_images}")
    print(f"Labels dir: {args.labels_dir.resolve()}")


if __name__ == "__main__":
    main()
