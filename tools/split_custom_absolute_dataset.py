from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将自制绝对面积数据集从 raw 划分为 YOLO 分割训练所需的 train/val 目录。"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset_custom_absolute"),
        help="数据集根目录",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="验证集比例",
    )
    parser.add_argument(
        "--seed-key",
        type=str,
        default="custom-absolute-seg-v1",
        help="稳定划分用的种子字符串",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="覆盖类别名，默认读取原 data.yaml 中的 names",
    )
    parser.add_argument(
        "--allow-empty-label",
        action="store_true",
        help="允许空标签文件进入训练集，用于保留负样本",
    )
    return parser.parse_args()


def list_raw_images(images_raw_dir: Path) -> Iterable[Path]:
    for path in sorted(images_raw_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def stable_split(stem: str, seed_key: str, val_ratio: float) -> str:
    key = f"{seed_key}:{stem}".encode("utf-8")
    bucket = int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_ratio else "train"


def load_class_names(dataset_dir: Path, override: List[str] | None) -> List[str]:
    if override:
        return override

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        return ["pothole"]

    names_line = ""
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            names_line = line.split(":", 1)[1].strip()
            break

    if not names_line:
        return ["pothole"]

    try:
        names = json.loads(names_line)
        if isinstance(names, list) and names:
            return [str(item) for item in names]
    except json.JSONDecodeError:
        pass
    return ["pothole"]


def build_yaml(dataset_dir: Path, class_names: List[str]) -> None:
    yaml_path = dataset_dir / "data.yaml"
    content = (
        f"path: ./{dataset_dir.name}\n"
        "train: manifests/train.txt\n"
        "val: manifests/val.txt\n"
        f"nc: {len(class_names)}\n"
        f"names: {json.dumps(class_names, ensure_ascii=False)}\n"
    )
    yaml_path.write_text(content, encoding="utf-8", newline="\n")


def validate_label_file(label_path: Path) -> Tuple[bool, str]:
    if not label_path.exists():
        return False, "missing_label"

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return True, "empty_label"

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) < 7:
            return False, f"line_{line_no}_too_short"
        if len(parts[1:]) % 2 != 0:
            return False, f"line_{line_no}_odd_polygon_coords"
        try:
            int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            return False, f"line_{line_no}_not_numeric"
        if any(v < 0 or v > 1 for v in coords):
            return False, f"line_{line_no}_coord_out_of_range"
    return True, "ok"


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    images_raw_dir = dataset_dir / "images" / "raw"
    labels_raw_dir = dataset_dir / "labels" / "raw"
    manifests_dir = dataset_dir / "manifests"

    if not images_raw_dir.exists() or not labels_raw_dir.exists():
        raise FileNotFoundError("未找到 images/raw 或 labels/raw，请先运行数据整理脚本。")

    manifests_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(dataset_dir, args.class_names)

    stats: Dict[str, int] = {
        "images_total": 0,
        "train_images": 0,
        "val_images": 0,
        "empty_labels": 0,
        "missing_labels": 0,
        "invalid_labels": 0,
        "skipped_empty_labels": 0,
    }
    invalid_samples: List[Dict[str, str]] = []
    train_images: List[str] = []
    val_images: List[str] = []

    for image_path in list_raw_images(images_raw_dir):
        stats["images_total"] += 1
        label_path = labels_raw_dir / f"{image_path.stem}.txt"
        is_valid, reason = validate_label_file(label_path)

        if not is_valid:
            if reason == "missing_label":
                stats["missing_labels"] += 1
            else:
                stats["invalid_labels"] += 1
            invalid_samples.append({"image": image_path.name, "reason": reason})
            continue

        if reason == "empty_label":
            stats["empty_labels"] += 1
            if not args.allow_empty_label:
                stats["skipped_empty_labels"] += 1
                continue

        split = stable_split(image_path.stem, args.seed_key, args.val_ratio)
        stats[f"{split}_images"] += 1
        image_ref = image_path.resolve().as_posix()
        if split == "train":
            train_images.append(image_ref)
        else:
            val_images.append(image_ref)

    build_yaml(dataset_dir, class_names)
    (manifests_dir / "train.txt").write_text(
        "\n".join(train_images) + ("\n" if train_images else ""),
        encoding="utf-8",
        newline="\n",
    )
    (manifests_dir / "val.txt").write_text(
        "\n".join(val_images) + ("\n" if val_images else ""),
        encoding="utf-8",
        newline="\n",
    )

    split_manifest = {
        "dataset_dir": str(dataset_dir),
        "val_ratio": args.val_ratio,
        "seed_key": args.seed_key,
        "allow_empty_label": bool(args.allow_empty_label),
        "class_names": class_names,
        "stats": stats,
        "invalid_samples": invalid_samples,
    }
    manifest_path = manifests_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )

    print(f"Images total: {stats['images_total']}")
    print(f"Train images: {stats['train_images']}")
    print(f"Val images: {stats['val_images']}")
    print(f"Empty labels: {stats['empty_labels']}")
    print(f"Skipped empty labels: {stats['skipped_empty_labels']}")
    print(f"Missing labels: {stats['missing_labels']}")
    print(f"Invalid labels: {stats['invalid_labels']}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
