import argparse
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["pothole", "D00", "D10", "D20", "D40"]
RDD_CLASS_MAP = {
    "D00": 1,
    "D10": 2,
    "D20": 3,
    "D40": 4,
}
IGNORED_RDD_CLASSES = {"Repair", "Block crack"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="将现有 pothole 数据集与 RDD2022 XML 数据集合并为新的 YOLO 数据集"
    )
    parser.add_argument(
        "--pothole-root",
        type=Path,
        default=Path("dataset"),
        help="现有 pothole YOLO 数据集根目录",
    )
    parser.add_argument(
        "--rdd-root",
        type=Path,
        default=Path(r"G:\DS\RDD022\RDD2022\RDD2022_all_countries"),
        help="RDD2022_all_countries 根目录",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset_merged"),
        help="新的合并数据集输出目录",
    )
    parser.add_argument(
        "--lendd-root",
        type=Path,
        default=Path(r"G:\DS\LenDD"),
        help="LenDD 手拍数据根目录，要求包含图片与 json/ 标注目录",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="RDD train 数据拆分到 val 的比例，范围 0~1",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="输出图片时使用复制还是硬链接",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="生成前清空 output-root",
    )
    return parser.parse_args()


def ensure_layout(root: Path):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def reset_output(root: Path):
    if root.exists():
        shutil.rmtree(root)
    ensure_layout(root)


def stable_bucket(name: str, val_ratio: float) -> str:
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if value < val_ratio else "train"


def write_yolo_label(label_path: Path, rows):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(row + "\n")


def copy_or_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def find_image(image_dir: Path, stem: str):
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def convert_box(size, box):
    width, height = size
    xmin, ymin, xmax, ymax = box
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def points_to_bbox(points, width: int, height: int):
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    xmin = clamp(min(xs), 0.0, float(width))
    xmax = clamp(max(xs), 0.0, float(width))
    ymin = clamp(min(ys), 0.0, float(height))
    ymax = clamp(max(ys), 0.0, float(height))
    return xmin, ymin, xmax, ymax


def convert_xml_to_yolo(xml_path: Path, stats: Counter):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        stats["xml_missing_size"] += 1
        return []

    width = int(size.findtext("width", default="0"))
    height = int(size.findtext("height", default="0"))
    if width <= 0 or height <= 0:
        stats["xml_bad_size"] += 1
        return []

    lines = []
    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        if class_name in IGNORED_RDD_CLASSES:
            stats[f"ignored_{class_name}"] += 1
            continue
        if class_name not in RDD_CLASS_MAP:
            stats[f"unknown_{class_name or 'EMPTY'}"] += 1
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            stats["xml_missing_bbox"] += 1
            continue

        xmin = float(bbox.findtext("xmin", default="0"))
        ymin = float(bbox.findtext("ymin", default="0"))
        xmax = float(bbox.findtext("xmax", default="0"))
        ymax = float(bbox.findtext("ymax", default="0"))
        x_center, y_center, box_width, box_height = convert_box(
            (width, height), (xmin, ymin, xmax, ymax)
        )
        lines.append(
            f"{RDD_CLASS_MAP[class_name]} "
            f"{x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )
        stats[f"class_{class_name}"] += 1

    return lines


def convert_lendd_json_to_yolo(json_path: Path, stats: Counter):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    width = int(data.get("imageWidth", 0) or 0)
    height = int(data.get("imageHeight", 0) or 0)
    if width <= 0 or height <= 0:
        stats["lendd_bad_size"] += 1
        return []

    shapes = data.get("shapes", [])
    lines = []
    for shape in shapes:
        shape_type = str(shape.get("shape_type") or "polygon")
        points = shape.get("points") or []
        if len(points) < 2:
            stats["lendd_bad_shape"] += 1
            continue

        xmin, ymin, xmax, ymax = points_to_bbox(points, width, height)
        if math.isclose(xmin, xmax) or math.isclose(ymin, ymax):
            stats["lendd_zero_area"] += 1
            continue

        x_center, y_center, box_width, box_height = convert_box(
            (width, height), (xmin, ymin, xmax, ymax)
        )
        lines.append(
            f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )
        stats["class_pothole_lendd"] += 1
        stats[f"lendd_shape_{shape_type}"] += 1

    return lines


def merge_pothole_dataset(src_root: Path, dst_root: Path, copy_mode: str, stats: Counter):
    for split in ("train", "val"):
        image_dir = src_root / "images" / split
        label_dir = src_root / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue

        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                stats["pothole_missing_label"] += 1
                continue

            target_stem = f"pothole_{image_path.stem}"
            target_image = dst_root / "images" / split / f"{target_stem}{image_path.suffix.lower()}"
            target_label = dst_root / "labels" / split / f"{target_stem}.txt"

            copy_or_link(image_path, target_image, copy_mode)
            shutil.copy2(label_path, target_label)
            stats[f"pothole_{split}_images"] += 1


def merge_rdd_country(
    country_root: Path,
    prefix: str,
    dst_root: Path,
    val_ratio: float,
    copy_mode: str,
    stats: Counter,
):
    train_root = country_root / "train"
    image_dir = train_root / "images"
    xml_dir = train_root / "annotations" / "xmls"
    if not image_dir.exists() or not xml_dir.exists():
        stats[f"{prefix}_missing_train_layout"] += 1
        return

    xml_files = sorted(xml_dir.glob("*.xml"))
    for xml_path in xml_files:
        stem = xml_path.stem
        image_path = find_image(image_dir, stem)
        if image_path is None:
            stats[f"{prefix}_missing_image"] += 1
            continue

        yolo_lines = convert_xml_to_yolo(xml_path, stats)
        if not yolo_lines:
            stats[f"{prefix}_empty_after_filter"] += 1
            continue

        split = stable_bucket(f"{prefix}:{stem}", val_ratio)
        target_stem = f"{prefix}_{stem}"
        target_image = dst_root / "images" / split / f"{target_stem}{image_path.suffix.lower()}"
        target_label = dst_root / "labels" / split / f"{target_stem}.txt"

        copy_or_link(image_path, target_image, copy_mode)
        write_yolo_label(target_label, yolo_lines)
        stats[f"{prefix}_{split}_images"] += 1
        stats[f"{prefix}_{split}_boxes"] += len(yolo_lines)


def merge_lendd_dataset(
    lendd_root: Path,
    dst_root: Path,
    val_ratio: float,
    copy_mode: str,
    stats: Counter,
):
    image_files = sorted(
        [
            path
            for path in lendd_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )
    json_dir = lendd_root / "json"
    if not json_dir.exists():
        stats["lendd_missing_json_dir"] += 1
        return

    for image_path in image_files:
        json_path = json_dir / f"{image_path.stem}.json"
        if not json_path.exists():
            stats["lendd_missing_json"] += 1
            continue

        yolo_lines = convert_lendd_json_to_yolo(json_path, stats)
        if not yolo_lines:
            stats["lendd_empty_after_filter"] += 1
            continue

        split = stable_bucket(f"lendd:{image_path.stem}", val_ratio)
        target_stem = f"lendd_{image_path.stem}"
        target_image = dst_root / "images" / split / f"{target_stem}{image_path.suffix.lower()}"
        target_label = dst_root / "labels" / split / f"{target_stem}.txt"

        copy_or_link(image_path, target_image, copy_mode)
        write_yolo_label(target_label, yolo_lines)
        stats[f"lendd_{split}_images"] += 1
        stats[f"lendd_{split}_boxes"] += len(yolo_lines)


def count_output_labels(dst_root: Path):
    class_counts = defaultdict(int)
    for split in ("train", "val"):
        label_dir = dst_root / "labels" / split
        if not label_dir.exists():
            continue
        for label_file in label_dir.glob("*.txt"):
            with label_file.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
    return class_counts


def print_summary(output_root: Path, stats: Counter):
    print("\n=== 合并完成 ===")
    print(f"输出目录: {output_root}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")

    class_counts = count_output_labels(output_root)
    print("\n=== 最终类别统计 ===")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"{class_id} {class_name}: {class_counts.get(class_id, 0)}")


def main():
    args = parse_args()
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio 必须在 [0, 1) 范围内")

    if args.clean:
        reset_output(args.output_root)
    else:
        ensure_layout(args.output_root)

    stats = Counter()

    merge_pothole_dataset(args.pothole_root, args.output_root, args.copy_mode, stats)
    merge_rdd_country(
        args.rdd_root / "China_Drone",
        "china_drone",
        args.output_root,
        args.val_ratio,
        args.copy_mode,
        stats,
    )
    merge_rdd_country(
        args.rdd_root / "China_MotorBike",
        "china_motorbike",
        args.output_root,
        args.val_ratio,
        args.copy_mode,
        stats,
    )
    merge_lendd_dataset(
        args.lendd_root,
        args.output_root,
        args.val_ratio,
        args.copy_mode,
        stats,
    )

    print_summary(args.output_root, stats)


if __name__ == "__main__":
    main()
