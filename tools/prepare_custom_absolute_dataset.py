from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image


NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="整理自制坑洼图片数据集，并生成绝对面积换算所需的元数据清单。"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="原始图片目录，例如 G:\\数据集\\自制数据集",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_custom_absolute"),
        help="整理后的输出目录",
    )
    parser.add_argument(
        "--camera-height-cm",
        type=float,
        default=133.0,
        help="拍摄高度（厘米）",
    )
    parser.add_argument(
        "--focal-length-mm-eq",
        type=float,
        default=28.0,
        help="35mm 等效焦距（毫米）",
    )
    parser.add_argument(
        "--full-frame-diagonal-mm",
        type=float,
        default=43.2666,
        help="35mm 全画幅对角线长度（毫米）",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["pothole"],
        help="待标注类别列表，默认仅 pothole",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="整理图片时使用复制或硬链接",
    )
    return parser.parse_args()


def load_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    try:
        raw = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(raw)
    values: List[str] = []
    for si in root.findall("x:si", NS):
        texts = []
        for t in si.iterfind(".//x:t", NS):
            texts.append(t.text or "")
        values.append("".join(texts))
    return values


def parse_xlsx_table(xlsx_path: Path) -> List[Dict[str, str]]:
    if not xlsx_path.exists():
        return []

    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = load_shared_strings(zf)
        raw = zf.read("xl/worksheets/sheet1.xml")
        root = ET.fromstring(raw)

    rows: List[List[str]] = []
    for row in root.findall(".//x:sheetData/x:row", NS):
        values: List[str] = []
        for cell in row.findall("x:c", NS):
            cell_type = cell.attrib.get("t", "")
            value_node = cell.find("x:v", NS)
            value = value_node.text if value_node is not None else ""
            if cell_type == "s" and value:
                value = shared_strings[int(value)]
            values.append(value or "")
        rows.append(values)

    if not rows:
        return []

    header = rows[0]
    records: List[Dict[str, str]] = []
    for values in rows[1:]:
        row = {header[idx]: (values[idx] if idx < len(values) else "") for idx in range(len(header))}
        if any(str(v).strip() for v in row.values()):
            records.append(row)
    return records


def safe_float(value: str) -> Optional[float]:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def estimate_shape_area_cm2(row: Dict[str, str]) -> Optional[float]:
    length = safe_float(row.get("长度", ""))
    width = safe_float(row.get("宽度", ""))
    shape = str(row.get("形状类型", "")).strip()
    if length is None:
        return None

    if shape == "三角形" and width is not None:
        return 0.5 * length * width
    if shape == "圆形":
        radius = length / 2.0
        return math.pi * radius * radius
    if width is not None:
        return length * width
    return None


def image_records(source_dir: Path) -> Iterable[Path]:
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        yield from sorted(source_dir.glob(suffix))


def build_dataset_yaml(output_dir: Path, class_names: List[str]) -> None:
    yaml_path = output_dir / "data.yaml"
    names_json = json.dumps(class_names, ensure_ascii=False)
    content = (
        f"path: ./{output_dir.name}\n"
        "train: images/raw\n"
        "val: images/raw\n"
        f"nc: {len(class_names)}\n"
        f"names: {names_json}\n"
    )
    yaml_path.write_text(content, encoding="utf-8", newline="\n")


def ensure_image(target_path: Path, source_path: Path, copy_mode: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return
    if copy_mode == "hardlink":
        os.link(source_path, target_path)
    else:
        shutil.copy2(source_path, target_path)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir
    output_dir = args.output_dir
    images_dir = output_dir / "images" / "raw"
    labels_dir = output_dir / "labels" / "raw"
    manifests_dir = output_dir / "manifests"

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    xlsx_files = sorted(source_dir.glob("*.xlsx"))
    reference_rows = parse_xlsx_table(xlsx_files[0]) if xlsx_files else []

    ground_diagonal_cm = (
        args.camera_height_cm * args.full_frame_diagonal_mm / args.focal_length_mm_eq
    )

    image_manifest: List[Dict[str, object]] = []
    image_paths = list(image_records(source_dir))
    if not image_paths:
        raise FileNotFoundError(f"未在 {source_dir} 找到图片文件。")

    for image_path in image_paths:
        with Image.open(image_path) as img:
            width_px, height_px = img.size
        diag_px = math.hypot(width_px, height_px)
        cm_per_pixel = ground_diagonal_cm / diag_px
        pixel_area_cm2 = cm_per_pixel ** 2
        ground_width_cm = width_px * cm_per_pixel
        ground_height_cm = height_px * cm_per_pixel

        target_image = images_dir / image_path.name
        ensure_image(target_image, image_path, args.copy_mode)
        (labels_dir / f"{image_path.stem}.txt").write_text("", encoding="utf-8", newline="\n")

        image_manifest.append(
            {
                "image_name": image_path.name,
                "original_path": str(image_path),
                "prepared_path": str(target_image),
                "width_px": width_px,
                "height_px": height_px,
                "orientation": "landscape" if width_px >= height_px else "portrait",
                "camera_height_cm": round(args.camera_height_cm, 4),
                "focal_length_mm_eq": round(args.focal_length_mm_eq, 4),
                "estimated_ground_diagonal_cm": round(ground_diagonal_cm, 4),
                "estimated_ground_width_cm": round(ground_width_cm, 4),
                "estimated_ground_height_cm": round(ground_height_cm, 4),
                "cm_per_pixel": round(cm_per_pixel, 6),
                "pixel_area_cm2": round(pixel_area_cm2, 8),
                "area_formula": "absolute_area_m2 = polygon_pixel_count * pixel_area_cm2 / 10000",
                "length_formula": "absolute_length_cm = pixel_length * cm_per_pixel",
            }
        )

    for row in reference_rows:
        row["估算面积_cm2"] = (
            f"{estimate_shape_area_cm2(row):.4f}" if estimate_shape_area_cm2(row) is not None else ""
        )
        row["关联图片名"] = row.get("关联图片名", "")

    config = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "camera_height_cm": args.camera_height_cm,
        "focal_length_mm_eq": args.focal_length_mm_eq,
        "full_frame_diagonal_mm": args.full_frame_diagonal_mm,
        "estimated_ground_diagonal_cm": round(ground_diagonal_cm, 4),
        "default_cm_per_pixel": round(image_manifest[0]["cm_per_pixel"], 6),
        "default_pixel_area_cm2": round(image_manifest[0]["pixel_area_cm2"], 8),
        "assumption": "默认假设手机镜头近似垂直朝下拍摄，且 28mm 为可靠的 35mm 等效焦距。",
        "warning": "如果拍摄存在明显倾斜、畸变或高度波动，绝对面积会产生系统误差，建议后续补做标定板校准。",
    }

    (manifests_dir / "absolute_scale_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    (manifests_dir / "image_manifest.json").write_text(
        json.dumps(image_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )

    with (manifests_dir / "image_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_manifest[0].keys()))
        writer.writeheader()
        writer.writerows(image_manifest)

    if reference_rows:
        with (manifests_dir / "reference_measurements.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(reference_rows[0].keys()))
            writer.writeheader()
            writer.writerows(reference_rows)

    build_dataset_yaml(output_dir, args.class_names)

    print(f"Image count: {len(image_manifest)}")
    print(f"Reference measurement rows: {len(reference_rows)}")
    print(f"Default cm_per_pixel: {config['default_cm_per_pixel']}")
    print(f"Default pixel_area_cm2: {config['default_pixel_area_cm2']}")
    print(f"Output dir: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
