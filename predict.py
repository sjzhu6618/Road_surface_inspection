import argparse
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
from ultralytics import YOLO


# -----------------------------
# 基础工具
# -----------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_image_file(path: str) -> bool:
    return path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff"))


def is_video_file(path: str) -> bool:
    return path.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"))


def resolve_depth_model_path(args) -> str:
    if args.depth_model_path:
        if os.path.isdir(args.depth_model_path):
            return args.depth_model_path
        raise FileNotFoundError(f"未找到深度模型目录: {args.depth_model_path}")

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "depth_model",
        script_dir.parent / "depth_model",
        Path.cwd() / "depth_model",
    ]
    for cand in candidates:
        if cand.is_dir():
            return str(cand)
    raise FileNotFoundError("未找到 depth_model 目录，请使用 --depth-model-path 指定")


_FONT_CACHE: Dict[Tuple[str, int], Any] = {}
CLASS_DISPLAY_NAMES = {
    "pothole": "坑洼",
    "D00": "纵向裂缝",
    "D10": "横向裂缝",
    "D20": "网状裂缝",
    "D40": "坑槽/破损",
}


def normalize_category_name(name: Optional[str]) -> str:
    raw = (name or "").strip()
    if not raw:
        return "未知病害"
    return CLASS_DISPLAY_NAMES.get(raw, raw)


def get_record_category(record: Dict[str, Any]) -> str:
    return normalize_category_name(record.get("category"))


def is_crack_category(name: Optional[str]) -> bool:
    raw = (name or "").strip().upper()
    return raw in {"D00", "D10", "D20", "D40"} or "CRACK" in raw


def get_category_thresholds(args, category_name: Optional[str]) -> Tuple[int, int]:
    min_bbox_area = int(getattr(args, "min_bbox_area_px", 900))
    min_mask_pixels = int(getattr(args, "min_mask_pixels", 240))
    if is_crack_category(category_name):
        min_bbox_area = max(160, min_bbox_area // 4)
        min_mask_pixels = max(40, min_mask_pixels // 4)
    return min_bbox_area, min_mask_pixels


def get_font(font_path: Optional[str], size: int):
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates.extend([
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ])
    for path in candidates:
        key = (path, size)
        if key in _FONT_CACHE:
            return _FONT_CACHE[key]
        try:
            font = ImageFont.truetype(path, size=size)
            _FONT_CACHE[key] = font
            return font
        except Exception:
            continue
    return None


def draw_text(img_bgr: np.ndarray, text: str, pos: Tuple[int, int], color=(255, 255, 255),
              font_size: int = 24, bg_color: Optional[Tuple[int, int, int]] = None,
              font_path: Optional[str] = None) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = get_font(font_path, font_size)
    x, y = pos

    if font is not None:
        bbox = draw.textbbox((x, y), text, font=font)
        if bg_color is not None:
            draw.rounded_rectangle(
                (bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4),
                radius=8,
                fill=(bg_color[2], bg_color[1], bg_color[0]),
            )
        draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if bg_color is not None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size / 32.0, 2)
        cv2.rectangle(img_bgr, (x - 6, y - th - 8), (x + tw + 6, y + 6), bg_color, -1)
    cv2.putText(img_bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_size / 32.0, color, 2, cv2.LINE_AA)
    return img_bgr


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def make_task_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def collect_image_files(source_dir: str) -> List[str]:
    root = Path(source_dir)
    if not root.is_dir():
        return []
    files = [str(p) for p in root.rglob("*") if p.is_file() and is_image_file(p.name)]
    files.sort()
    return files


def get_task_archive_dir(output_dir: str, task_run_id: str) -> str:
    archive_dir = os.path.join(output_dir, "tasks", task_run_id)
    ensure_dir(archive_dir)
    return archive_dir


def copy_artifact_if_exists(src_path: str, dst_dir: str) -> None:
    if os.path.exists(src_path):
        copy2(src_path, os.path.join(dst_dir, os.path.basename(src_path)))


def preprocess_frame(frame: np.ndarray, args) -> np.ndarray:
    out = frame.copy()
    if args.enable_denoise:
        out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)
    if args.enable_clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=args.clahe_clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return out


def build_road_roi_mask(frame_shape: Tuple[int, int, int], args) -> Tuple[np.ndarray, np.ndarray]:
    h, w = frame_shape[:2]
    top_y = int(h * args.road_roi_top_ratio)
    bottom_y = int(h * args.road_roi_bottom_ratio)
    top_half_width = int(w * args.road_roi_top_width_ratio / 2.0)
    bottom_half_width = int(w * args.road_roi_bottom_width_ratio / 2.0)
    center_x = int(w * args.road_roi_center_x_ratio)

    pts = np.array([
        [max(0, center_x - top_half_width), top_y],
        [min(w - 1, center_x + top_half_width), top_y],
        [min(w - 1, center_x + bottom_half_width), bottom_y],
        [max(0, center_x - bottom_half_width), bottom_y],
    ], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    return mask, pts


def overlay_road_roi(frame: np.ndarray, polygon: np.ndarray, alpha: float = 0.12) -> np.ndarray:
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, polygon, (70, 180, 70))
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def calc_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a.astype(float)
    bx1, by1, bx2, by2 = box_b.astype(float)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def box_center(box: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = box.astype(float)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def absolute_pixel_area_m2(args) -> float:
    meters_per_pixel = (float(args.calibrated_scene_width_cm) / 100.0) / max(1, int(args.calibrated_image_width_px))
    return meters_per_pixel * meters_per_pixel


def perspective_pixel_area_m2(bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int], args) -> float:
    x1, y1, x2, y2 = bbox
    t = 0.0
    pixel_area_cm2 = args.pixel_area_cm2_far + (args.pixel_area_cm2_near - args.pixel_area_cm2_far) * (t ** args.perspective_gamma)
    bbox_pixel_area = max(1.0, float((x2 - x1) * (y2 - y1)))
    return bbox_pixel_area * absolute_pixel_area_m2(args)


def make_pseudo_mask(depth_map: np.ndarray, bbox: Tuple[int, int, int, int], args) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    roi = depth_map[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi_norm = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    roi_blur = cv2.GaussianBlur(roi_norm, (5, 5), 0)
    _, mask = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(roi_blur[mask > 0]) > np.mean(roi_blur[mask == 0]) if np.any(mask == 0) else False:
        mask = 255 - mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.mask_morph_kernel, args.mask_morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    best_idx = None
    best_area = 0
    roi_h, roi_w = roi.shape[:2]
    roi_cx, roi_cy = roi_w / 2.0, roi_h / 2.0
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area < args.min_mask_pixels:
            continue
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        w = stats[idx, cv2.CC_STAT_WIDTH]
        h = stats[idx, cv2.CC_STAT_HEIGHT]
        cx = x + w / 2.0
        cy = y + h / 2.0
        dist = (cx - roi_cx) ** 2 + (cy - roi_cy) ** 2
        score = area - 0.002 * dist
        if score > best_area:
            best_area = score
            best_idx = idx

    if best_idx is None:
        return None

    comp_mask = (labels == best_idx).astype(np.uint8) * 255
    full_mask = np.zeros(depth_map.shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = comp_mask
    return full_mask


def local_reference_ring(depth_map: np.ndarray, bbox: Tuple[int, int, int, int], args) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    h, w = depth_map.shape[:2]
    pad_x = int((x2 - x1) * args.reference_ring_ratio)
    pad_y = int((y2 - y1) * args.reference_ring_ratio)
    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y)
    ex2 = min(w, x2 + pad_x)
    ey2 = min(h, y2 + pad_y)

    outer = np.zeros((h, w), dtype=np.uint8)
    inner = np.zeros((h, w), dtype=np.uint8)
    outer[ey1:ey2, ex1:ex2] = 255
    inner[y1:y2, x1:x2] = 255
    ring = cv2.subtract(outer, inner)
    return ring if np.count_nonzero(ring) > 0 else None


def calc_damage_metrics(depth_map: np.ndarray, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int], args,
                        mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    x1, y1, x2, y2 = bbox
    h, w = depth_map.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))

    if mask is None:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

    ring = local_reference_ring(depth_map, (x1, y1, x2, y2), args)
    roi_values = depth_map[(mask > 0)]
    ref_values = depth_map[(ring > 0)] if ring is not None else None

    if roi_values.size == 0:
        return {
            "area_m2": 0.0,
            "max_depth_drop": 0.0,
            "mean_depth_drop": 0.0,
            "volume_m3": 0.0,
            "urgency_score": 0.0,
            "reference_depth": 0.0,
            "mask_pixels": 0,
        }

    reference_depth = float(np.median(ref_values)) if ref_values is not None and ref_values.size > 10 else float(np.median(roi_values))
    depth_drop = np.clip(reference_depth - roi_values, 0, None)
    if depth_drop.size == 0:
        depth_drop = np.array([0.0], dtype=np.float32)

    mean_depth_drop = float(np.percentile(depth_drop, 75))
    max_depth_drop = float(np.percentile(depth_drop, 95))

    depth_cm = np.clip(mean_depth_drop * args.depth_scale_cm, 0.0, args.max_depth_cm)
    area_bbox_m2 = perspective_pixel_area_m2((x1, y1, x2, y2), frame_shape, args)

    mask_pixels = int(np.count_nonzero(mask))
    area_m2 = np.clip(area_bbox_m2, 0.0, args.max_area_m2)
    volume_m3 = np.clip(area_m2 * (depth_cm / 100.0), 0.0, args.max_volume_m3)

    area_score = min(100.0, (area_m2 / max(args.score_area_ref_m2, 1e-6)) * 100.0)
    depth_score = min(100.0, (depth_cm / max(args.score_depth_ref_cm, 1e-6)) * 100.0)
    volume_score = min(100.0, (volume_m3 / max(args.score_volume_ref_m3, 1e-6)) * 100.0)
    urgency_score = area_score * args.score_weight_area + depth_score * args.score_weight_depth + volume_score * args.score_weight_volume

    return {
        "area_m2": round(float(area_m2), 4),
        "max_depth_drop": round(float(depth_cm), 2),
        "mean_depth_drop": round(float(depth_cm), 2),
        "volume_m3": round(float(volume_m3), 5),
        "urgency_score": round(float(urgency_score), 2),
        "reference_depth": round(reference_depth, 3),
        "mask_pixels": mask_pixels,
    }


def risk_to_status(score: float) -> Tuple[str, Tuple[int, int, int]]:
    if score >= 20:
        return "重度", (40, 60, 230)
    if score >= 10:
        return "中度", (0, 165, 255)
    return "轻度", (0, 215, 255)


# -----------------------------
# 视频级跟踪
# -----------------------------


@dataclass
class Track:
    track_id: int
    category: str
    bbox: np.ndarray
    conf: float
    first_frame: int
    last_frame: int
    missed: int
    hits: int
    confirmed: bool
    best_area_m2: float
    best_depth_drop: float
    best_volume_m3: float
    highest_score: float
    highest_status: str


class SimpleDamageTracker:
    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 12, min_confirm_hits: int = 2, center_dist_ratio: float = 0.08):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_confirm_hits = min_confirm_hits
        self.center_dist_ratio = center_dist_ratio
        self.tracks: Dict[int, Track] = {}
        self.finished_tracks: List[Track] = []
        self.next_id = 1

    def _match_score(self, det_box: np.ndarray, track_box: np.ndarray, frame_shape: Tuple[int, int]) -> float:
        iou = calc_iou(det_box, track_box)
        dcx, dcy = box_center(det_box)
        tcx, tcy = box_center(track_box)
        diag = max(1.0, (frame_shape[0] ** 2 + frame_shape[1] ** 2) ** 0.5)
        dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5 / diag
        det_area = max(1.0, float((det_box[2] - det_box[0]) * (det_box[3] - det_box[1])))
        trk_area = max(1.0, float((track_box[2] - track_box[0]) * (track_box[3] - track_box[1])))
        area_ratio = min(det_area, trk_area) / max(det_area, trk_area)
        if iou < self.iou_threshold and dist > self.center_dist_ratio:
            return -1.0
        return iou * 0.65 + area_ratio * 0.20 + max(0.0, 1.0 - dist / max(self.center_dist_ratio, 1e-6)) * 0.15

    def update(self, detections: List[Dict[str, Any]], frame_index: int, frame_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        unmatched_track_ids = set(self.tracks.keys())

        for det in detections:
            det_box = np.array(det["bbox"], dtype=np.float32)
            best_track_id = None
            best_score = -1.0

            for track_id, track in self.tracks.items():
                score = self._match_score(det_box, track.bbox, frame_shape)
                if score > best_score:
                    best_score = score
                    best_track_id = track_id

            if best_track_id is not None and best_score >= 0:
                track = self.tracks[best_track_id]
                track.bbox = det_box
                track.conf = max(track.conf, float(det["confidence"]))
                track.last_frame = frame_index
                track.missed = 0
                track.hits += 1
                if track.hits >= self.min_confirm_hits:
                    track.confirmed = True
                track.best_area_m2 = max(track.best_area_m2, det["metrics"]["area_m2"])
                track.best_depth_drop = max(track.best_depth_drop, det["metrics"]["max_depth_drop"])
                track.best_volume_m3 = max(track.best_volume_m3, det["metrics"]["volume_m3"])
                if det["metrics"]["urgency_score"] >= track.highest_score:
                    track.highest_score = det["metrics"]["urgency_score"]
                    track.highest_status = det["status"]
                    track.category = det.get("category", track.category)
                det["track_id"] = track.track_id
                det["confirmed_track"] = track.confirmed
                unmatched_track_ids.discard(track.track_id)
            else:
                track_id = self.next_id
                self.next_id += 1
                track = Track(
                    track_id=track_id,
                    category=det.get("category", ""),
                    bbox=det_box,
                    conf=float(det["confidence"]),
                    first_frame=frame_index,
                    last_frame=frame_index,
                    missed=0,
                    hits=1,
                    confirmed=self.min_confirm_hits <= 1,
                    best_area_m2=det["metrics"]["area_m2"],
                    best_depth_drop=det["metrics"]["max_depth_drop"],
                    best_volume_m3=det["metrics"]["volume_m3"],
                    highest_score=det["metrics"]["urgency_score"],
                    highest_status=det["status"],
                )
                self.tracks[track_id] = track
                det["track_id"] = track_id
                det["confirmed_track"] = track.confirmed

        to_delete = []
        for track_id in list(unmatched_track_ids):
            track = self.tracks[track_id]
            track.missed += 1
            if track.missed > self.max_missed:
                self.finished_tracks.append(track)
                to_delete.append(track_id)
        for track_id in to_delete:
            self.tracks.pop(track_id, None)

        return detections

    def finalize(self) -> List[Dict[str, Any]]:
        all_tracks = self.finished_tracks + list(self.tracks.values())
        summary = []
        for track in all_tracks:
            if not track.confirmed:
                continue
            summary.append({
                "track_id": track.track_id,
                "category": track.category,
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "持续帧数": track.hits,
                "最高置信度": round(track.conf, 3),
                "最大受损面积_m2": round(track.best_area_m2, 4),
                "最大深度落差_cm": round(track.best_depth_drop, 2),
                "最大填补体积_m3": round(track.best_volume_m3, 5),
                "最高风险评分": round(track.highest_score, 2),
                "最高风险等级": track.highest_status,
            })
        return summary


# -----------------------------
# 模型与推理
# -----------------------------


def init_models(args):
    print(f"[*] 正在加载 YOLO 模型: {args.model}")
    yolo_model = YOLO(args.model)

    depth_model_path = resolve_depth_model_path(args)
    print(f"[*] 正在加载深度估计模型: {depth_model_path}")
    device = 0 if torch.cuda.is_available() and args.device != 'cpu' else -1
    depth_estimator = pipeline(task="depth-estimation", model=depth_model_path, device=device)
    print("[+] 所有模型加载完毕！\n")
    return yolo_model, depth_estimator



SEVERITY_COLOR_IMAGE = {
    "轻度": (0, 215, 255),   # BGR 亮黄
    "中度": (0, 165, 255),   # BGR 橙色
    "重度": (65, 85, 235),   # BGR 偏红
}

IMAGE_SEVERITY_VISUAL = {
    "轻度": {
        "box": (0, 215, 255),
        "fill": (0, 190, 255),
        "label_rgb": (255, 196, 40),
        "inner": (20, 145, 225),
    },
    "中度": {
        "box": (0, 165, 255),
        "fill": (0, 150, 245),
        "label_rgb": (255, 155, 35),
        "inner": (0, 115, 215),
    },
    "重度": {
        "box": (65, 85, 235),
        "fill": (72, 98, 228),
        "label_rgb": (232, 86, 86),
        "inner": (42, 52, 190),
    },
}

IMAGE_LABEL_TEXT_COLOR = (255, 255, 255)

def parse_yolo_result_image(result, offset=(0, 0)) -> List[Dict[str, Any]]:
    detections = []
    boxes = result.boxes if result.boxes is not None else []
    names = result.names if hasattr(result, "names") and isinstance(result.names, dict) else {}
    names = result.names if hasattr(result, "names") else {}
    ox, oy = offset
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        cls_id = int(box.cls[0]) if box.cls is not None else 0
        detections.append({
            "bbox": [x1 + ox, y1 + oy, x2 + ox, y2 + oy],
            "confidence": conf,
            "cls_id": cls_id,
            "cls_name": names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id),
        })
    return detections


def iou_xyxy_image(box1, box2) -> float:
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def nms_detections_image(detections: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: List[Dict[str, Any]] = []
    while ordered:
        best = ordered.pop(0)
        kept.append(best)
        ordered = [d for d in ordered if iou_xyxy_image(d["bbox"], best["bbox"]) < iou_threshold]
    return kept


def run_yolo_predict_image(yolo_model, image: np.ndarray, args) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    base = yolo_model.predict(
        source=image,
        conf=args.conf_threshold,
        iou=args.iou_threshold,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
        augment=args.augment,
        max_det=args.max_det,
    )
    detections.extend(parse_yolo_result_image(base[0]))

    if getattr(args, 'enable_tiling', False):
        h, w = image.shape[:2]
        tile = max(256, int(args.tile_size))
        overlap = float(args.tile_overlap)
        step = max(64, int(tile * (1.0 - overlap)))
        for y in range(0, h, step):
            for x in range(0, w, step):
                x2 = min(w, x + tile); y2 = min(h, y + tile)
                crop = image[y:y2, x:x2]
                if crop.shape[0] < 80 or crop.shape[1] < 80:
                    continue
                rs = yolo_model.predict(
                    source=crop,
                    conf=max(0.05, args.conf_threshold * 0.75),
                    iou=args.iou_threshold,
                    imgsz=min(args.imgsz, max(crop.shape[:2])),
                    device=args.device,
                    verbose=False,
                    augment=args.augment,
                    max_det=args.max_det,
                )
                detections.extend(parse_yolo_result_image(rs[0], offset=(x, y)))

    return nms_detections_image(detections, getattr(args, 'merge_iou_threshold', args.iou_threshold))


def keep_largest_component_image(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    max_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == max_label).astype(np.uint8)


def build_pseudo_mask_image(frame: np.ndarray, bbox: List[int], args) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mean_val = float(np.mean(blur))
    dark_offset = int(getattr(args, 'mask_dark_offset', 10))
    thresh_val = max(0, min(255, int(mean_val - dark_offset)))
    _, mask_dark = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)

    grad = cv2.Laplacian(blur, cv2.CV_32F)
    grad = np.uint8(np.clip(np.abs(grad), 0, 255))
    edge_threshold = int(getattr(args, 'mask_edge_threshold', 18))
    _, mask_edge = cv2.threshold(grad, edge_threshold, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(mask_dark, mask_edge)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = (mask > 0).astype(np.uint8)

    if mask.sum() > 0:
        mask = keep_largest_component_image(mask)

    if int(mask.sum()) < int(getattr(args, 'min_mask_pixels', 240)):
        mask = np.zeros_like(gray, dtype=np.uint8)
        center = ((x2 - x1) // 2, (y2 - y1) // 2)
        axes = (max(4, (x2 - x1) // 2), max(4, (y2 - y1) // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    return mask


def compute_depth_map_image(frame: np.ndarray, depth_estimator) -> np.ndarray:
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    depth_output = depth_estimator(pil_img)
    depth_map = np.array(depth_output["depth"], dtype=np.float32)
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze()
    return depth_map


def compute_depth_metrics_image(depth_map: np.ndarray, bbox: List[int], local_mask: np.ndarray) -> Dict[str, float]:
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))
    roi_depth = depth_map[y1:y2, x1:x2]
    if roi_depth.size == 0:
        return {"reference_depth": 0.0, "mean_depth_delta": 0.0, "max_depth_delta": 0.0}

    mask_bool = local_mask.astype(bool)
    if mask_bool.shape != roi_depth.shape:
        mask_bool = cv2.resize(local_mask.astype(np.uint8), (roi_depth.shape[1], roi_depth.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    damage_values = roi_depth[mask_bool] if mask_bool.any() else roi_depth.reshape(-1)

    pad_x = max(5, int((x2 - x1) * 0.25)); pad_y = max(5, int((y2 - y1) * 0.25))
    ex1, ey1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    ex2, ey2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    expanded = depth_map[ey1:ey2, ex1:ex2]
    surround_mask = np.ones(expanded.shape[:2], dtype=np.uint8)
    surround_mask[(y1 - ey1):(y2 - ey1), (x1 - ex1):(x2 - ex1)] = 0
    surround_values = expanded[surround_mask.astype(bool)]
    if surround_values.size == 0:
        surround_values = roi_depth.reshape(-1)
    reference_depth = float(np.median(surround_values))
    delta_values = np.abs(damage_values - reference_depth)
    return {
        "reference_depth": reference_depth,
        "mean_depth_delta": float(np.median(delta_values)) if delta_values.size else 0.0,
        "max_depth_delta": float(np.percentile(delta_values, 90)) if delta_values.size else 0.0,
    }

# ==============================================================================
# 修改核心：面积使用矩形框长乘宽
# ==============================================================================
def estimate_area_and_volume_image(mask_pixels: int, bbox: List[int], frame_shape, depth_metrics: Dict[str, float], args) -> Dict[str, float]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    cy = ((y1 + y2) / 2.0) / max(h, 1)

    calibrated_scene_width_cm = float(getattr(args, 'calibrated_scene_width_cm', 0.0) or 0.0)
    calibrated_image_width_px = int(getattr(args, 'calibrated_image_width_px', 0) or 0)
    if calibrated_scene_width_cm > 0 and calibrated_image_width_px > 0 and w == calibrated_image_width_px:
        meters_per_pixel = (calibrated_scene_width_cm / calibrated_image_width_px) / 100.0
    else:
        perspective_factor = np.clip(1.14 - 0.42 * cy, 0.78, 1.12)
        meters_per_pixel = getattr(args, 'base_meters_per_pixel', 0.0088) * perspective_factor

    # 【核心修改点：这里使用矩形框的像素面积，不再使用 mask_pixels】
    bbox_pixel_area = max(1, (x2 - x1) * (y2 - y1))
    area_m2 = float(bbox_pixel_area) * (meters_per_pixel ** 2)

    depth_scale = getattr(args, 'depth_unit_scale', 0.00012)
    mean_depth_m = float(depth_metrics['mean_depth_delta']) * depth_scale
    max_depth_m = float(depth_metrics['max_depth_delta']) * depth_scale

    mean_depth_m = float(np.clip(mean_depth_m, 0.0, 0.045))
    max_depth_m = float(np.clip(max_depth_m, mean_depth_m, 0.070))

    volume_m3 = area_m2 * mean_depth_m
    volume_m3 = float(np.clip(volume_m3, 0.0, 0.020))

    return {
        'area_m2': float(area_m2),
        'mean_depth_m': float(mean_depth_m),
        'max_depth_m': float(max_depth_m),
        'volume_m3': float(volume_m3),
        'mean_depth_cm': float(mean_depth_m * 100.0),
        'max_depth_cm': float(max_depth_m * 100.0),
        'meters_per_pixel': float(meters_per_pixel),
    }


def score_damage_image(area_m2: float, mean_depth_m: float, volume_m3: float) -> Tuple[float, str]:
    area_score = min(100.0, (area_m2 / 0.22) * 100.0)
    depth_score = min(100.0, (mean_depth_m / 0.055) * 100.0)
    volume_score = min(100.0, (volume_m3 / 0.010) * 100.0)

    score = 0.46 * area_score + 0.3 * depth_score + 0.24 * volume_score

    if score >= 20:
        return float(score), '重度'
    if score >= 10:
        return float(score), '中度'
    return float(score), '轻度'


def overlay_mask_image(frame: np.ndarray, bbox: List[int], local_mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.28) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    overlay = frame.copy()
    region = overlay[y1:y2, x1:x2]
    mask_bool = local_mask.astype(bool)
    if mask_bool.shape != region.shape[:2]:
        mask_bool = cv2.resize(local_mask.astype(np.uint8), (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    region[mask_bool] = (np.array(region[mask_bool], dtype=np.float32) * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    overlay[y1:y2, x1:x2] = region
    return overlay


def annotate_scene_image(frame: np.ndarray, records: List[Dict[str, Any]], masks: List[np.ndarray], args) -> np.ndarray:
    out = frame.copy()

    for rec, mask in zip(records, masks):
        style = IMAGE_SEVERITY_VISUAL.get(rec['status'], IMAGE_SEVERITY_VISUAL['轻度'])
        box_color = style['box']
        fill_color = style['fill']
        inner_color = style['inner']
        label_rgb = style['label_rgb']

        out = overlay_mask_image(out, rec['bbox'], mask, fill_color, alpha=0.28)

        x1, y1, x2, y2 = rec['bbox']
        region_mask = mask.astype(np.uint8)
        if region_mask.ndim == 2 and region_mask.size > 0:
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                shifted = [c + np.array([[[x1, y1]]], dtype=c.dtype) for c in contours]
                cv2.drawContours(out, shifted, -1, box_color, 3, lineType=cv2.LINE_AA)
                cv2.drawContours(out, shifted, -1, inner_color, 1, lineType=cv2.LINE_AA)

        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 3, lineType=cv2.LINE_AA)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(out, (cx, cy), 4, box_color, -1, lineType=cv2.LINE_AA)

    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    tiny_font = get_font(args.font_path, max(13, args.font_size - 7))

    for rec in records:
        style = IMAGE_SEVERITY_VISUAL.get(rec['status'], IMAGE_SEVERITY_VISUAL['轻度'])
        x1, y1, x2, y2 = rec['bbox']
        label = f"#{rec['序号']} {get_record_category(rec)} · {rec['status']}"
        label_w = max(118, int(len(label) * 15))
        y_box = max(8, y1 - 30)
        draw.rounded_rectangle(
            (x1, y_box, min(x1 + label_w, pil.width - 8), y_box + 24),
            radius=9,
            fill=style['label_rgb']
        )
        if tiny_font is not None:
            draw.text((x1 + 8, y_box + 3), label, fill=IMAGE_LABEL_TEXT_COLOR, font=tiny_font)

        sub = f"面积 {rec['metrics']['area_m2']:.4f}㎡"
        if tiny_font is not None:
            draw.text(
                (x1 + 2, min(y2 + 4, pil.height - 18)),
                sub,
                fill=(250, 250, 250),
                font=tiny_font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def compose_dashboard_view_image(base_image: np.ndarray, depth_map: np.ndarray, records: List[Dict[str, Any]], frame_id: str, inference_time_ms: float, args) -> np.ndarray:
    h, w = base_image.shape[:2]
    panel_w = max(380, int(w * 0.42))
    canvas = Image.new('RGB', (w + panel_w, h), (12, 18, 28))
    canvas.paste(Image.fromarray(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)), (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(args.font_path, 26)
    text_font = get_font(args.font_path, 15)
    small_font = get_font(args.font_path, 13)

    draw.text((w + 20, 18), '路面病害多维量化评估面板', fill=(235, 240, 247), font=title_font)
    draw.text((w + 20, 54), f'来源帧：{frame_id}', fill=(165, 178, 195), font=small_font)
    draw.text((w + 20, 74), f'推理耗时：{inference_time_ms:.1f} ms', fill=(165, 178, 195), font=small_font)

    total_count = len(records)
    total_area = sum(r['metrics']['area_m2'] for r in records)
    total_volume = sum(r['metrics']['volume_m3'] for r in records)
    severe_count = sum(1 for r in records if r['status'] == '重度')

    cards = [
        ('识别病害数', f'{total_count}', (69, 136, 255)),
        ('重度病害数', f'{severe_count}', (220, 60, 60)),
        ('估算受损面积', f'{total_area:.4f} ㎡', (62, 201, 147)),
        ('估算填补体积', f'{total_volume:.5f} m³', (245, 153, 35)),
    ]
    card_w = (panel_w - 60) // 2
    card_h = 82
    y_cards = 108
    for idx, (title, value, accent) in enumerate(cards):
        row = idx // 2; col = idx % 2
        x0 = w + 20 + col * (card_w + 20); y0 = y_cards + row * (card_h + 20)
        draw.rounded_rectangle((x0, y0, x0 + card_w, y0 + card_h), radius=18, fill=(18, 28, 45), outline=(40, 58, 90))
        draw.rounded_rectangle((x0 + 12, y0 + 14, x0 + 18, y0 + card_h - 14), radius=3, fill=accent)
        draw.text((x0 + 28, y0 + 16), title, fill=(190, 205, 220), font=small_font)
        draw.text((x0 + 28, y0 + 42), value, fill=(245, 248, 250), font=text_font)

    y = y_cards + 2 * (card_h + 20) + 8
    if depth_map is not None and args.show_depth_preview:
        depth_preview = cv2.applyColorMap(cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_TURBO)
        preview = cv2.resize(depth_preview, (panel_w - 40, max(128, h // 5)))
        canvas.paste(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)), (w + 20, y + 28))
        draw.text((w + 20, y), '深度热力图概览', fill=(235, 240, 247), font=text_font)
        y += max(128, h // 5) + 42

    draw.text((w + 20, y), '目标清单', fill=(235, 240, 247), font=text_font)
    y += 28
    max_rows = max(1, (h - y - 10) // 42)
    for rec in records[:max_rows]:
        line1 = f"#{rec['序号']} {get_record_category(rec)} {rec['status']}  置信度{rec['confidence']:.2f}"
        line2 = f"面积{rec['metrics']['area_m2']:.4f}㎡ 深度{rec['metrics']['max_depth_cm']:.2f}cm 体积{rec['metrics']['volume_m3']:.5f}m³"
        draw.text((w + 20, y), line1, fill=(240, 240, 240), font=small_font)
        y += 18
        draw.text((w + 20, y), line2, fill=(182, 204, 226), font=small_font)
        y += 24
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

def contained_dedup(records: List[Dict[str, Any]], overlap_threshold: float = 0.88, area_ratio_threshold: float = 0.55,
                    conf_tolerance: float = 0.15) -> List[Dict[str, Any]]:
    keep = [True] * len(records)
    for i in range(len(records)):
        if not keep[i]:
            continue
        bi = np.array(records[i]["bbox"], dtype=np.float32)
        ai = max(1.0, float((bi[2] - bi[0]) * (bi[3] - bi[1])))
        for j in range(len(records)):
            if i == j or not keep[j]:
                continue
            bj = np.array(records[j]["bbox"], dtype=np.float32)
            aj = max(1.0, float((bj[2] - bj[0]) * (bj[3] - bj[1])))
            inter = calc_iou(bi, bj) * (ai + aj - calc_iou(bi, bj) * 0)
            inter_x1 = max(bi[0], bj[0]); inter_y1 = max(bi[1], bj[1])
            inter_x2 = min(bi[2], bj[2]); inter_y2 = min(bi[3], bj[3])
            iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
            inter_area = iw * ih
            small_area = min(ai, aj)
            cover_ratio = inter_area / max(1e-6, small_area)
            area_ratio = small_area / max(ai, aj)

            if cover_ratio >= overlap_threshold and area_ratio <= area_ratio_threshold:
                if ai >= aj and records[i]["confidence"] + conf_tolerance >= records[j]["confidence"]:
                    keep[j] = False
                elif aj > ai and records[j]["confidence"] + conf_tolerance >= records[i]["confidence"]:
                    keep[i] = False
                    break
    return [r for r, k in zip(records, keep) if k]


def process_frame(frame: np.ndarray, yolo_model, depth_estimator, args, frame_id="img"):
    start_time = time.time()
    frame_data_list = []
    work_frame = preprocess_frame(frame, args)
    roi_mask, roi_polygon = build_road_roi_mask(work_frame.shape, args)

    results = yolo_model.predict(
        source=work_frame,
        conf=args.conf_threshold,
        iou=args.iou_threshold,
        imgsz=args.imgsz,
        device=args.device,
        augment=args.augment,
        verbose=False,
    )
    result = results[0]
    names = result.names if hasattr(result, "names") and isinstance(result.names, dict) else {}
    if not names and hasattr(yolo_model, "names") and isinstance(yolo_model.names, dict):
        names = yolo_model.names
    annotated_frame = frame.copy()
    if args.show_road_roi:
        annotated_frame = overlay_road_roi(annotated_frame, roi_polygon, alpha=0.10)

    depth_map = None
    depth_colormap = None
    boxes = result.boxes if result.boxes is not None else []
    if len(boxes) > 0:
        pil_img = Image.fromarray(cv2.cvtColor(work_frame, cv2.COLOR_BGR2RGB))
        depth_output = depth_estimator(pil_img)
        depth_map = np.array(depth_output["depth"], dtype=np.float32)
        max_depth = max(float(depth_map.max()), 1e-6)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_map, alpha=255.0 / max_depth),
            cv2.COLORMAP_INFERNO,
        )

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0]) if box.cls is not None else 0
            cls_name = names.get(cls_id, str(cls_id))
            category_name = normalize_category_name(cls_name)

            if conf < args.valid_conf_threshold:
                continue
            min_bbox_area, min_mask_pixels = get_category_thresholds(args, cls_name)
            bbox_area = max(1, (x2 - x1) * (y2 - y1))
            if bbox_area < min_bbox_area:
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            if roi_mask[cy, cx] == 0:
                continue

            mask = make_pseudo_mask(depth_map, (x1, y1, x2, y2), args)
            if mask is None:
                continue

            mask_pixels = int(np.count_nonzero(mask))
            if mask_pixels < min_mask_pixels:
                continue

            metrics = calc_damage_metrics(depth_map, (x1, y1, x2, y2), work_frame.shape, args, mask)
            if metrics["area_m2"] <= 0 or metrics["volume_m3"] < 0:
                continue

            status, box_color = risk_to_status(metrics["urgency_score"])
            damage_record = {
                "damage_id": f"{frame_id}_{len(frame_data_list) + 1}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_frame": frame_id,
                "category": category_name,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "mask_pixels": mask_pixels,
                "metrics": metrics,
                "status": status,
                "box_color": box_color,
                "mask": mask,
            }
            frame_data_list.append(damage_record)

        frame_data_list = contained_dedup(
            frame_data_list,
            overlap_threshold=args.contained_overlap_threshold,
            area_ratio_threshold=args.contained_area_ratio_threshold,
            conf_tolerance=args.contained_confidence_tolerance,
        )

        for idx, rec in enumerate(frame_data_list, start=1):
            x1, y1, x2, y2 = rec["bbox"]
            box_color = rec["box_color"]
            mask = rec["mask"]

            overlay = annotated_frame.copy()
            overlay[mask > 0] = (overlay[mask > 0] * 0.55 + np.array(box_color) * 0.45).astype(np.uint8)
            annotated_frame = overlay
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated_frame, contours, -1, box_color, 2)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            label = f"#{idx} {rec['category']} {rec['status']} {rec['confidence']:.2f}"
            annotated_frame = draw_text(
                annotated_frame,
                label,
                (x1, max(10, y1 - 30)),
                color=(255, 255, 255),
                font_size=args.font_size,
                bg_color=box_color,
                font_path=args.font_path,
            )

    inference_time = time.time() - start_time
    annotated_frame = draw_dashboard_panel(
        annotated_frame,
        frame_data_list,
        inference_time,
        depth_colormap,
        args,
    )

    for rec in frame_data_list:
        rec.pop("box_color", None)
        rec.pop("mask", None)

    return annotated_frame, inference_time, frame_data_list


def draw_dashboard_panel(frame: np.ndarray, records: List[Dict[str, Any]], inference_time: float,
                         depth_colormap: Optional[np.ndarray], args) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_width = min(max(args.panel_width, 320), max(320, w // 2))
    canvas = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    canvas[:, w:] = (22, 26, 33)

    total = len(records)
    total_area = sum(r["metrics"]["area_m2"] for r in records)
    total_volume = sum(r["metrics"]["volume_m3"] for r in records)
    max_depth = max([r["metrics"]["max_depth_drop"] for r in records], default=0.0)
    counter = Counter([r["status"] for r in records])

    x0 = w + 18
    y = 18
    line_gap = max(28, args.font_size + 8)

    canvas = draw_text(canvas, "路面损伤多维量化监测面板", (x0, y), (255, 255, 255), args.font_size + 4, font_path=args.font_path)
    y += line_gap + 10

    stats = [
        f"当前帧识别病害数：{total}",
        f"推理耗时：{inference_time * 1000:.1f} ms",
        f"当前帧受损面积：{total_area:.3f} m²",
        f"当前帧填补体积：{total_volume:.4f} m³",
        f"当前帧代表深度：{max_depth:.2f} cm",
        f"风险等级分布：重{counter.get('重度', 0)} / 中{counter.get('中度', 0)} / 轻{counter.get('轻度', 0)}",
    ]
    for text in stats:
        canvas = draw_text(canvas, text, (x0, y), (220, 235, 255), args.font_size, font_path=args.font_path)
        y += line_gap

    if depth_colormap is not None and args.show_depth_preview:
        preview_w = panel_width - 36
        preview_h = max(140, int(preview_w * 9 / 16))
        preview = cv2.resize(depth_colormap, (preview_w, preview_h))
        canvas[y:y + preview_h, x0:x0 + preview_w] = preview
        canvas = draw_text(canvas, "深度热力图", (x0, y - 28), (255, 200, 120), args.font_size, font_path=args.font_path)
        y += preview_h + 18

    canvas = draw_text(canvas, "目标清单", (x0, y), (255, 200, 120), args.font_size, font_path=args.font_path)
    y += line_gap

    max_rows = max(1, (h - y - 10) // line_gap)
    for idx, rec in enumerate(records[:max_rows], start=1):
        line1 = f"#{idx} {rec['category']} {rec['status']}  置信度{rec['confidence']:.2f}"
        line2 = f"面积{rec['metrics']['area_m2']:.3f}m²  深度{rec['metrics']['max_depth_drop']:.2f}cm  体积{rec['metrics']['volume_m3']:.4f}m³"
        canvas = draw_text(canvas, line1, (x0, y), (240, 240, 240), args.font_size - 1, font_path=args.font_path)
        y += line_gap - 2
        canvas = draw_text(canvas, line2, (x0, y), (185, 205, 225), args.font_size - 3, font_path=args.font_path)
        y += line_gap + 2
    return canvas


def infer_image_task(models, source_path: str, args, task_run_id: str) -> Dict[str, Any]:
    yolo_model, depth_estimator = models
    frame = cv2.imread(source_path)
    if frame is None:
        raise FileNotFoundError(f"无法读取图片: {source_path}")

    start_time = time.time()
    filename = os.path.basename(source_path)
    processed = preprocess_frame(frame, args)
    detections = run_yolo_predict_image(yolo_model, processed, args)
    depth_map = compute_depth_map_image(processed, depth_estimator) if detections else np.zeros(frame.shape[:2], dtype=np.float32)

    records: List[Dict[str, Any]] = []
    masks: List[np.ndarray] = []
    image_min_bbox = min(args.min_bbox_area_px, 650)
    image_min_mask = min(args.min_mask_pixels, 160)

    for det in detections:
        bbox = [int(round(v)) for v in det['bbox']]
        x1, y1, x2, y2 = bbox
        cls_name = det.get('cls_name', '')
        category_name = normalize_category_name(cls_name)
        det_min_bbox, det_min_mask = get_category_thresholds(args, cls_name)
        if det['confidence'] < args.valid_conf_threshold:
            continue
        if max(1, (x2 - x1) * (y2 - y1)) < min(image_min_bbox, det_min_bbox):
            continue
        local_mask = build_pseudo_mask_image(processed, bbox, args)
        mask_pixels = int(local_mask.sum())
        if mask_pixels < min(image_min_mask, det_min_mask):
            continue
        depth_metrics = compute_depth_metrics_image(depth_map, bbox, local_mask)
        quant = estimate_area_and_volume_image(mask_pixels, bbox, frame.shape, depth_metrics, args)
        score, severity = score_damage_image(quant['area_m2'], quant['mean_depth_m'], quant['volume_m3'])
        records.append({
            'damage_id': f'{filename}_{len(records) + 1}',
            '序号': len(records) + 1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source_frame': filename,
            'category': category_name,
            'confidence': round(det['confidence'], 4),
            'bbox': bbox,
            'metrics': {
                'area_m2': round(quant['area_m2'], 5),
                'mean_depth_m': round(quant['mean_depth_m'], 5),
                'max_depth_m': round(quant['max_depth_m'], 5),
                'mean_depth_cm': round(quant['mean_depth_cm'], 2),
                'max_depth_cm': round(quant['max_depth_cm'], 2),
                'volume_m3': round(quant['volume_m3'], 6),
                'severity_score': round(score, 2),
                'reference_depth': round(depth_metrics['reference_depth'], 3),
                'mean_depth_delta': round(depth_metrics['mean_depth_delta'], 3),
                'max_depth_delta': round(depth_metrics['max_depth_delta'], 3),
            },
            'status': severity,
        })
        masks.append(local_mask)

    records = contained_dedup(records, overlap_threshold=args.contained_overlap_threshold, area_ratio_threshold=args.contained_area_ratio_threshold, conf_tolerance=args.contained_confidence_tolerance)
    bbox_to_mask = {tuple(rec['bbox']): m for rec, m in zip(records if len(records) == len(masks) else records, masks)}
    remasks = []
    for i, rec in enumerate(records, start=1):
        rec['序号'] = i
        remasks.append(bbox_to_mask.get(tuple(rec['bbox']), build_pseudo_mask_image(processed, rec['bbox'], args)))
    masks = remasks

    scene = annotate_scene_image(processed, records, masks, args)
    annotated = compose_dashboard_view_image(scene, depth_map, records, filename, (time.time() - start_time) * 1000.0, args)

    depth_colormap_img = None
    if depth_map is not None and depth_map.size > 0:
        max_d = max(float(depth_map.max()), 1e-6)
        depth_colormap_img = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_map, alpha=255.0 / max_d),
            cv2.COLORMAP_INFERNO,
        )

    export_records = []
    for rec in records:
        export_records.append({
            'damage_id': rec['damage_id'],
            'task_run_id': task_run_id,
            '序号': rec['序号'],
            'timestamp': rec['timestamp'],
            'source_frame': rec['source_frame'],
            'category': rec['category'],
            'confidence': rec['confidence'],
            'bbox': rec['bbox'],
            'metrics': rec['metrics'],
            'status': rec['status'],
        })

    summary = {
        'task_run_id': task_run_id,
        'source': source_path,
        '类型': '单图',
        '识别目标数': len(export_records),
        '总受损面积_m2': round(sum(r['metrics']['area_m2'] for r in export_records), 4),
        '总填补体积_m3': round(sum(r['metrics']['volume_m3'] for r in export_records), 5),
        '最大相对深度_cm': round(max([r['metrics']['max_depth_cm'] for r in export_records], default=0.0), 2),
    }
    return {
        'filename': filename,
        'annotated': annotated,
        'depth_preview': depth_colormap_img,
        'records': export_records,
        'summary': summary,
    }


def process_image(models, source_path: str, args) -> None:
    task_run_id = make_task_run_id()
    task_dir = get_task_archive_dir(args.output_dir, task_run_id)
    result = infer_image_task(models, source_path, args, task_run_id)

    save_path = os.path.join(args.output_dir, f"pred_{result['filename']}")
    cv2.imwrite(save_path, result['annotated'])
    print(f'单图推理完成，结果已保存至: {save_path}')
    copy_artifact_if_exists(save_path, task_dir)

    if result['depth_preview'] is not None:
        depth_save_path = os.path.join(args.output_dir, f"depth_{result['filename']}")
        cv2.imwrite(depth_save_path, result['depth_preview'])
        print(f'深度热力图已保存至: {depth_save_path}')
        copy_artifact_if_exists(depth_save_path, task_dir)

    save_json(os.path.join(args.output_dir, 'dashboard_data.json'), result['records'])
    save_json(os.path.join(args.output_dir, 'summary.json'), result['summary'])
    save_json(os.path.join(args.output_dir, f'dashboard_data_{task_run_id}.json'), result['records'])
    save_json(os.path.join(args.output_dir, f'summary_{task_run_id}.json'), result['summary'])
    save_json(os.path.join(task_dir, 'dashboard_data.json'), result['records'])
    save_json(os.path.join(task_dir, 'summary.json'), result['summary'])
    print(f"[+] 大屏展示数据已保存至: {os.path.join(args.output_dir, 'dashboard_data.json')}")
    print(f"[+] 汇总信息已保存至: {os.path.join(args.output_dir, 'summary.json')}")

    if args.show:
        cv2.imshow('路面损伤多维量化评估', result['annotated'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_folder(models, source_dir: str, args) -> None:
    task_run_id = make_task_run_id()
    task_dir = get_task_archive_dir(args.output_dir, task_run_id)
    image_files = collect_image_files(source_dir)
    if not image_files:
        raise FileNotFoundError(f"文件夹中未找到可处理图片: {source_dir}")

    all_records: List[Dict[str, Any]] = []
    processed_images = 0
    last_preview_path = ""
    last_depth_path = ""
    max_depth_cm = 0.0
    severity_counter = {"轻度": 0, "中度": 0, "重度": 0}

    print(f"[*] 开始处理图片文件夹: {source_dir}")
    print(f"[*] 共发现 {len(image_files)} 张图片")

    for idx, image_path in enumerate(image_files, start=1):
        result = infer_image_task(models, image_path, args, task_run_id)
        pred_path = os.path.join(task_dir, f"pred_{result['filename']}")
        cv2.imwrite(pred_path, result['annotated'])
        last_preview_path = pred_path

        if result['depth_preview'] is not None:
            depth_path = os.path.join(task_dir, f"depth_{result['filename']}")
            cv2.imwrite(depth_path, result['depth_preview'])
            last_depth_path = depth_path

        for record in result['records']:
            all_records.append(record)
            severity = str(record.get('status', ''))
            if severity in severity_counter:
                severity_counter[severity] += 1
            metrics = record.get('metrics', {})
            if isinstance(metrics, dict):
                max_depth_cm = max(max_depth_cm, float(metrics.get('max_depth_cm', 0.0) or 0.0))

        processed_images += 1
        print(f"[进度] 已处理 {idx} / {len(image_files)} 张图片")

    summary = {
        'task_run_id': task_run_id,
        'source': source_dir,
        '类型': '文件夹',
        '图片数': len(image_files),
        '成功处理图片数': processed_images,
        '识别目标数': len(all_records),
        '风险等级分布': severity_counter,
        '总受损面积_m2': round(sum(r['metrics']['area_m2'] for r in all_records), 4),
        '总填补体积_m3': round(sum(r['metrics']['volume_m3'] for r in all_records), 5),
        '最大相对深度_cm': round(max_depth_cm, 2),
    }

    save_json(os.path.join(args.output_dir, 'dashboard_data.json'), all_records)
    save_json(os.path.join(args.output_dir, 'summary.json'), summary)
    save_json(os.path.join(args.output_dir, f'dashboard_data_{task_run_id}.json'), all_records)
    save_json(os.path.join(args.output_dir, f'summary_{task_run_id}.json'), summary)
    save_json(os.path.join(task_dir, 'dashboard_data.json'), all_records)
    save_json(os.path.join(task_dir, 'summary.json'), summary)

    if last_preview_path:
        batch_preview = os.path.join(args.output_dir, 'pred_batch_preview.png')
        copy2(last_preview_path, batch_preview)
    if last_depth_path:
        batch_depth_preview = os.path.join(args.output_dir, 'depth_batch_preview.png')
        copy2(last_depth_path, batch_depth_preview)

    print(f"[+] 文件夹任务完成，共处理 {processed_images} 张图片")
    print(f"[+] 汇总信息已保存至: {os.path.join(args.output_dir, 'summary.json')}")
    print(f"[+] 任务归档目录: {task_dir}")

def process_video(models, source, args) -> None:
    yolo_model, depth_estimator = models
    task_run_id = make_task_run_id()
    task_dir = get_task_archive_dir(args.output_dir, task_run_id)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频流: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    panel_width = min(max(args.panel_width, 320), max(320, width // 2))
    out_w = width + panel_width
    out_h = height

    source_name = str(source).replace(':', '_').replace('/', '_').replace('\\', '_')
    output_video_path = os.path.join(args.output_dir, f"pred_{Path(source_name).stem or 'stream'}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*args.video_codec)
    out = cv2.VideoWriter(output_video_path, fourcc, max(1.0, fps / max(1, args.frame_stride)), (out_w, out_h))

    tracker = SimpleDamageTracker(
        iou_threshold=args.track_iou_threshold,
        max_missed=args.track_max_missed,
        min_confirm_hits=args.track_min_confirm_hits,
        center_dist_ratio=args.track_center_dist_ratio,
    )
    all_frame_records: List[Dict[str, Any]] = []

    frame_index = 0
    processed_count = 0
    t0 = time.time()
    print(f"[*] 开始处理视频流: {source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if (frame_index - 1) % args.frame_stride != 0:
            continue
        if args.max_frames > 0 and processed_count >= args.max_frames:
            break

        frame_id = f"frame_{frame_index:06d}"
        annotated_frame, inference_time, frame_data = process_frame(frame, yolo_model, depth_estimator, args, frame_id=frame_id)
        frame_data = tracker.update(frame_data, frame_index, frame.shape[:2])

        for rec in frame_data:
            if not rec.get("confirmed_track", False):
                continue
            x1, y1, x2, y2 = rec["bbox"]
            text = f"{get_record_category(rec)}{rec['track_id']}"
            annotated_frame = draw_text(
                annotated_frame,
                text,
                (x1 + 4, min(y2 + 6, height - 30)),
                color=(255, 255, 255),
                font_size=max(18, args.font_size - 2),
                bg_color=(70, 70, 70),
                font_path=args.font_path,
            )

        visible_frame_data = [d for d in frame_data if d.get("confirmed_track", False)]
        frame_payload = {
            "task_run_id": task_run_id,
            "frame_index": frame_index,
            "frame_id": frame_id,
            "timestamp": round(frame_index / fps, 3),
            "inference_time_ms": round(inference_time * 1000.0, 2),
            "detections": visible_frame_data,
        }
        all_frame_records.append(frame_payload)
        out.write(annotated_frame)
        if processed_count == 0:
            _fh, _fw = annotated_frame.shape[:2]
            _panel_w = _fw - width
            if _panel_w > 0:
                depth_preview_frame = annotated_frame[:, width:]
                depth_preview_path = os.path.join(args.output_dir, 'depth_frame_preview.png')
                cv2.imwrite(depth_preview_path, depth_preview_frame)
        if processed_count % 5 == 0:
            preview_path = os.path.join(args.output_dir, 'preview_current.jpg')
            cv2.imwrite(preview_path, annotated_frame)
        processed_count += 1

        if args.show:
            cv2.imshow("路面损伤视频巡检", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

        if processed_count % 10 == 0:
            msg = f"[进度] 已处理 {processed_count} 帧"
            if total_frames > 0:
                msg += f" / 约 {int(np.ceil(total_frames / max(1, args.frame_stride)))} 帧"
            print(msg)

    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()

    unique_tracks = tracker.finalize()
    elapsed = time.time() - t0

    dashboard_path = os.path.join(args.output_dir, "dashboard_data_video.json")
    summary_path = os.path.join(args.output_dir, "summary_video.json")
    save_json(dashboard_path, all_frame_records)

    status_counter = Counter([t["最高风险等级"] for t in unique_tracks])
    summary = {
        "task_run_id": task_run_id,
        "source": str(source),
        "类型": "视频流",
        "原始帧总数": total_frames if total_frames > 0 else None,
        "实际处理帧数": processed_count,
        "frame_stride": args.frame_stride,
        "总耗时_s": round(elapsed, 2),
        "平均帧耗时_ms": round((elapsed / max(1, processed_count)) * 1000.0, 2),
        "唯一病害数": len(unique_tracks),
        "唯一坑洼数": len(unique_tracks),
        "风险等级分布": {
            "重度": status_counter.get("重度", 0),
            "中度": status_counter.get("中度", 0),
            "轻度": status_counter.get("轻度", 0),
        },
        "最大单坑受损面积_m2": round(max([t["最大受损面积_m2"] for t in unique_tracks], default=0.0), 4),
        "最大单坑填补体积_m3": round(max([t["最大填补体积_m3"] for t in unique_tracks], default=0.0), 5),
        "轨迹汇总": unique_tracks,
        "输出视频": output_video_path,
    }
    save_json(summary_path, summary)
    save_json(os.path.join(args.output_dir, f"dashboard_data_video_{task_run_id}.json"), all_frame_records)
    save_json(os.path.join(args.output_dir, f"summary_video_{task_run_id}.json"), summary)
    save_json(os.path.join(task_dir, "dashboard_data_video.json"), all_frame_records)
    save_json(os.path.join(task_dir, "summary_video.json"), summary)
    copy_artifact_if_exists(output_video_path, task_dir)
    copy_artifact_if_exists(os.path.join(args.output_dir, 'depth_frame_preview.png'), task_dir)
    copy_artifact_if_exists(os.path.join(args.output_dir, 'preview_current.jpg'), task_dir)

    print(f"[+] 视频处理完成，结果视频已保存至: {output_video_path}")
    print(f"[+] 逐帧数据已保存至: {dashboard_path}")
    print(f"[+] 视频汇总信息已保存至: {summary_path}")


def main(args):
    ensure_dir(args.output_dir)
    models = init_models(args)
    source = str(args.source)

    if os.path.isfile(source) and is_image_file(source):
        process_image(models, source, args)
        return

    if os.path.isdir(source):
        process_folder(models, source, args)
        return

    if source.isdigit():
        process_video(models, int(source), args)
        return

    if source.startswith(("rtsp://", "http://", "https://")):
        process_video(models, source, args)
        return

    if os.path.isfile(source) and is_video_file(source):
        process_video(models, source, args)
        return

    raise ValueError("无法识别输入源，请传入图片、文件夹、视频、摄像头编号或 RTSP/HTTP 流")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多任务感知路面损伤多维量化评估系统 - 视频流处理增强版")
    parser.add_argument('--model', type=str, required=True, help='YOLO 模型权重路径')
    parser.add_argument('--source', type=str, default='0', help='输入源：图片 / 视频 / 摄像头编号 / RTSP')
    parser.add_argument('--depth-model-path', type=str, default='', help='Depth Anything 本地模型目录')
    parser.add_argument('--conf-threshold', type=float, default=0.20, help='YOLO 置信度阈值')
    parser.add_argument('--valid-conf-threshold', type=float, default=0.35, help='进入量化与统计的有效置信度阈值')
    parser.add_argument('--iou-threshold', type=float, default=0.45, help='NMS IoU阈值')
    parser.add_argument('--imgsz', type=int, default=960, help='推理图像尺寸')
    parser.add_argument('--device', type=str, default='', help='推理设备')
    parser.add_argument('--output-dir', type=str, default='runs/predict', help='结果保存目录')
    parser.add_argument('--show', action='store_true', help='实时显示视频窗口')
    parser.add_argument('--frame-stride', type=int, default=1, help='视频跳帧间隔，2 表示每 2 帧处理 1 帧')
    parser.add_argument('--max-frames', type=int, default=0, help='最多处理多少帧，0 表示不限制')
    parser.add_argument('--track-iou-threshold', type=float, default=0.18, help='视频级去重跟踪 IoU 阈值')
    parser.add_argument('--track-max-missed', type=int, default=15, help='轨迹允许丢失的最大帧数')
    parser.add_argument('--track-min-confirm-hits', type=int, default=2, help='轨迹至少出现几次才计入唯一坑洼')
    parser.add_argument('--track-center-dist-ratio', type=float, default=0.10, help='中心点匹配距离比例阈值')
    parser.add_argument('--panel-width', type=int, default=420, help='右侧信息面板宽度')
    parser.add_argument('--font-size', type=int, default=22, help='中文文字大小')
    parser.add_argument('--font-path', type=str, default='', help='中文字体路径，例如 C:\\Windows\\Fonts\\msyh.ttc')
    parser.add_argument('--video-codec', type=str, default='mp4v', help='视频编码，默认 mp4v')
    parser.add_argument('--augment', action='store_true', help='启用测试时增强，提高召回率')
    parser.add_argument('--enable-tiling', action='store_true', help='启用切片推理，提高单图多目标召回')
    parser.add_argument('--tile-size', type=int, default=640, help='切片尺寸')
    parser.add_argument('--tile-overlap', type=float, default=0.25, help='切片重叠比例')
    parser.add_argument('--merge-iou-threshold', type=float, default=0.45, help='切片融合 NMS 阈值')
    parser.add_argument('--max-det', type=int, default=300, help='最大候选数')
    parser.add_argument('--mask-dark-offset', type=int, default=10, help='图片伪分割暗区阈值偏移')
    parser.add_argument('--mask-edge-threshold', type=int, default=18, help='图片伪分割边缘阈值')
    parser.add_argument('--base-meters-per-pixel', type=float, default=0.0105, help='图片模式基础像素尺度')
    parser.add_argument('--calibrated-scene-width-cm', type=float, default=167.0, help='标定场景下图片宽边对应的实际宽度（cm）')
    parser.add_argument('--calibrated-image-width-px', type=int, default=4096, help='标定场景下图片宽边像素数')
    parser.add_argument('--depth-unit-scale', type=float, default=0.00012, help='相对深度到米的缩放系数')
    parser.add_argument('--enable-denoise', action='store_true', help='启用简单去噪')
    parser.add_argument('--enable-clahe', action='store_true', help='启用 CLAHE 增强')
    parser.add_argument('--clahe-clip-limit', type=float, default=2.0, help='CLAHE clip limit')
    parser.add_argument('--show-depth-preview', action='store_true', help='在面板中展示深度热力图')
    parser.add_argument('--show-road-roi', action='store_true', help='在画面中绘制路面有效区域')
    parser.add_argument('--road-roi-top-ratio', type=float, default=0.45, help='路面 ROI 顶部 y 比例')
    parser.add_argument('--road-roi-bottom-ratio', type=float, default=0.98, help='路面 ROI 底部 y 比例')
    parser.add_argument('--road-roi-top-width-ratio', type=float, default=0.28, help='路面 ROI 顶部宽度比例')
    parser.add_argument('--road-roi-bottom-width-ratio', type=float, default=0.96, help='路面 ROI 底部宽度比例')
    parser.add_argument('--road-roi-center-x-ratio', type=float, default=0.52, help='路面 ROI 中心 x 比例')
    parser.add_argument('--min-bbox-area-px', type=int, default=900, help='最小检测框像素面积')
    parser.add_argument('--min-mask-pixels', type=int, default=240, help='最小掩码像素数')
    parser.add_argument('--mask-morph-kernel', type=int, default=5, help='伪分割形态学核大小，建议奇数')
    parser.add_argument('--min-mask-ratio-for-area', type=float, default=0.08, help='掩码占框最小比例，用于面积估算')
    parser.add_argument('--reference-ring-ratio', type=float, default=0.25, help='局部参考环带扩张比例')
    parser.add_argument('--pixel-area-cm2-near', type=float, default=3.0, help='图像近处单像素近似地面面积（cm²）')
    parser.add_argument('--pixel-area-cm2-far', type=float, default=0.08, help='图像远处单像素近似地面面积（cm²）')
    parser.add_argument('--perspective-gamma', type=float, default=2.0, help='透视尺度变化曲线')
    parser.add_argument('--depth-scale-cm', type=float, default=0.20, help='深度差到厘米的映射系数')
    parser.add_argument('--max-depth-cm', type=float, default=18.0, help='最大有效深度，超过则钳制')
    parser.add_argument('--max-area-m2', type=float, default=2.5, help='单坑最大有效面积，超过则钳制')
    parser.add_argument('--max-volume-m3', type=float, default=0.20, help='单坑最大有效体积，超过则钳制')
    parser.add_argument('--score-area-ref-m2', type=float, default=0.60, help='面积评分参考值')
    parser.add_argument('--score-depth-ref-cm', type=float, default=6.0, help='深度评分参考值')
    parser.add_argument('--score-volume-ref-m3', type=float, default=0.03, help='体积评分参考值')
    parser.add_argument('--score-weight-area', type=float, default=0.35, help='面积评分权重')
    parser.add_argument('--score-weight-depth', type=float, default=0.35, help='深度评分权重')
    parser.add_argument('--score-weight-volume', type=float, default=0.30, help='体积评分权重')
    parser.add_argument('--contained-overlap-threshold', type=float, default=0.88, help='包含去重覆盖阈值')
    parser.add_argument('--contained-area-ratio-threshold', type=float, default=0.55, help='包含去重面积比阈值')
    parser.add_argument('--contained-confidence-tolerance', type=float, default=0.15, help='包含去重置信度容差')

    main(parser.parse_args())