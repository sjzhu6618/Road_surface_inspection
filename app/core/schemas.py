import os
from dataclasses import dataclass, field
from typing import Dict, List


def _default_font_path() -> str:
    candidates = [
        r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\simhei.ttf',
        r'C:\Windows\Fonts\simsun.ttc',
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ''


@dataclass
class InferenceConfig:
    model_path: str = ''
    source: str = ''
    output_dir: str = 'runs/predict'
    depth_model_path: str = 'depth_model'
    font_path: str = field(default_factory=_default_font_path)
    conf_threshold: float = 0.10
    valid_conf_threshold: float = 0.15
    iou_threshold: float = 0.45
    imgsz: int = 960
    frame_stride: int = 2
    max_frames: int = 0
    show_depth_preview: bool = True
    show_road_roi: bool = True
    enable_clahe: bool = True
    enable_denoise: bool = False
    augment: bool = False
    enable_tiling: bool = True
    save_result: bool = True
    source_mode: str = '视频'
    device: str = ''
    calibrated_scene_width_cm: float = 167.0
    calibrated_image_width_px: int = 4096


@dataclass
class TaskSummary:
    task_name: str = ''
    summary_path: str = ''
    dashboard_path: str = ''
    output_preview_path: str = ''
    depth_preview_path: str = ''
    raw_summary: Dict = field(default_factory=dict)
    unique_damage_count: int = 0
    severe_count: int = 0
    estimated_total_area_m2: float = 0.0
    estimated_total_volume_m3: float = 0.0
    average_depth_cm: float = 0.0
    notes: List[str] = field(default_factory=list)
