import os
import sys
from pathlib import Path
from typing import List, Optional

from app.core.schemas import InferenceConfig


def _predict_script_path() -> str:
    """Locate predict.py in dev and PyInstaller environments."""
    if getattr(sys, 'frozen', False):
        return str(Path(sys._MEIPASS) / 'predict.py')
    return str(Path(__file__).parent.parent.parent / 'predict.py')


def _ensure_exists(path: str, label: str, allow_empty: bool = False) -> None:
    if not path:
        if allow_empty:
            return
        raise FileNotFoundError(f'{label}未配置')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{label}不存在: {path}')


def _is_live_source(source: str) -> bool:
    src = (source or '').strip()
    return src.isdigit() or src.startswith(('rtsp://', 'http://', 'https://'))


class CommandBuilder:
    @staticmethod
    def build(config: InferenceConfig, python_executable: Optional[str] = None) -> List[str]:
        if python_executable and os.path.exists(python_executable):
            exe = python_executable
        elif python_executable and not os.path.exists(python_executable):
            exe = sys.executable
        else:
            exe = sys.executable

        predict_script = _predict_script_path()
        _ensure_exists(exe, 'Python解释器')
        _ensure_exists(predict_script, '推理脚本 predict.py')
        _ensure_exists(config.model_path, '模型权重文件')
        if not _is_live_source(config.source):
            _ensure_exists(config.source, '输入源')
        _ensure_exists(config.depth_model_path, '深度模型目录', allow_empty=True)
        _ensure_exists(config.font_path, '字体文件', allow_empty=True)

        cmd = [
            exe, predict_script,
            '--model', config.model_path,
            '--source', config.source,
            '--output-dir', config.output_dir,
            '--depth-model-path', config.depth_model_path,
            '--font-path', config.font_path,
            '--conf-threshold', str(config.conf_threshold),
            '--valid-conf-threshold', str(config.valid_conf_threshold),
            '--iou-threshold', str(config.iou_threshold),
            '--imgsz', str(config.imgsz),
            '--frame-stride', str(config.frame_stride),
            '--calibrated-scene-width-cm', str(config.calibrated_scene_width_cm),
            '--calibrated-image-width-px', str(config.calibrated_image_width_px),
        ]
        if config.max_frames > 0:
            cmd.extend(['--max-frames', str(config.max_frames)])
        if config.show_depth_preview:
            cmd.append('--show-depth-preview')
        if config.show_road_roi:
            cmd.append('--show-road-roi')
        if config.enable_clahe:
            cmd.append('--enable-clahe')
        if config.enable_denoise:
            cmd.append('--enable-denoise')
        if config.augment:
            cmd.append('--augment')
        if config.enable_tiling:
            cmd.append('--enable-tiling')
        if config.device:
            cmd.extend(['--device', config.device])
        return cmd
