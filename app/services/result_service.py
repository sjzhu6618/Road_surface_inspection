import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.core.schemas import TaskSummary


class ResultService:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.source_filename: Optional[str] = None

    def set_source_filename(self, filename: str):
        self.source_filename = Path(filename).stem if filename else None

    def _read_json(self, path: Optional[Path]):
        if path and path.exists():
            try:
                return json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                return {}
        return {}

    @staticmethod
    def _pick_first(data: Dict[str, Any], keys: Iterable[str], default: Any = 0):
        for key in keys:
            if key in data and data[key] not in (None, ''):
                return data[key]
        return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _flatten_dashboard_records(self, dashboard_data: Any) -> List[Dict[str, Any]]:
        if not isinstance(dashboard_data, list):
            return []
        if dashboard_data and isinstance(dashboard_data[0], dict) and 'metrics' in dashboard_data[0]:
            return [d for d in dashboard_data if isinstance(d, dict)]
        if dashboard_data and isinstance(dashboard_data[0], dict) and 'detections' in dashboard_data[0]:
            records: List[Dict[str, Any]] = []
            for frame in dashboard_data:
                detections = frame.get('detections', []) if isinstance(frame, dict) else []
                for det in detections:
                    if isinstance(det, dict):
                        records.append(det)
            return records
        return []

    def _find_latest(self, patterns: Iterable[str]) -> str:
        for pattern in patterns:
            matches = [p for p in self.output_dir.glob(pattern) if p.exists()]
            if matches:
                latest = max(matches, key=lambda x: x.stat().st_mtime)
                return str(latest)
        return ''

    @staticmethod
    def _latest_path(paths: List[Path]) -> str:
        if not paths:
            return ''
        return str(max(paths, key=lambda x: x.stat().st_mtime))

    def _task_dir(self, summary_data: Dict[str, Any]) -> Optional[Path]:
        task_run_id = str(summary_data.get('task_run_id', '')).strip() if isinstance(summary_data, dict) else ''
        if not task_run_id:
            return None
        task_dir = self.output_dir / 'tasks' / task_run_id
        return task_dir if task_dir.exists() else None

    def _source_name(self, summary_data: Dict[str, Any]) -> str:
        if self.source_filename:
            return Path(self.source_filename).name
        if not isinstance(summary_data, dict):
            return ''
        source = str(summary_data.get('source', '')).strip()
        return Path(source).name if source else ''

    def _find_preview(self, summary_data: Dict[str, Any]) -> str:
        search_dirs: List[Path] = []
        task_dir = self._task_dir(summary_data)
        if task_dir is not None:
            search_dirs.append(task_dir)
        search_dirs.append(self.output_dir)

        source_name = self._source_name(summary_data)
        exact_names: List[str] = []
        if source_name:
            exact_names.extend([f'pred_{source_name}', f'annotated_{source_name}'])
            source_stem = Path(source_name).stem
            exact_names.extend([
                f'pred_{source_stem}.png',
                f'pred_{source_stem}.jpg',
                f'pred_{source_stem}.jpeg',
                f'annotated_{source_stem}.png',
                f'annotated_{source_stem}.jpg',
                f'annotated_{source_stem}.jpeg',
            ])

        for base_dir in search_dirs:
            exact_matches = [base_dir / name for name in exact_names if (base_dir / name).is_file()]
            if exact_matches:
                return self._latest_path(exact_matches)

            pattern_matches: List[Path] = []
            for pattern in (
                'pred_batch_preview.png',
                'annotated_*',
                'pred_*',
                'preview_current.jpg',
                'pred_*.mp4',
                'pred_*.avi',
                'annotated_*.mp4',
            ):
                pattern_matches.extend([p for p in base_dir.glob(pattern) if p.is_file()])
            if pattern_matches:
                return self._latest_path(pattern_matches)
        return ''

    def _find_depth_preview(self, summary_data: Dict[str, Any]) -> str:
        search_dirs: List[Path] = []
        task_dir = self._task_dir(summary_data)
        if task_dir is not None:
            search_dirs.append(task_dir)
        search_dirs.append(self.output_dir)

        source_name = self._source_name(summary_data)
        exact_names: List[str] = []
        if source_name:
            source_stem = Path(source_name).stem
            exact_names.extend([
                f'depth_{source_name}',
                f'depth_{source_stem}.png',
                f'depth_{source_stem}.jpg',
                f'depth_{source_stem}.jpeg',
            ])

        for base_dir in search_dirs:
            exact_matches = [base_dir / name for name in exact_names if (base_dir / name).is_file()]
            if exact_matches:
                return self._latest_path(exact_matches)

            pattern_matches: List[Path] = []
            for pattern in ('depth_batch_preview.png', 'depth_*', 'depth_frame_preview.png'):
                pattern_matches.extend([p for p in base_dir.glob(pattern) if p.is_file()])
            if pattern_matches:
                return self._latest_path(pattern_matches)
        return ''

    def _find_result_video(self) -> str:
        if self.source_filename:
            video = self._find_latest(
                (
                    f'pred_{self.source_filename}.mp4',
                    f'pred_{self.source_filename}.avi',
                    f'annotated_{self.source_filename}.mp4',
                )
            )
            if video:
                return video
        return self._find_latest(('pred_*.mp4', 'pred_*.avi', 'annotated_*.mp4'))

    def _resolve_task_dir(self, summary_data: Dict[str, Any]) -> Optional[Path]:
        task_run_id = summary_data.get('task_run_id')
        if not task_run_id:
            return None
        task_dir = self.output_dir / 'tasks' / str(task_run_id)
        return task_dir if task_dir.exists() else None

    def _collect_image_records(self, task_dir: Optional[Path]) -> List[Dict[str, str]]:
        if not task_dir or not task_dir.exists():
            return []

        image_records: List[Dict[str, str]] = []
        for path in sorted(task_dir.glob('pred_*')):
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}:
                image_records.append({'image_type': 'pred', 'image_path': str(path)})
        for path in sorted(task_dir.glob('depth_*')):
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}:
                image_records.append({'image_type': 'depth', 'image_path': str(path)})
        return image_records

    def build_summary(self) -> TaskSummary:
        summary_candidates = [p for p in self.output_dir.glob('summary*.json') if p.exists()]
        dashboard_candidates = [p for p in self.output_dir.glob('dashboard_data*.json') if p.exists()]
        summary_path = max(summary_candidates, key=lambda x: x.stat().st_mtime) if summary_candidates else (self.output_dir / 'summary.json')
        dashboard_path = max(dashboard_candidates, key=lambda x: x.stat().st_mtime) if dashboard_candidates else (self.output_dir / 'dashboard_data.json')

        summary_data = self._read_json(summary_path) if summary_path else {}
        dashboard_data = self._read_json(dashboard_path) if dashboard_path else {}
        records = self._flatten_dashboard_records(dashboard_data)

        severity_dist = self._pick_first(summary_data, ['severity_distribution', '风险等级分布'], {})
        if not isinstance(severity_dist, dict):
            severity_dist = {}

        unique_damage_count = self._safe_int(
            self._pick_first(summary_data, ['unique_damage_count', 'damage_count', '唯一病害数', '唯一坑洼数', '唯一坑洞数', '识别目标数'], len(records))
        )
        severe_count = self._safe_int(severity_dist.get('重度', 0))
        if severe_count == 0 and records:
            severe_count = sum(1 for r in records if str(r.get('status', '')) == '重度')

        total_area = self._safe_float(
            self._pick_first(summary_data, ['estimated_total_area_m2', 'total_area_m2', '总受损面积_m2', '总受损面积m2'], 0.0)
        )
        if total_area == 0.0 and records:
            total_area = sum(self._safe_float((r.get('metrics') or {}).get('area_m2', 0.0)) for r in records)

        total_volume = self._safe_float(
            self._pick_first(summary_data, ['estimated_total_volume_m3', 'total_volume_m3', '总填补体积_m3', '总填补体积m3'], 0.0)
        )
        if total_volume == 0.0 and records:
            total_volume = sum(self._safe_float((r.get('metrics') or {}).get('volume_m3', 0.0)) for r in records)

        avg_depth = self._safe_float(
            self._pick_first(summary_data, ['avg_depth_cm', 'average_depth_cm', '平均深度_cm', '最大相对深度cm'], 0.0)
        )
        if avg_depth == 0.0:
            track_summary = summary_data.get('轨迹汇总', []) if isinstance(summary_data, dict) else []
            if isinstance(track_summary, list) and track_summary:
                vals = [self._safe_float(t.get('最大深度落差_cm', 0.0)) for t in track_summary if isinstance(t, dict)]
                vals = [v for v in vals if v > 0]
                if vals:
                    avg_depth = sum(vals) / len(vals)
        if avg_depth == 0.0 and records:
            vals = [
                self._safe_float((r.get('metrics') or {}).get('max_depth_cm', (r.get('metrics') or {}).get('mean_depth_drop', 0.0)))
                for r in records
            ]
            vals = [v for v in vals if v > 0]
            if vals:
                avg_depth = sum(vals) / len(vals)

        raw_summary: Dict[str, Any] = dict(summary_data) if isinstance(summary_data, dict) else {}
        if dashboard_path and dashboard_path.exists():
            raw_summary.setdefault('dashboard_path', str(dashboard_path))

        source_path = self._pick_first(summary_data, ['source', '来源'], '')
        raw_summary['source'] = source_path
        if source_path:
            task_name = Path(source_path).stem
        elif summary_path:
            task_name = summary_path.stem.replace('summary_', '').replace('summary_video_', '')
        else:
            task_name = '未命名任务'
        raw_summary['task_name'] = task_name

        raw_summary.setdefault('unique_damage_count', unique_damage_count)
        raw_summary.setdefault(
            'severity_distribution',
            severity_dist if severity_dist else {
                '重度': severe_count,
                '中度': sum(1 for r in records if str(r.get('status', '')) == '中度'),
                '轻度': sum(1 for r in records if str(r.get('status', '')) == '轻度'),
            },
        )
        raw_summary.setdefault('estimated_total_area_m2', round(total_area, 4))
        raw_summary.setdefault('estimated_total_volume_m3', round(total_volume, 5))
        raw_summary.setdefault('average_depth_cm', round(avg_depth, 2))

        preview_path = self._find_preview(summary_data)
        depth_path = self._find_depth_preview(summary_data)

        task_type = raw_summary.get('类型', '单图')
        video_path = self._find_result_video() if task_type == '视频流' else ''
        task_dir = self._resolve_task_dir(summary_data)
        image_records = self._collect_image_records(task_dir)

        try:
            from app.services.db_manager import init_database, save_detection_task

            init_database()

            preview_paths = {}
            if preview_path:
                preview_paths['pred'] = preview_path
            if depth_path:
                preview_paths['depth'] = depth_path

            save_detection_task(
                summary_data=raw_summary,
                dashboard_data=records,
                preview_paths=preview_paths if preview_paths else None,
                video_path=video_path if video_path else None,
                image_records=image_records if image_records else None,
            )
        except Exception as e:
            print(f"保存到数据库失败: {e}")

        return TaskSummary(
            task_name=task_name,
            summary_path=str(summary_path) if summary_path else '',
            dashboard_path=str(dashboard_path) if dashboard_path else '',
            output_preview_path=preview_path,
            depth_preview_path=depth_path,
            raw_summary=raw_summary,
            unique_damage_count=unique_damage_count,
            severe_count=severe_count,
            estimated_total_area_m2=float(total_area),
            estimated_total_volume_m3=float(total_volume),
            average_depth_cm=float(avg_depth),
        )
