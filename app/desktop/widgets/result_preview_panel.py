from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QTabWidget, QVBoxLayout

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover
    cv2 = None
    np = None


class _ScalableImageLabel(QLabel):
    def __init__(self, empty_text: str):
        super().__init__(empty_text)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap = None

    def setScaledPixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._refresh()

    def clearScaledPixmap(self, text: str):
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            max(120, self.width() - 24),
            max(120, self.height() - 24),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)


class ResultPreviewPanel(QFrame):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        self.title = QLabel('检测结果预览')
        self.title.setObjectName('sectionTitle')
        self.subtitle = QLabel('这里只显示框出来并标注后的原图。深度辅助图与任务说明可通过下方标签查看。')
        self.subtitle.setWordWrap(True)
        self.subtitle.setObjectName('sectionSubtitle')

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.preview_label = _ScalableImageLabel('等待结果预览')
        self.depth_label = _ScalableImageLabel('深度 / 辅助视图将在此显示')
        self.note_label = QLabel('建议先在图片模式下完成参数校准，再切换视频模式进行整段评估。')
        self.note_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.note_label.setWordWrap(True)
        self.note_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        base_style = 'background-color: #050b14; border-radius: 14px; border: 1px solid #17314d; color: #8fb3d9; padding: 14px;'
        for w in [self.preview_label, self.depth_label, self.note_label]:
            w.setStyleSheet(base_style)

        self.tabs.addTab(self.preview_label, '检测视图')
        self.tabs.addTab(self.depth_label, '深度视图')
        self.tabs.addTab(self.note_label, '任务提示')

        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addWidget(self.tabs, 1)

        self.setStyleSheet(
            """
            QFrame { background-color: #0b1624; border: 1px solid #173150; border-radius: 18px; }
            QLabel#sectionTitle { color: #eaf2ff; font-size: 18px; font-weight: 800; }
            QLabel#sectionSubtitle { color: #93a8c2; font-size: 12px; }
            QTabWidget::pane { border: none; }
            QTabBar::tab { background: #0a1625; color: #b9cbe2; padding: 10px 18px; border-top-left-radius: 10px; border-top-right-radius: 10px; min-width: 92px; }
            QTabBar::tab:selected { background: #133154; color: #ffffff; }
            QTabBar::tab:hover:!selected { background: #0f243b; }
            """
        )

    def clear_preview(self) -> None:
        self.preview_label.clearScaledPixmap('等待结果预览')
        self.depth_label.clearScaledPixmap('深度 / 辅助视图将在此显示')
        self.tabs.setCurrentIndex(0)

    def set_note(self, text: str) -> None:
        self.note_label.setText(text)

    def set_summary_text(self, summary) -> None:
        lines = [f'任务：{summary.task_name or "未命名任务"}']
        if summary.summary_path:
            lines.append(f'汇总文件：{summary.summary_path}')
        if summary.dashboard_path:
            lines.append(f'大屏数据：{summary.dashboard_path}')
        if summary.raw_summary:
            lines.append(f'识别总数：{summary.unique_damage_count}，重度：{summary.severe_count}')
        self.note_label.setText('\n'.join(lines))

    def set_preview(self, file_path: str) -> None:
        path = Path(file_path)
        if not path.exists():
            self.preview_label.clearScaledPixmap(
                f'结果文件不存在：\n{file_path}\n\n请检查输出目录设置是否正确。'
            )
            return

        suffix = path.suffix.lower()
        if suffix in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}:
            pixmap = self._prepare_image_pixmap(path)
            if pixmap is None or pixmap.isNull():
                self.preview_label.clearScaledPixmap(f'无法加载预览图：\n{file_path}')
                return
            self.preview_label.setText('')
            self.preview_label.setScaledPixmap(pixmap)
            self.tabs.setCurrentIndex(0)
            return

        if suffix in {'.mp4', '.avi', '.mov', '.mkv'}:
            frame_pixmap = self._extract_video_frame(path)
            if frame_pixmap is not None:
                self.preview_label.setText('')
                self.preview_label.setScaledPixmap(frame_pixmap)
                self.note_label.setText(f'当前结果为视频：\n{file_path}\n\n当前版本显示首帧中的标注画面。')
                self.tabs.setCurrentIndex(0)
            else:
                self.preview_label.clearScaledPixmap(
                    f'当前结果为视频：\n{file_path}\n\n已保存完成，但当前环境无法提取首帧缩略图。'
                )
            return

        self.preview_label.clearScaledPixmap(f'暂不支持预览此类型文件：\n{file_path}')

    def set_depth_preview(self, file_path: str) -> None:
        if not file_path:
            return
        path = Path(file_path)
        if not path.exists():
            return
        if cv2 is not None and np is not None:
            img = cv2.imread(str(path))
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(image.copy())
                self.depth_label.setText('')
                self.depth_label.setScaledPixmap(pixmap)
                return
        pixmap = QPixmap(str(path))
        if not pixmap.isNull():
            self.depth_label.setText('')
            self.depth_label.setScaledPixmap(pixmap)

    def _prepare_image_pixmap(self, path: Path):
        if cv2 is None or np is None:
            return QPixmap(str(path))
        img = cv2.imread(str(path))
        if img is None:
            return QPixmap(str(path))
        cropped = self._crop_dashboard_to_annotated(img)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(image.copy())

    def _crop_dashboard_to_annotated(self, img):
        h, w = img.shape[:2]
        if w < int(h * 1.18) or cv2 is None or np is None:
            return img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col_mean = gray.mean(axis=0)
        best_idx = None
        best_score = -1e9
        for idx in range(int(w * 0.45), int(w * 0.82)):
            right = gray[:, idx:]
            if right.shape[1] < 220:
                continue
            left_mean = float(col_mean[:idx].mean())
            right_mean = float(col_mean[idx:].mean())
            right_std = float(col_mean[idx:].std())
            score = (left_mean - right_mean) + max(0.0, 22.0 - right_std)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            right_mean = float(col_mean[best_idx:].mean())
            left_mean = float(col_mean[:best_idx].mean())
            if right_mean < left_mean * 0.78 and best_score > 18:
                return img[:, :best_idx].copy()
        return img

    def _extract_video_frame(self, path: Path):
        if cv2 is None or np is None:
            return None
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        frame = self._crop_dashboard_to_annotated(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        image = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(image.copy())
