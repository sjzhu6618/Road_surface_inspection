from pathlib import Path
import time
from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
)

from app.core.schemas import TaskSummary
from app.desktop.widgets.log_panel import LogPanel
from app.desktop.widgets.metrics_panel import MetricsPanel
from app.desktop.widgets.result_preview_panel import ResultPreviewPanel
from app.desktop.widgets.source_config_panel import SourceConfigPanel
from app.services.inference_service import InferenceService


class DetectionPage(QFrame):
    def __init__(self, config_service):
        super().__init__()
        self.config_service = config_service
        self.inference_service = InferenceService(config_service=config_service)
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(1500)
        self._preview_timer.timeout.connect(self._poll_live_preview)
        self._live_preview_path = ''
        self._task_started_at = 0.0
        self._build_ui()
        self._bind_signals()

    def _wrap_scroll(self, widget: QFrame) -> QScrollArea:
        area = QScrollArea()
        area.setWidget(widget)
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        area.setStyleSheet('QScrollArea { background: transparent; border: none; } QScrollBar:vertical { width: 10px; background: #07111d; } QScrollBar::handle:vertical { background: #193756; border-radius: 5px; min-height: 24px; }')
        return area

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)

        banner = QFrame()
        banner.setObjectName('banner')
        banner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        banner_layout = QHBoxLayout(banner)
        banner_layout.setContentsMargins(22, 18, 22, 18)
        banner_layout.setSpacing(18)

        left = QVBoxLayout()
        left.setSpacing(6)
        title = QLabel('检测与评估工作台')
        title.setObjectName('pageTitle')
        subtitle = QLabel('围绕视频流采集、检测分割、深度估计、量化评估和可视化交互组织任务。')
        subtitle.setObjectName('pageSubtitle')
        subtitle.setWordWrap(True)
        left.addWidget(title)
        left.addWidget(subtitle)

        action_wrap = QHBoxLayout()
        action_wrap.setSpacing(10)
        self.open_output_button = QPushButton('打开输出目录')
        self.stop_button = QPushButton('停止任务')
        self.start_button = QPushButton('开始任务')
        self.stop_button.setObjectName('secondaryBtn')
        self.open_output_button.setObjectName('secondaryBtn')
        self.start_button.setObjectName('primaryBtn')
        for btn in [self.open_output_button, self.stop_button, self.start_button]:
            btn.setCursor(Qt.PointingHandCursor)
            btn.setMinimumWidth(116)
            btn.setMinimumHeight(40)
            action_wrap.addWidget(btn)

        banner_layout.addLayout(left, 1)
        banner_layout.addLayout(action_wrap, 0)
        root.addWidget(banner)

        content_splitter = QSplitter(Qt.Vertical)
        content_splitter.setChildrenCollapsible(False)
        content_splitter.setHandleWidth(10)

        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        top_splitter.setHandleWidth(10)

        self.config_panel = SourceConfigPanel()
        self.preview_panel = ResultPreviewPanel()
        self.metrics_panel = MetricsPanel()

        top_splitter.addWidget(self._wrap_scroll(self.config_panel))
        top_splitter.addWidget(self.preview_panel)
        top_splitter.addWidget(self._wrap_scroll(self.metrics_panel))
        top_splitter.setStretchFactor(0, 4)
        top_splitter.setStretchFactor(1, 5)
        top_splitter.setStretchFactor(2, 4)
        top_splitter.setSizes([420, 620, 420])

        self.log_panel = LogPanel()
        self.log_panel.setMinimumHeight(200)
        self.log_panel.setMaximumHeight(360)

        content_splitter.addWidget(top_splitter)
        content_splitter.addWidget(self.log_panel)
        content_splitter.setStretchFactor(0, 5)
        content_splitter.setStretchFactor(1, 2)
        content_splitter.setSizes([650, 240])

        root.addWidget(content_splitter, 1)

        self.setStyleSheet(
            """
            QFrame#banner { background-color: #0b1624; border: 1px solid #173150; border-radius: 18px; }
            QLabel#pageTitle { font-size: 24px; font-weight: 800; color: #f8fbff; }
            QLabel#pageSubtitle { font-size: 13px; color: #9fb6cf; }
            QPushButton { min-height: 40px; border-radius: 12px; padding: 0 18px; font-size: 13px; font-weight: 700; }
            QPushButton#primaryBtn { background-color: #1f75ff; color: white; border: 1px solid #3c89ff; }
            QPushButton#primaryBtn:hover { background-color: #2b82ff; }
            QPushButton#secondaryBtn { background-color: #102740; color: #d8e7ff; border: 1px solid #23476d; }
            QPushButton#secondaryBtn:hover { background-color: #163557; }
            QSplitter::handle { background: #0a1524; border-radius: 5px; }
            QSplitter::handle:hover { background: #18314d; }
            """
        )

    def _bind_signals(self) -> None:
        self.start_button.clicked.connect(self._start_task)
        self.stop_button.clicked.connect(self.inference_service.stop_task)
        self.open_output_button.clicked.connect(self._open_output_dir)
        self.inference_service.log.connect(self._on_log_line)
        self.inference_service.status.connect(self._show_status)
        self.inference_service.started.connect(self._on_started)
        self.inference_service.finished.connect(self._on_task_finished)
        self.inference_service.failed.connect(self._on_task_failed)

    def _on_started(self):
        self.log_panel.set_running()
        self.preview_panel.set_note('任务已启动，当前将用同一个多分类模型同步检测裂缝与坑洼，并等待本次任务结果返回。')
        self.log_panel.append('任务已提交，正在启动...')
        output_dir = self.config_panel.output_edit.text().strip() or 'runs/predict'
        self._live_preview_path = str(Path(output_dir) / 'preview_current.jpg')
        self._preview_timer.start()

    def _start_task(self) -> None:
        config = self.config_panel.to_config()
        if not config.model_path:
            mb = QMessageBox(QMessageBox.Warning, '缺少模型路径', '请先选择模型权重。', QMessageBox.Ok, self)
            mb.setStyleSheet('QLabel { color: black; } QMessageBox { background-color: #ffffff; }')
            mb.exec()
            return
        if not config.source:
            mb = QMessageBox(QMessageBox.Warning, '缺少输入源', '请先选择图片、视频、文件夹或摄像头。', QMessageBox.Ok, self)
            mb.setStyleSheet('QLabel { color: black; } QMessageBox { background-color: #ffffff; }')
            mb.exec()
            return
        self._preview_timer.stop()
        self._live_preview_path = ''
        self._task_started_at = time.time()
        self.log_panel.append('开始执行新的同步检测任务，界面将只展示本次任务新生成的结果。')
        self.preview_panel.clear_preview()
        self.preview_panel.set_note('当前任务将同步输出裂缝与坑洼检测结果，不再复用上一张图片的预览。')
        self.metrics_panel.reset()
        self.inference_service.start_task(config)
        persisted = self.config_service.load()
        persisted['last_model_path'] = config.model_path
        persisted['last_source_path'] = config.source
        persisted['last_source_mode'] = config.source_mode
        persisted['last_output_dir'] = config.output_dir
        persisted['last_depth_model_path'] = config.depth_model_path
        persisted['last_imgsz'] = config.imgsz
        persisted['last_frame_stride'] = config.frame_stride
        persisted['last_max_frames'] = config.max_frames
        persisted['imgsz'] = config.imgsz
        persisted['frame_stride'] = config.frame_stride
        persisted['max_frames'] = config.max_frames
        persisted['depth_model_path'] = config.depth_model_path
        persisted['source_mode'] = config.source_mode
        self.config_service.save(persisted)

    def _show_status(self, text: str) -> None:
        self.log_panel.append(f'[状态] {text}')

    def _on_log_line(self, line: str) -> None:
        self.log_panel.append(line)
        if '[进度]' in line:
            import re
            m = re.search(r'已处理\s*(\d+)\s*帧.*?\s*(\d+)\s*帧', line)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                if total > 0:
                    pct = min(99, int(done * 100 / total))
                    self.log_panel.progress.setRange(0, 100)
                    self.log_panel.progress.setValue(pct)

    def _poll_live_preview(self) -> None:
        if self._live_preview_path and Path(self._live_preview_path).exists():
            preview_path = Path(self._live_preview_path)
            if self._task_started_at and preview_path.stat().st_mtime < self._task_started_at:
                return
            pixmap = QPixmap(self._live_preview_path)
            if not pixmap.isNull():
                self.preview_panel.preview_label.setText('')
                self.preview_panel.preview_label.setScaledPixmap(pixmap)

    def _on_task_finished(self, summary) -> None:
        self._preview_timer.stop()
        if summary is None:
            self.log_panel.set_finished()
            self.log_panel.append('任务已停止，未找到可用的结果数据。')
            self.preview_panel.set_note('任务已手动停止，无可用结果。')
            return
        self.log_panel.set_finished()
        self.log_panel.append('任务执行完成。')
        self.metrics_panel.update_from_summary(summary)
        self.preview_panel.set_summary_text(summary)
        if summary.output_preview_path:
            self.preview_panel.set_preview(summary.output_preview_path)
        if summary.depth_preview_path:
            self.preview_panel.set_depth_preview(summary.depth_preview_path)

    def _on_task_failed(self, message: str) -> None:
        self._preview_timer.stop()
        self.log_panel.set_failed()
        self.log_panel.append(f'[错误] {message}')
        self.preview_panel.set_note(message)
        msg_box = QMessageBox(QMessageBox.Critical, '任务失败', message, QMessageBox.Ok, self)
        msg_box.setStyleSheet('QLabel { color: black; } QMessageBox { background-color: #ffffff; }')
        msg_box.exec()

    def _open_output_dir(self) -> None:
        path = Path(self.config_panel.output_edit.text().strip() or 'runs/predict')
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
