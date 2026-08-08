import json
from pathlib import Path

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    cv2 = None
from PySide6.QtCore import Qt, QUrl, QTimer
from PySide6.QtGui import QDesktopServices, QPixmap, QImage
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.services.statistics_service import StatisticsService

try:
    import matplotlib as mpl
    mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei UI', 'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
    mpl.rcParams['axes.unicode_minus'] = False
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
    FigureCanvas = None
    Figure = None


class ScaledImageLabel(QLabel):
    def __init__(self, placeholder_text='暂无预览'):
        super().__init__(placeholder_text)
        self._pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 240)
        self.setStyleSheet(
            'background-color: #050c15; border: 1px solid #173150; border-radius: 18px; color: #8da6c5; font-size: 14px;'
        )

    def set_image(self, pixmap: QPixmap | None):
        if pixmap is None or pixmap.isNull():
            self._pixmap = None
            self.setText('暂无预览')
            self.setPixmap(QPixmap())
        else:
            self._pixmap = pixmap
            self._refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap is not None:
            self._refresh()

    def _refresh(self):
        if self._pixmap is None or self._pixmap.isNull():
            self.setText('暂无预览')
            self.setPixmap(QPixmap())
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setText('')
        self.setPixmap(scaled)


class JsonMetricCard(QFrame):
    def __init__(self, title: str = '', value: str = ''):
        super().__init__()
        self.setObjectName('jsonMetricCard')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(5)
        self.title_label = QLabel(title)
        self.title_label.setObjectName('jsonMetricTitle')
        self.value_label = QLabel(value)
        self.value_label.setObjectName('jsonMetricValue')
        self.value_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_data(self, title: str, value: str):
        self.title_label.setText(title)
        self.value_label.setText(value)


class SummaryChartsWidget(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName('summaryCharts')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.header = QLabel('任务统计分析')
        self.header.setObjectName('chartHeader')
        self.sub = QLabel('针对当前 JSON 摘要自动提取关键指标，生成更适合系统查看的可视化统计结果。')
        self.sub.setObjectName('chartSub')
        self.sub.setWordWrap(True)
        layout.addWidget(self.header)
        layout.addWidget(self.sub)

        self.metrics_grid = QGridLayout()
        self.metrics_grid.setContentsMargins(0, 0, 0, 0)
        self.metrics_grid.setHorizontalSpacing(10)
        self.metrics_grid.setVerticalSpacing(10)

        self.metric_cards = []
        titles = ['识别目标数', '总受损面积', '总填补体积', '最大相对深度']
        for idx, title in enumerate(titles):
            card = JsonMetricCard(title, '--')
            self.metric_cards.append(card)
            self.metrics_grid.addWidget(card, idx // 2, idx % 2)
        layout.addLayout(self.metrics_grid)

        chart_wrap = QFrame()
        chart_wrap.setObjectName('chartWrap')
        chart_layout = QGridLayout(chart_wrap)
        chart_layout.setContentsMargins(10, 10, 10, 10)
        chart_layout.setHorizontalSpacing(12)
        chart_layout.setVerticalSpacing(12)

        if HAVE_MPL:
            self.figure_donut = Figure(figsize=(4.0, 3.1), dpi=100)
            self.figure_donut.patch.set_facecolor('#0b1624')
            self.canvas_donut = FigureCanvas(self.figure_donut)
            self.canvas_donut.setMinimumHeight(240)
            self.canvas_donut.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            self.figure_bar = Figure(figsize=(5.0, 3.1), dpi=100)
            self.figure_bar.patch.set_facecolor('#0b1624')
            self.canvas_bar = FigureCanvas(self.figure_bar)
            self.canvas_bar.setMinimumHeight(240)
            self.canvas_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            chart_layout.addWidget(self.canvas_donut, 0, 0)
            chart_layout.addWidget(self.canvas_bar, 0, 1)
            chart_layout.setColumnStretch(0, 1)
            chart_layout.setColumnStretch(1, 1)
        else:
            self.canvas_donut = QTextEdit()
            self.canvas_donut.setReadOnly(True)
            self.canvas_donut.setObjectName('detailText')
            self.canvas_donut.setMinimumHeight(220)
            chart_layout.addWidget(self.canvas_donut, 0, 0, 1, 2)
            self.figure_donut = None
            self.figure_bar = None
            self.canvas_bar = None
        layout.addWidget(chart_wrap, 1)

        self.insight_box = QTextEdit()
        self.insight_box.setReadOnly(True)
        self.insight_box.setMinimumHeight(92)
        self.insight_box.setMaximumHeight(140)
        self.insight_box.setObjectName('detailText')
        layout.addWidget(self.insight_box, 0)

    def update_from_summary(self, summary: dict):
        metrics = summary.get('metrics', {})
        severity = summary.get('severity', {})

        self.metric_cards[0].set_data('识别目标数', str(metrics.get('count_display', '--')))
        self.metric_cards[1].set_data('总受损面积', metrics.get('area_display', '--'))
        self.metric_cards[2].set_data('总填补体积', metrics.get('volume_display', '--'))
        self.metric_cards[3].set_data('最大相对深度', metrics.get('depth_display', '--'))

        if HAVE_MPL:
            self._render_donut(severity)
            self._render_bar(metrics)
        else:
            self.canvas_donut.setText('当前环境未启用 matplotlib，已退化为文字统计视图。')

        self.insight_box.setText(self._build_insight_text(metrics, severity))

    def _build_insight_text(self, metrics: dict, severity: dict) -> str:
        total = int(float(metrics.get('count', 0.0)))
        area = metrics.get('area_display', '--')
        volume = metrics.get('volume_display', '--')
        depth = metrics.get('depth_display', '--')
        s_light = int(severity.get('轻度', 0))
        s_mid = int(severity.get('中度', 0))
        s_heavy = int(severity.get('重度', 0))

        if total == 0:
            return '当前 JSON 中未识别出可用于统计的病害数据。'

        dominant = '轻度'
        if s_heavy >= max(s_light, s_mid):
            dominant = '重度'
        elif s_mid >= max(s_light, s_heavy):
            dominant = '中度'

        return (
            f'任务摘要：当前结果共统计到 {total} 个病害目标，累计受损面积约 {area}，估算填补体积约 {volume}，最大相对深度约 {depth}。\n'
            f'风险分布：轻度 {s_light} 个，中度 {s_mid} 个，重度 {s_heavy} 个，当前以 {dominant} 风险为主。\n'
            f'建议：可优先结合"重度数量"和"填补体积"判断是否需要尽快安排养护。'
        )

    def _render_donut(self, severity: dict):
        self.figure_donut.clear()
        ax = self.figure_donut.add_subplot(111)
        ax.set_facecolor('#0b1624')
        for sp in ax.spines.values():
            sp.set_visible(False)

        labels = ['轻度', '中度', '重度']
        values = [int(severity.get(k, 0)) for k in labels]
        colors = ['#ffd84a', '#ffad33', '#ef5b5b']
        total = sum(values)

        if total > 0:
            ax.pie(values, colors=colors, startangle=90, wedgeprops={'width': 0.34, 'edgecolor': '#0b1624'})
            ax.text(0, 0.08, '病害总数', ha='center', va='center', color='#dce9fb', fontsize=12, fontweight='bold')
            ax.text(0, -0.10, f'{total}', ha='center', va='center', color='#7fd0ff', fontsize=22, fontweight='bold')
            ax.text(0.5, 0.05, f'轻度 {values[0]}   中度 {values[1]}   重度 {values[2]}',
                    transform=ax.transAxes, ha='center', va='center', color='#d8e7ff', fontsize=9)
        else:
            ax.text(0.5, 0.5, '暂无等级分布数据', transform=ax.transAxes, ha='center', va='center', color='#8da6c5', fontsize=11)
        ax.set_title('风险等级分布', color='#f4f8ff', fontsize=12, pad=8)
        self.figure_donut.subplots_adjust(left=0.06, right=0.94, top=0.88, bottom=0.10)
        self.canvas_donut.draw_idle()

    def _render_bar(self, metrics: dict):
        self.figure_bar.clear()
        ax = self.figure_bar.add_subplot(111)
        ax.set_facecolor('#0b1624')
        for sp in ax.spines.values():
            sp.set_visible(False)

        names = ['病害数量', '受损面积', '填补体积', '最大深度']
        idx_values = [
            float(metrics.get('count_index', 0.0)),
            float(metrics.get('area_index', 0.0)),
            float(metrics.get('volume_index', 0.0)),
            float(metrics.get('depth_index', 0.0)),
        ]
        actual_labels = [
            str(metrics.get('count_display', '--')),
            metrics.get('area_display', '--'),
            metrics.get('volume_display', '--'),
            metrics.get('depth_display', '--'),
        ]
        bar_colors = ['#3db5ff', '#32d6a4', '#ffbf3d', '#ff7a7a']

        y_pos = list(range(len(names)))
        ax.barh(y_pos, idx_values, color=bar_colors, edgecolor='none', height=0.54)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, color='#dce9fb', fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(['0', '25', '50', '75', '100'], color='#95aecb', fontsize=8)
        ax.grid(axis='x', color='#173150', linestyle='--', alpha=0.8)
        ax.set_title('关键指标风险指数', color='#f4f8ff', fontsize=12, pad=8)
        ax.invert_yaxis()

        for i, (val, label) in enumerate(zip(idx_values, actual_labels)):
            ax.text(min(val + 1.2, 97), i, label, va='center', ha='left', color='#f8fbff', fontsize=8.5)

        self.figure_bar.subplots_adjust(left=0.20, right=0.96, top=0.86, bottom=0.16)
        self.canvas_bar.draw_idle()


class ResultsPage(QFrame):
    def __init__(self, config_service=None):
        super().__init__()
        self.config_service = config_service
        self.stats_service = StatisticsService()
        self.output_dir = Path('runs/predict')
        self.current_task_id: int | None = None
        self.current_file: Path | None = None
        self.current_file_type: str = ''
        self.video_cap = None
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self._next_video_frame)
        self.video_fps = 25.0
        self.video_frame_count = 0
        self.video_current_frame = 0
        self._slider_dragging = False
        self._build_ui()
        self.refresh()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        title = QLabel('结果中心')
        title.setObjectName('pageTitle')
        desc = QLabel('统一浏览任务生成的图片、视频和 JSON 摘要，支持预览、内容查看和快速打开输出文件。')
        desc.setObjectName('pageSubtitle')
        desc.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(desc)

        toolbar = QFrame()
        toolbar.setObjectName('toolbar')
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(14, 10, 14, 10)
        toolbar_layout.setSpacing(10)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(['全部任务', '单图任务', '视频任务'])
        self.filter_combo.setMinimumWidth(150)
        self.refresh_button = QPushButton('刷新任务列表')
        self.open_dir_button = QPushButton('打开输出目录')
        self.open_file_button = QPushButton('打开当前文件')
        self.open_file_button.setEnabled(False)
        toolbar_layout.addWidget(self.filter_combo, 0)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(self.refresh_button, 0)
        toolbar_layout.addWidget(self.open_dir_button, 0)
        toolbar_layout.addWidget(self.open_file_button, 0)
        root.addWidget(toolbar)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(8)

        left_card = QFrame()
        left_card.setObjectName('card')
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)
        left_title = QLabel('结果文件列表')
        left_title.setObjectName('sectionTitle')
        left_desc = QLabel('按文件类型与时间浏览当前任务的输出内容。')
        left_desc.setObjectName('sectionDesc')
        left_desc.setWordWrap(True)
        self.list_widget = QListWidget()
        self.list_widget.setObjectName('resultList')
        left_layout.addWidget(left_title)
        left_layout.addWidget(left_desc)
        left_layout.addWidget(self.list_widget, 1)

        right_card = QFrame()
        right_card.setObjectName('card')
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)

        header_row = QHBoxLayout()
        header_row.setSpacing(12)
        title_col = QVBoxLayout()
        title_col.setSpacing(6)
        self.preview_title = QLabel('summary.json')
        self.preview_title.setObjectName('sectionTitle')
        self.path_label = QLabel('请选择左侧文件查看详情。')
        self.path_label.setObjectName('pathLabel')
        self.path_label.setWordWrap(True)
        title_col.addWidget(self.preview_title)
        title_col.addWidget(self.path_label)
        header_row.addLayout(title_col, 1)

        badge_col = QVBoxLayout()
        badge_col.setSpacing(8)
        self.type_badge = QLabel('统计分析')
        self.meta_badge = QLabel('--')
        for badge in (self.type_badge, self.meta_badge):
            badge.setAlignment(Qt.AlignCenter)
            badge.setMinimumWidth(96)
        badge_col.addWidget(self.type_badge, 0, Qt.AlignRight)
        badge_col.addWidget(self.meta_badge, 0, Qt.AlignRight)
        header_row.addLayout(badge_col, 0)
        right_layout.addLayout(header_row)

        self.preview_stack = QStackedWidget()
        self.preview_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        image_page = QWidget()
        image_layout = QVBoxLayout(image_page)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(10)
        self.preview_label = ScaledImageLabel()
        image_layout.addWidget(self.preview_label, 1)

        video_page = QWidget()
        video_layout = QVBoxLayout(video_page)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(10)
        self.video_preview = ScaledImageLabel()
        self.video_preview.setMinimumHeight(360)
        controls = QFrame()
        controls.setObjectName('controlsBar')
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(12, 10, 12, 10)
        controls_layout.setSpacing(8)
        self.play_button = QPushButton('播放')
        self.pause_button = QPushButton('暂停')
        self.stop_button = QPushButton('停止')
        self.position_label = QLabel('00:00 / 00:00')
        self.position_label.setObjectName('timeLabel')
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        controls_layout.addWidget(self.play_button, 0)
        controls_layout.addWidget(self.pause_button, 0)
        controls_layout.addWidget(self.stop_button, 0)
        controls_layout.addWidget(self.position_slider, 1)
        controls_layout.addWidget(self.position_label, 0)
        video_layout.addWidget(self.video_preview, 1)
        video_layout.addWidget(controls, 0)

        summary_page = QWidget()
        summary_layout = QVBoxLayout(summary_page)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(10)
        self.summary_charts = SummaryChartsWidget()
        summary_layout.addWidget(self.summary_charts, 1)

        empty_page = QWidget()
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.setContentsMargins(0, 0, 0, 0)
        self.empty_label = QLabel('当前没有可预览内容')
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet('color: #8da6c5; font-size: 14px; background-color: #07111d; border: 1px dashed #21466e; border-radius: 16px;')
        empty_layout.addWidget(self.empty_label, 1)

        self.preview_stack.addWidget(image_page)
        self.preview_stack.addWidget(video_page)
        self.preview_stack.addWidget(summary_page)
        self.preview_stack.addWidget(empty_page)
        right_layout.addWidget(self.preview_stack, 1)

        main_splitter.addWidget(left_card)
        main_splitter.addWidget(right_card)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 5)
        main_splitter.setSizes([320, 980])
        root.addWidget(main_splitter, 1)

        self.filter_combo.currentIndexChanged.connect(self.refresh)
        self.refresh_button.clicked.connect(self.refresh)
        self.open_dir_button.clicked.connect(self._open_output_dir)
        self.open_file_button.clicked.connect(self._open_current_file)
        self.list_widget.currentItemChanged.connect(self._on_item_changed)
        self.play_button.clicked.connect(self._play_video)
        self.pause_button.clicked.connect(self._pause_video)
        self.stop_button.clicked.connect(self._stop_video)
        self.position_slider.sliderPressed.connect(self._on_slider_pressed)
        self.position_slider.sliderReleased.connect(self._on_slider_released)
        self.position_slider.valueChanged.connect(self._on_slider_value_changed)

        self.setStyleSheet("""
            QLabel#pageTitle { font-size: 22px; font-weight: 800; color: #f8fbff; }
            QLabel#pageSubtitle { font-size: 13px; color: #9fb6cf; }
            QFrame#toolbar, QFrame#card, QFrame#summaryCharts, QFrame#chartWrap {
                background-color: #0b1624;
                border: 1px solid #173150;
                border-radius: 18px;
            }
            QFrame#jsonMetricCard {
                background-color: #10233a;
                border: 1px solid #21466e;
                border-radius: 16px;
            }
            QLabel#jsonMetricTitle { color: #9ec3e6; font-size: 12px; font-weight: 700; }
            QLabel#jsonMetricValue { color: #f8fbff; font-size: 14px; font-weight: 800; }
            QLabel#chartHeader { color: #f4f8ff; font-size: 16px; font-weight: 800; }
            QLabel#chartSub { color: #95aecb; font-size: 12px; }
            QLabel#sectionTitle { font-size: 17px; font-weight: 800; color: #f4f8ff; }
            QLabel#sectionDesc { font-size: 12px; color: #95aecb; }
            QLabel#pathLabel { color: #d9e8fb; font-size: 12px; }
            QLabel#timeLabel { color: #dbe9ff; font-size: 12px; font-weight: 700; min-width: 96px; }
            QFrame#controlsBar {
                background-color: #091321;
                border: 1px solid #173150;
                border-radius: 14px;
            }
            QComboBox, QPushButton {
                min-height: 38px; border-radius: 12px; background-color: #102740; color: #eaf3ff;
                border: 1px solid #24476d; padding: 0 14px; font-size: 13px; font-weight: 700;
            }
            QComboBox:hover, QPushButton:hover { background-color: #153557; }
            QListWidget#resultList {
                background-color: #091321; color: #eaf3ff; border: 1px solid #173150; border-radius: 14px;
                padding: 6px; outline: none; font-size: 13px;
            }
            QListWidget#resultList::item {
                min-height: 38px; border-radius: 12px; padding-left: 12px; margin: 4px 0;
            }
            QListWidget#resultList::item:selected {
                background-color: #15365f; border-left: 4px solid #3dc7ff;
            }
            QTextEdit#detailText {
                background-color: #091321; color: #d8e7ff; border: 1px solid #173150; border-radius: 14px;
                padding: 10px; font-family: Consolas, Microsoft YaHei UI; font-size: 12px;
            }
            QSlider::groove:horizontal { height: 8px; background: #102740; border-radius: 4px; }
            QSlider::sub-page:horizontal { background: #2b7cff; border-radius: 4px; }
            QSlider::add-page:horizontal { background: #173150; border-radius: 4px; }
            QSlider::handle:horizontal { background: #ecf5ff; width: 16px; margin: -4px 0; border-radius: 8px; }
            QSplitter::handle { background: #08111d; border-radius: 5px; }
            QSplitter::handle:hover { background: #18314d; }
        """)
        self.type_badge.setStyleSheet('background-color: #12335a; color: #7fd0ff; border-radius: 12px; padding: 8px 12px; font-weight: 700;')
        self.meta_badge.setStyleSheet('background-color: #0e243c; color: #dce9fb; border: 1px solid #1a385c; border-radius: 12px; padding: 8px 12px; font-weight: 600;')

    def refresh(self) -> None:
        """从数据库刷新任务列表，以树状结构组织任务文件"""
        self.list_widget.clear()

        try:
            mode = self.filter_combo.currentText()
            task_type = ''
            if mode == '单图任务':
                task_type = '单图'
            elif mode == '视频任务':
                task_type = '视频流'

            tasks = self.stats_service.get_recent_tasks(limit=50)

            for task in tasks:
                if task_type and task['task_type'] != task_type:
                    continue

                task_id = task['id']
                task_name = task['task_name']

                from app.services.db_manager import get_task_images
                images = get_task_images(task_id)

                parent_item = QListWidgetItem(f"{task_name} ({task['created_at'].strftime('%m-%d %H:%M')})")
                parent_item.setData(Qt.UserRole, {'type': 'task', 'task_id': task_id})
                parent_item.setFlags(parent_item.flags() & ~Qt.ItemIsSelectable)
                self.list_widget.addItem(parent_item)

                if images:
                    pred_images = [img for img in images if img['image_type'] == 'pred']
                    depth_images = [img for img in images if img['image_type'] == 'depth']

                    for img in depth_images:
                        file_item = QListWidgetItem(f"    {img['image_name']}")
                        file_item.setData(Qt.UserRole, {
                            'type': 'file',
                            'task_id': task_id,
                            'file_type': 'depth',
                            'file_path': img['image_path'],
                            'file_name': img['image_name']
                        })
                        self.list_widget.addItem(file_item)

                    for img in pred_images:
                        file_item = QListWidgetItem(f"    {img['image_name']}")
                        file_item.setData(Qt.UserRole, {
                            'type': 'file',
                            'task_id': task_id,
                            'file_type': 'pred',
                            'file_path': img['image_path'],
                            'file_name': img['image_name']
                        })
                        self.list_widget.addItem(file_item)

                if task.get('result_video_path'):
                    video_item = QListWidgetItem(f"    {task_name}.mp4")
                    video_item.setData(Qt.UserRole, {
                        'type': 'file',
                        'task_id': task_id,
                        'file_type': 'video',
                        'file_path': task['result_video_path'],
                        'file_name': f"{task_name}.mp4"
                    })
                    self.list_widget.addItem(video_item)

                summary_item = QListWidgetItem(f"     summary.json (统计分析)")
                summary_item.setData(Qt.UserRole, {
                    'type': 'json',
                    'task_id': task_id,
                    'json_type': 'summary',
                    'file_name': 'summary.json'
                })
                self.list_widget.addItem(summary_item)

            if self.list_widget.count() > 0:
                self.list_widget.setCurrentRow(1)
            else:
                self.current_task_id = None
                self.open_file_button.setEnabled(False)
                self.preview_title.setText('任务详情')
                self.path_label.setText('数据库中暂无任务记录。')
                self.type_badge.setText('空列表')
                self.meta_badge.setText('等待生成')
                self.preview_stack.setCurrentIndex(3)

        except Exception as e:
            print(f"刷新任务列表失败: {e}")

    def _on_item_changed(self, current: QListWidgetItem, previous=None) -> None:
        if not current:
            return

        data = current.data(Qt.UserRole)
        if not data:
            return

        item_type = data.get('type')
        if item_type == 'task':
            return

        if item_type == 'file':
            self._show_file(data)
        elif item_type == 'json':
            self._show_json_file(data)

    def _show_file(self, data: dict):
        """显示图片或视频文件 - 从数据库路径加载本地文件"""
        task_id = data['task_id']
        file_path = data['file_path']
        file_type = data['file_type']
        file_name = data['file_name']

        self.current_file = Path(file_path)
        self.current_file_type = file_type

        task = self.stats_service.get_task_detail(task_id)
        if not task:
            return

        self.preview_title.setText(file_name)
        self.path_label.setText(str(self.current_file))

        if file_type == 'pred':
            self.type_badge.setText('图片结果')
        elif file_type == 'depth':
            self.type_badge.setText('深度视图')
        elif file_type == 'video':
            self.type_badge.setText('视频结果')
        else:
            self.type_badge.setText('图片结果')

        self.meta_badge.setText(f"{task['total_detections']}个病害")

        if file_type in ['pred', 'depth']:
            if self.current_file.exists():
                self._show_image(self.current_file)
            else:
                self.preview_stack.setCurrentIndex(3)
                self.empty_label.setText(f'文件不存在：{file_name}\n\n数据库记录的路径：\n{file_path}')

        elif file_type == 'video':
            if self.current_file.exists():
                self._show_video(self.current_file, task)
            else:
                self.preview_stack.setCurrentIndex(3)
                self.empty_label.setText(f'视频文件不存在：{file_name}')
        else:
            if self.current_file.exists():
                self._show_image(self.current_file)
            else:
                self.preview_stack.setCurrentIndex(3)
                self.empty_label.setText(f'文件不存在：{file_name}')

        self.open_file_button.setEnabled(True)

    def _show_json_file(self, data: dict):
        """显示 summary.json 的可视化统计图表"""
        task_id = data['task_id']
        file_name = data['file_name']

        self.current_file_type = 'summary'

        task = self.stats_service.get_task_detail(task_id)
        if not task:
            return

        summary_data = task.get('summary_data', {})
        dashboard_data = task.get('dashboard_data', [])

        self.preview_title.setText(file_name)
        self.path_label.setText(f"任务：{task['task_name']}")
        self.type_badge.setText('统计分析')
        self.meta_badge.setText(f"{task['total_detections']}个病害")

        self._show_summary_chart(summary_data, dashboard_data)

        self.open_file_button.setEnabled(False)

    def _show_summary_chart(self, summary_data: dict, dashboard_data: list):
        """显示 Summary 的可视化图表"""
        summary_for_chart = self._build_summary_from_db_data(dashboard_data, summary_data)
        self.summary_charts.update_from_summary(summary_for_chart)
        self.preview_stack.setCurrentIndex(2)

    def _show_image(self, image_path: Path):
        """从本地路径加载并显示图片"""
        self.type_badge.setText('图片结果')

        try:
            pixmap = QPixmap(str(image_path))

            if pixmap.isNull():
                self.preview_label.set_image(None)
                self.empty_label.setText(
                    f'无法加载图片：{image_path.name}\n\n请检查:\n1. 文件格式是否正确(.jpg/.png等)\n2. 文件是否完整')
                self.preview_stack.setCurrentIndex(3)
            else:
                self.preview_label.set_image(pixmap)
                self.preview_stack.setCurrentIndex(0)

        except Exception as e:
            self.preview_label.set_image(None)
            self.empty_label.setText(f'加载图片出错：\n{str(e)}')
            self.preview_stack.setCurrentIndex(3)

    def _show_video(self, video_path: Path, task: dict):
        self.type_badge.setText('视频结果')
        self._stop_video()

        self.video_cap = cv2.VideoCapture(str(video_path))
        if not self.video_cap.isOpened():
            return

        self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.video_frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.video_current_frame = 0
        self.position_slider.setRange(0, max(0, self.video_frame_count - 1))
        self.position_slider.setValue(0)

        ok, frame = self.video_cap.read()
        if ok:
            self.video_current_frame = 1
            self._display_video_frame(frame)

        self.preview_stack.setCurrentIndex(1)

    def _build_summary_from_db_data(self, dashboard_data: list, summary_data: dict):
        severity = {
            '轻度': summary_data.get('severity_distribution', {}).get('轻度', 0),
            '中度': summary_data.get('severity_distribution', {}).get('中度', 0),
            '重度': summary_data.get('severity_distribution', {}).get('重度', 0),
        }

        metrics = {
            'count': summary_data.get('unique_damage_count', len(dashboard_data)),
            'area': summary_data.get('estimated_total_area_m2', 0),
            'volume': summary_data.get('estimated_total_volume_m3', 0),
            'depth_cm': summary_data.get('average_depth_cm', 0),
        }

        metrics['count_index'] = min(100.0, metrics['count'] / 20.0 * 100.0)
        metrics['area_index'] = min(100.0, metrics['area'] / 2.0 * 100.0)
        metrics['volume_index'] = min(100.0, metrics['volume'] / 0.08 * 100.0)
        metrics['depth_index'] = min(100.0, metrics['depth_cm'] / 12.0 * 100.0)
        metrics['count_display'] = int(metrics['count']) if float(metrics['count']).is_integer() else f"{metrics['count']:.1f}"
        metrics['area_display'] = f"{metrics['area']:.4f} m²"
        metrics['volume_display'] = f"{metrics['volume']:.4f} m³"
        metrics['depth_display'] = f"{metrics['depth_cm']:.2f} cm"

        return {'metrics': metrics, 'severity': severity}

    def _play_video(self):
        if self.video_cap and not self.video_timer.isActive():
            interval = max(20, int(1000 / max(self.video_fps, 1.0)))
            self.video_timer.start(interval)

    def _pause_video(self):
        self.video_timer.stop()

    def _stop_video(self):
        self.video_timer.stop()
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.position_slider.setValue(0)

    def _next_video_frame(self):
        if not self.video_cap:
            return
        ok, frame = self.video_cap.read()
        if not ok:
            self.video_timer.stop()
            return
        self.video_current_frame += 1
        self._display_video_frame(frame)
        if not self._slider_dragging:
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(self.video_current_frame)
            self.position_slider.blockSignals(False)
        self.position_label.setText(f'{self._format_frame_time(self.video_current_frame)} / {self._format_frame_time(self.video_frame_count)}')

    def _display_video_frame(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg.copy())
            if not pixmap.isNull():
                self.video_preview.set_image(pixmap)
            else:
                print("警告: 视频帧转换失败")
        except Exception as e:
            print(f"显示视频帧失败: {e}")

    def _on_slider_pressed(self):
        self._slider_dragging = True

    def _on_slider_released(self):
        self._slider_dragging = False
        if self.video_cap:
            target_frame = self.position_slider.value()
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            self.video_current_frame = target_frame
            ok, frame = self.video_cap.read()
            if ok:
                self.video_current_frame += 1
                self._display_video_frame(frame)

    def _on_slider_value_changed(self, value: int):
        if self._slider_dragging:
            self.position_label.setText(f'{self._format_frame_time(value)} / {self._format_frame_time(self.video_frame_count)}')

    def _format_frame_time(self, frame_index: int) -> str:
        seconds = frame_index / max(self.video_fps, 1.0)
        total = int(max(0, seconds))
        m, s = divmod(total, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f'{h:02d}:{m:02d}:{s:02d}'
        return f'{m:02d}:{s:02d}'

    def select_file_by_name(self, filename: str) -> bool:
        """根据任务ID定位到对应任务的summary.json"""
        try:
            task_id = int(filename)

            self.refresh()

            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                data = item.data(Qt.UserRole)

                if not data:
                    continue

                if data.get('type') == 'task' and data.get('task_id') == task_id:
                    for j in range(i + 1, self.list_widget.count()):
                        child_item = self.list_widget.item(j)
                        child_data = child_item.data(Qt.UserRole)

                        if child_data and child_data.get('type') == 'task':
                            break

                        if child_data and child_data.get('type') == 'json':
                            self.list_widget.setCurrentItem(child_item)
                            self.list_widget.scrollToItem(child_item)
                            return True
                    return True
            return False
        except (ValueError, TypeError):
            return False

    def _open_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_dir.resolve())))

    def _open_current_file(self) -> None:
        if self.current_file and self.current_file.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_file.resolve())))
