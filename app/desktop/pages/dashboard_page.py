from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
)

from app.desktop.widgets.info_card import InfoCard
from app.services.statistics_service import StatisticsService


class DashboardPage(QFrame):
    def __init__(self, config_service, navigate_to=None, open_result_file=None):
        super().__init__()
        self.config_service = config_service
        self.navigate_to = navigate_to
        self.open_result_file = open_result_file
        self.stats_service = StatisticsService()
        self.output_dir = Path("runs/predict")
        self._selected_task_id: int | None = None
        self._build_ui()
        self.refresh_data()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)

        title = QLabel("工作台概览")
        title.setObjectName("pageTitle")
        desc = QLabel("聚焦最近任务与对应配置，支持拖动调整左右区域宽度，并可直接跳转到结果中心定位查看。")
        desc.setWordWrap(True)
        desc.setObjectName("pageSubtitle")
        root.addWidget(title)
        root.addWidget(desc)

        top_actions = QFrame()
        top_actions.setObjectName("toolbar")
        top_layout = QHBoxLayout(top_actions)
        top_layout.setContentsMargins(16, 12, 16, 12)
        top_layout.setSpacing(10)
        self.refresh_button = QPushButton("刷新工作台")
        self.open_output_button = QPushButton("打开输出目录")
        self.to_detect_button = QPushButton("进入检测评估")
        self.to_results_button = QPushButton("查看结果中心")
        top_layout.addWidget(self.refresh_button)
        top_layout.addWidget(self.open_output_button)
        top_layout.addStretch(1)
        top_layout.addWidget(self.to_detect_button)
        top_layout.addWidget(self.to_results_button)
        root.addWidget(top_actions)

        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(12)
        metrics_grid.setVerticalSpacing(12)
        self.card_total_tasks = InfoCard("历史任务总数", "0", "#38bdf8", "数据库中累计任务数")
        self.card_total_detections = InfoCard("累计识别病害", "0", "#22c55e", "所有任务病害总和")
        self.card_total_severe = InfoCard("累计重度风险", "0", "#fb7185", "用于快速识别高风险趋势")
        self.card_total_area = InfoCard("累计受损面积(m²)", "0.0000", "#f59e0b", "所有任务面积累加")
        for i, card in enumerate([self.card_total_tasks, self.card_total_detections, self.card_total_severe, self.card_total_area]):
            metrics_grid.addWidget(card, i // 2, i % 2)
        root.addLayout(metrics_grid)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(8)

        # 左：最近任务
        left = QFrame()
        left.setObjectName("card")
        left.setMinimumWidth(320)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        left_title = QLabel("最近任务")
        left_title.setObjectName("sectionTitle")
        left_desc = QLabel("按时间列出最近检测任务。双击任务或点击按钮后，会自动跳转到结果中心定位对应文件。")
        left_desc.setObjectName("sectionDesc")
        left_desc.setWordWrap(True)

        self.recent_list = QListWidget()
        self.recent_list.setObjectName("recentList")
        self.recent_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.recent_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.recent_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.recent_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.recent_list.verticalScrollBar().setSingleStep(20)
        self.recent_list.setSpacing(2)

        left_hint = QLabel("提示：数据来自数据库 detection_tasks 表，二次检测会覆盖旧数据。")
        left_hint.setObjectName("hintLabel")
        left_hint.setWordWrap(True)

        left_layout.addWidget(left_title)
        left_layout.addWidget(left_desc)
        left_layout.addWidget(self.recent_list, 1)
        left_layout.addWidget(left_hint)

        # 右：任务详情
        right = QFrame()
        right.setObjectName("card")
        right.setMinimumWidth(720)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(12)

        self.detail_title = QLabel("任务详情")
        self.detail_title.setObjectName("sectionTitle")
        self.detail_desc = QLabel("展示当前选中任务的检测结果、量化指标和统计信息。")
        self.detail_desc.setObjectName("sectionDesc")
        self.detail_desc.setWordWrap(True)

        self.path_label = QLabel("尚未选择任务")
        self.path_label.setObjectName("pathLabel")
        self.path_label.setWordWrap(True)

        self.meta_label = QLabel("结果类型：--")
        self.meta_label.setObjectName("metaLabel")
        self.meta_label.setWordWrap(True)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setObjectName("detailText")
        self.detail_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        bottom_actions = QHBoxLayout()
        bottom_actions.setSpacing(10)
        self.open_result_button = QPushButton("在结果中心查看该任务")
        self.open_result_button.setEnabled(False)
        self.open_file_button = QPushButton("打开结果文件")
        self.open_file_button.setEnabled(False)
        bottom_actions.addStretch(1)
        bottom_actions.addWidget(self.open_result_button)
        bottom_actions.addWidget(self.open_file_button)

        right_layout.addWidget(self.detail_title)
        right_layout.addWidget(self.detail_desc)
        right_layout.addWidget(self.path_label)
        right_layout.addWidget(self.meta_label)
        right_layout.addWidget(self.detail_text, 1)
        right_layout.addLayout(bottom_actions)

        self.main_splitter.addWidget(left)
        self.main_splitter.addWidget(right)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 7)
        self.main_splitter.setSizes([360, 920])

        root.addWidget(self.main_splitter, 1)

        self.refresh_button.clicked.connect(self.refresh_data)
        self.open_output_button.clicked.connect(self._open_output_dir)
        self.to_detect_button.clicked.connect(lambda: self.navigate_to(1) if self.navigate_to else None)
        self.to_results_button.clicked.connect(lambda: self.navigate_to(2) if self.navigate_to else None)
        self.recent_list.currentItemChanged.connect(self._on_task_changed)
        self.recent_list.itemDoubleClicked.connect(lambda *_: self._open_in_results())
        self.open_result_button.clicked.connect(self._open_in_results)
        self.open_file_button.clicked.connect(self._open_current_file)

        self.setStyleSheet(
            '''
            QLabel#pageTitle { font-size: 22px; font-weight: 700; color: white; }
            QLabel#pageSubtitle { font-size: 13px; color: #9fb6cf; }
            QFrame#toolbar, QFrame#card {
                background-color: #0b1624;
                border: 1px solid #173150;
                border-radius: 18px;
            }
            QLabel#sectionTitle { font-size: 18px; font-weight: 800; color: #f4f8ff; }
            QLabel#sectionDesc, QLabel#pathLabel { font-size: 12px; color: #95aecb; }
            QLabel#metaLabel {
                font-size: 12px;
                color: #d6e6ff;
                background-color: #0f1f32;
                border: 1px solid #18314f;
                border-radius: 12px;
                padding: 10px 12px;
            }
            QLabel#hintLabel { font-size: 12px; color: #7ec8ff; }
            QPushButton {
                min-height: 38px;
                border-radius: 12px;
                background-color: #102740;
                color: #eaf3ff;
                border: 1px solid #24476d;
                padding: 0 14px;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton:hover { background-color: #153557; }
            QPushButton:disabled { color: #6d86a3; background-color: #0b1724; border-color: #1a2d44; }
            QListWidget#recentList {
                background-color: #091321;
                color: #eaf3ff;
                border: 1px solid #173150;
                border-radius: 14px;
                padding: 6px;
                outline: none;
                font-size: 13px;
            }
            QListWidget#recentList::item {
                min-height: 42px;
                border-radius: 10px;
                padding-left: 10px;
                margin: 3px 0;
            }
            QListWidget#recentList::item:selected {
                background-color: #15365f;
                border-left: 4px solid #3dc7ff;
            }
            QListWidget#recentList QScrollBar:vertical {
                width: 12px;
                background: #08111d;
                border-radius: 6px;
                margin: 6px 2px 6px 2px;
            }
            QListWidget#recentList QScrollBar::handle:vertical {
                background: #1f4d78;
                border-radius: 6px;
                min-height: 28px;
            }
            QListWidget#recentList QScrollBar::handle:vertical:hover {
                background: #2d6ba4;
            }
            QListWidget#recentList QScrollBar::add-line:vertical,
            QListWidget#recentList QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QListWidget#recentList QScrollBar::add-page:vertical,
            QListWidget#recentList QScrollBar::sub-page:vertical {
                background: transparent;
            }
            QTextEdit#detailText {
                background-color: #0d1a2c;
                color: #dbeafe;
                border: 1px solid #18314f;
                border-radius: 14px;
                padding: 10px;
                font-family: Microsoft YaHei UI, Consolas;
                font-size: 12px;
                line-height: 1.55;
            }
            QSplitter::handle { background-color: #08111d; border-radius: 4px; }
            QSplitter::handle:hover { background-color: #173150; }
            '''
        )

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_data()

    def _open_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_dir.resolve())))

    def refresh_data(self):
        """从数据库刷新工作台数据"""
        try:
            # 获取最近任务列表
            tasks = self.stats_service.get_recent_tasks(limit=15)

            self.recent_list.clear()
            for task in tasks:
                display_text = f"{task['task_name']} ({task['created_at'].strftime('%m-%d %H:%M')})"
                item = QListWidgetItem(display_text)
                item.setData(Qt.UserRole, task['id'])
                item.setToolTip(f"源文件: {task['source_path']}\n类型: {task['task_type']}")
                self.recent_list.addItem(item)

            # 计算累计统计数据
            total_count = self.stats_service.get_total_count()
            total_detections = sum(t['total_detections'] for t in tasks)
            total_severe = sum(t['severe_count'] for t in tasks)
            total_area = sum(t['total_area_m2'] for t in tasks)

            self.card_total_tasks.set_value(str(total_count))
            self.card_total_tasks.set_note("数据库中累计任务数")
            self.card_total_detections.set_value(str(total_detections))
            self.card_total_detections.set_note("最近15个任务病害总和")
            self.card_total_severe.set_value(str(total_severe))
            self.card_total_severe.set_note(f"累计重度目标数")
            self.card_total_area.set_value(f"{total_area:.4f}")
            self.card_total_area.set_note(f"累计受损面积 m²")

            if self.recent_list.count():
                self.recent_list.setCurrentRow(0)
            else:
                self._clear_detail("数据库中暂无任务记录。")

        except Exception as e:
            print(f"刷新工作台数据失败: {e}")
            self._clear_detail(f"加载数据失败: {e}")

    def _clear_detail(self, message: str):
        self._selected_task_id = None
        self.path_label.setText(message)
        self.meta_label.setText("结果类型：--")
        self.detail_text.setText("等待任务结果...")
        self.open_result_button.setEnabled(False)
        self.open_file_button.setEnabled(False)

    def _on_task_changed(self, current, previous=None):
        if not current:
            self._clear_detail("尚未选择任务。")
            return

        task_id = current.data(Qt.UserRole)
        self._selected_task_id = task_id

        try:
            task = self.stats_service.get_task_detail(task_id)
            if not task:
                self._clear_detail("任务不存在。")
                return

            self.open_result_button.setEnabled(True)
            self.open_file_button.setEnabled(True)

            self.path_label.setText(f"当前任务：{task['task_name']}\n源文件: {task['source_path']}")

            typ = task['task_type']
            self.meta_label.setText(f"结果类型：{typ} | 检测时间：{task['created_at'].strftime('%Y-%m-%d %H:%M')}")

            summary_data = task.get('summary_data', {})
            dashboard_data = task.get('dashboard_data', [])
            severity = summary_data.get('severity_distribution', {})
            if not isinstance(severity, dict):
                severity = {}

            detail = (
                "任务详情\n"
                "---------------------------------\n"
                f"任务名称：{task['task_name']}\n"
                f"源文件路径：{task['source_path']}\n"
                f"任务类型：{task['task_type']}\n"
                f"创建时间：{task['created_at'].strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"检测统计\n"
                "---------------------------------\n"
                f"识别目标数：{task['total_detections']}\n"
                f"总受损面积：{task['total_area_m2']:.4f} m²\n"
                f"总填补体积：{task['total_volume_m3']:.5f} m³\n"
                f"平均深度：{task['avg_depth_cm']:.2f} cm\n\n"
                f"风险等级分布\n"
                "---------------------------------\n"
                f"重度：{task['severe_count']}\n"
                f"中度：{task['moderate_count']}\n"
                f"轻度：{task['mild_count']}\n\n"
                f"详细分布：轻 {severity.get('轻度', 0)} / 中 {severity.get('中度', 0)} / 重 {severity.get('重度', 0)}\n\n"
                "交互说明：\n"
                "• 双击左侧任务，或点击下方按钮，可直接跳转结果中心并定位到该任务。\n"
                "• 数据来自数据库，同一图片二次检测会覆盖旧数据。\n"
            )
            self.detail_text.setText(detail)

        except Exception as e:
            print(f"加载任务详情失败: {e}")
            self._clear_detail(f"加载任务详情失败: {e}")

    def _open_in_results(self):
        if self._selected_task_id and self.open_result_file:
            self.open_result_file(self._selected_task_id)
        elif self.navigate_to:
            self.navigate_to(2)

    def _open_current_file(self):
        """打开任务的预览图片"""
        if not self._selected_task_id:
            return

        try:
            task = self.stats_service.get_task_detail(self._selected_task_id)
            if not task:
                return

            images = task.get('images', [])
            pred_image = next((img for img in images if img['image_type'] == 'pred'), None)

            if pred_image:
                image_path = Path(pred_image['image_path'])
                if image_path.exists():
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(image_path.resolve())))
                else:
                    print(f"文件不存在: {image_path}")
            else:
                print("该任务没有预览图片")
        except Exception as e:
            print(f"打开文件失败: {e}")
