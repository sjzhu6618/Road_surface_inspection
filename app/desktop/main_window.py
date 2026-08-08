from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.desktop.pages.dashboard_page import DashboardPage
from app.desktop.pages.detection_page import DetectionPage
from app.desktop.pages.history_page import HistoryPage
from app.desktop.pages.results_page import ResultsPage
from app.desktop.pages.settings_page import SettingsPage


class MainWindow(QMainWindow):
    def __init__(self, config_service):
        super().__init__()
        self.config_service = config_service
        self._nav_collapsed = False
        self._full_nav_width = 280
        self._collapsed_nav_width = 84
        self._nav_full_labels = ['工作台', '检测评估', '结果中心', '历史统计', '系统设置']
        self._nav_short_labels = ['工', '检', '果', '历', '设']
        self.setWindowTitle('多任务感知路面损伤多维量化评估系统')
        self.resize(1500, 940)
        self.setMinimumSize(1240, 780)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName('root')
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_nav(), 0)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 10, 12, 12)
        content_layout.setSpacing(10)

        header = QFrame()
        header.setObjectName('headerBar')
        header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 10, 18, 10)
        header_layout.setSpacing(12)

        title_wrap = QVBoxLayout()
        title_wrap.setSpacing(2)
        title = QLabel('多任务感知路面损伤多维量化评估系统')
        title.setObjectName('headerTitle')
        subtitle = QLabel('面向视频采集、病害检测分割、深度估计、量化评估和可视化监测的一体化桌面工作台')
        subtitle.setObjectName('headerSubtitle')
        subtitle.setWordWrap(True)
        title_wrap.addWidget(title)
        title_wrap.addWidget(subtitle)

        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        self.mode_chip = QLabel('桌面工作台')
        self.mode_chip.setAlignment(Qt.AlignCenter)
        self.mode_chip.setMinimumWidth(96)
        self.status_chip = QLabel('就绪')
        self.status_chip.setAlignment(Qt.AlignCenter)
        self.status_chip.setMinimumWidth(72)
        right_col.addWidget(self.mode_chip, 0, Qt.AlignRight)
        right_col.addWidget(self.status_chip, 0, Qt.AlignRight)

        header_layout.addLayout(title_wrap, 1)
        header_layout.addLayout(right_col, 0)
        content_layout.addWidget(header)

        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.results_page = ResultsPage(config_service=self.config_service)
        self.dashboard_page = DashboardPage(
            config_service=self.config_service,
            navigate_to=self._go_to_page,
            open_result_file=self.open_result_file,
        )

        self.stack.addWidget(self.dashboard_page)
        self.stack.addWidget(DetectionPage(config_service=self.config_service))
        self.stack.addWidget(self.results_page)
        self.stack.addWidget(HistoryPage(config_service=self.config_service, open_result_file=self.open_result_file))
        self.stack.addWidget(SettingsPage(config_service=self.config_service))
        content_layout.addWidget(self.stack, 1)
        layout.addWidget(content, 1)

        self.nav_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav_list.currentRowChanged.connect(self._sync_page_state)
        self.nav_list.setCurrentRow(1)
        self._apply_style()

    def open_result_file(self, filename: str):
        self.nav_list.setCurrentRow(2)
        if self.results_page.config_service:
            cfg = self.results_page.config_service.load()
            from pathlib import Path
            self.results_page.output_dir = Path(cfg.get('last_output_dir') or 'runs/predict')
        self.results_page.refresh()
        self.results_page.select_file_by_name(filename)

    def _go_to_page(self, index: int):
        if 0 <= index < self.stack.count():
            self.nav_list.setCurrentRow(index)

    def _build_nav(self) -> QWidget:
        self.nav = QFrame()
        self.nav.setObjectName('navPanel')
        self.nav.setMinimumWidth(self._full_nav_width)
        self.nav.setMaximumWidth(self._full_nav_width)
        self.nav.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        layout = QVBoxLayout(self.nav)
        layout.setContentsMargins(10, 14, 10, 14)
        layout.setSpacing(12)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        self.toggle_btn = QPushButton('≡')
        self.toggle_btn.setObjectName('collapseBtn')
        self.toggle_btn.setFixedSize(34, 34)
        self.toggle_btn.clicked.connect(self._toggle_nav)
        top_row.addWidget(self.toggle_btn, 0, Qt.AlignLeft)
        top_row.addStretch(1)
        layout.addLayout(top_row)

        self.brand_wrap = QFrame()
        self.brand_wrap.setObjectName('brandWrap')
        brand_layout = QVBoxLayout(self.brand_wrap)
        brand_layout.setContentsMargins(16, 14, 16, 14)
        brand_layout.setSpacing(4)
        self.brand = QLabel('路面病害系统')
        self.brand.setObjectName('brandTitle')
        self.brand_sub = QLabel('Desktop Console')
        self.brand_sub.setObjectName('brandSub')
        brand_layout.addWidget(self.brand)
        brand_layout.addWidget(self.brand_sub)

        self.section = QLabel('功能导航')
        self.section.setObjectName('navSection')

        self.nav_list = QListWidget()
        self.nav_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        for text in self._nav_full_labels:
            QListWidgetItem(text, self.nav_list)

        self.foot = QLabel('')
        self.foot.setWordWrap(True)
        self.foot.setObjectName('navFoot')

        layout.addWidget(self.brand_wrap)
        layout.addWidget(self.section)
        layout.addWidget(self.nav_list, 1)
        layout.addWidget(self.foot)
        return self.nav

    def _toggle_nav(self):
        self._nav_collapsed = not self._nav_collapsed
        if self._nav_collapsed:
            self.nav.setMinimumWidth(self._collapsed_nav_width)
            self.nav.setMaximumWidth(self._collapsed_nav_width)
            self.brand_wrap.setVisible(False)
            self.section.setVisible(False)
            self.foot.setVisible(False)
            self.toggle_btn.setText('»')
            for i, text in enumerate(self._nav_short_labels):
                self.nav_list.item(i).setText(text)
            self.nav_list.setStyleSheet(
                'QListWidget { background-color: transparent; color: #e0ecff; border: none; outline: none; font-size: 18px; }'
                'QListWidget::item { min-height: 50px; padding-left: 0px; text-align:center; border-radius: 14px; margin: 3px 0; }'
                'QListWidget::item:selected { background-color: #133154; border-left: 0px solid transparent; color: #ffffff; }'
                'QListWidget::item:hover:!selected { background-color: #0e2238; }'
            )
        else:
            self.nav.setMinimumWidth(self._full_nav_width)
            self.nav.setMaximumWidth(self._full_nav_width)
            self.brand_wrap.setVisible(True)
            self.section.setVisible(True)
            self.foot.setVisible(True)
            self.toggle_btn.setText('≡')
            for i, text in enumerate(self._nav_full_labels):
                self.nav_list.item(i).setText(text)
            self.nav_list.setStyleSheet('')
            self._apply_style()

    def _sync_page_state(self, index: int) -> None:
        page_names = ['工作台', '检测评估', '结果中心', '历史统计', '系统设置']
        if 0 <= index < len(page_names):
            self.status_chip.setText(page_names[index])

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget#root {
                background-color: #060d16;
            }
            QFrame#navPanel {
                background-color: #08111d;
                border-right: 1px solid #16263a;
            }
            QFrame#brandWrap {
                background-color: #0b1828;
                border: 1px solid #19304b;
                border-radius: 16px;
            }
            QFrame#headerBar {
                background-color: #0b1624;
                border: 1px solid #173150;
                border-radius: 18px;
            }
            QLabel#brandTitle { color: #f4f8ff; font-size: 22px; font-weight: 800; }
            QLabel#brandSub { color: #7fb5ff; font-size: 13px; }
            QLabel#navSection { color: #7c95b6; font-size: 12px; font-weight: 700; letter-spacing: 1px; }
            QLabel#navFoot { color: #8ea4c2; font-size: 12px; line-height: 1.55; }
            QLabel#headerTitle { color: #f8fbff; font-size: 22px; font-weight: 800; }
            QLabel#headerSubtitle { color: #9fb6cf; font-size: 12px; }
            QListWidget { background-color: transparent; color: #e0ecff; border: none; outline: none; font-size: 15px; }
            QListWidget::item { min-height: 50px; padding-left: 18px; border-radius: 14px; margin: 3px 0; }
            QListWidget::item:selected { background-color: #133154; border-left: 4px solid #3cc7ff; color: #ffffff; }
            QListWidget::item:hover:!selected { background-color: #0e2238; }
            QLabel { color: #d8e7ff; }
            QPushButton#collapseBtn {
                background-color: #0f2239; color: #dbeaff; border: 1px solid #22496f; border-radius: 10px;
                font-size: 18px; font-weight: 800;
            }
            QPushButton#collapseBtn:hover { background-color: #133154; }
            """
        )
        self.mode_chip.setStyleSheet('background-color: #12335a; color: #7fd0ff; border-radius: 13px; padding: 8px 12px; font-weight: 700; font-size: 12px;')
        self.status_chip.setStyleSheet('background-color: #0e243c; color: #dce9fb; border: 1px solid #1a385c; border-radius: 11px; padding: 6px 10px; font-weight: 600; font-size: 11px;')
