from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from app.desktop.pages.dashboard_page import DashboardPage
from app.desktop.pages.detection_page import DetectionPage
from app.desktop.pages.history_page import HistoryPage
from app.desktop.pages.results_page import ResultsPage
from app.desktop.pages.settings_page import SettingsPage


class MainWindow(QMainWindow):
    def __init__(self, config_service):
        super().__init__()
        self.config_service = config_service
        self.setWindowTitle('多任务感知路面损伤多维量化评估系统')
        self.resize(1560, 920)
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
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 18, 20, 18)
        content_layout.setSpacing(14)

        header = QFrame()
        header.setObjectName('headerBar')
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 16, 18, 16)
        title_wrap = QVBoxLayout()
        title = QLabel('多任务感知路面损伤多维量化评估系统')
        title.setObjectName('headerTitle')
        subtitle = QLabel('面向视频采集、病害检测分割、深度估计、量化评估和可视化监测的一体化桌面工作台')
        subtitle.setObjectName('headerSubtitle')
        title_wrap.addWidget(title)
        title_wrap.addWidget(subtitle)
        self.mode_chip = QLabel('桌面版原型')
        self.mode_chip.setAlignment(Qt.AlignCenter)
        self.mode_chip.setFixedWidth(110)
        header_layout.addLayout(title_wrap, 1)
        header_layout.addWidget(self.mode_chip)
        content_layout.addWidget(header)

        self.stack = QStackedWidget()
        self.stack.addWidget(DashboardPage())
        self.stack.addWidget(DetectionPage(config_service=self.config_service))
        self.stack.addWidget(ResultsPage())
        self.stack.addWidget(HistoryPage())
        self.stack.addWidget(SettingsPage(config_service=self.config_service))
        content_layout.addWidget(self.stack, 1)
        layout.addWidget(content, 1)

        self.nav_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav_list.setCurrentRow(1)
        self._apply_style()

    def _build_nav(self) -> QWidget:
        nav = QFrame()
        nav.setObjectName('navPanel')
        nav.setFixedWidth(240)
        layout = QVBoxLayout(nav)
        layout.setContentsMargins(14, 18, 14, 18)
        layout.setSpacing(12)
        brand = QLabel('路面病害系统')
        brand.setObjectName('brandTitle')
        brand_sub = QLabel('Desktop Console')
        brand_sub.setObjectName('brandSub')
        layout.addWidget(brand)
        layout.addWidget(brand_sub)
        self.nav_list = QListWidget()
        for text in ['工作台', '检测评估', '结果中心', '历史统计', '系统设置']:
            QListWidgetItem(text, self.nav_list)
        layout.addWidget(self.nav_list, 1)
        foot = QLabel('')
        foot.setWordWrap(True)
        foot.setObjectName('navFoot')
        layout.addWidget(foot)
        return nav

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget#root { background-color: #07111f; }
            QFrame#navPanel { background-color: #081321; border-right: 1px solid #17314d; }
            QFrame#headerBar { background-color: #0d1a2c; border: 1px solid #18314f; border-radius: 16px; }
            QLabel#brandTitle { color: #f8fbff; font-size: 22px; font-weight: 800; }
            QLabel#brandSub { color: #7dd3fc; font-size: 12px; }
            QLabel#navFoot { color: #8fa9c4; font-size: 12px; }
            QLabel#headerTitle { color: #f8fbff; font-size: 24px; font-weight: 800; }
            QLabel#headerSubtitle { color: #9fb6cf; font-size: 12px; }
            QListWidget { background-color: transparent; color: #d8e7ff; border: none; outline: none; }
            QListWidget::item { height: 48px; padding-left: 16px; border-radius: 10px; margin: 2px 0; }
            QListWidget::item:selected { background-color: #113157; border-left: 3px solid #38bdf8; }
            QListWidget::item:hover:!selected { background-color: #0d233c; }
            QLabel { color: #d8e7ff; }
            """
        )
        self.mode_chip.setStyleSheet('background-color: #113157; color: #7dd3fc; border-radius: 12px; padding: 8px 12px; font-weight: 700;')
