from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout

from app.core.schemas import TaskSummary
from app.desktop.widgets.info_card import InfoCard


class MetricsPanel(QFrame):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        title = QLabel('实时评估面板')
        title.setObjectName('sectionTitle')
        subtitle = QLabel('展示识别数量、量化指标和任务摘要，用于支撑后续大屏与报表。')
        subtitle.setWordWrap(True)
        subtitle.setObjectName('sectionSubtitle')
        root.addWidget(title)
        root.addWidget(subtitle)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        self.card_count = InfoCard('累计唯一病害数', '0', '#38bdf8')
        self.card_severe = InfoCard('重度病害数', '0', '#fb7185')
        self.card_area = InfoCard('估算受损面积(m²)', '0.0000', '#34d399')
        self.card_volume = InfoCard('估算填补体积(m³)', '0.0000', '#f59e0b')
        self.card_depth = InfoCard('平均深度(cm)', '0.00', '#22c55e', '当前为相对深度估算值')
        grid.addWidget(self.card_count, 0, 0)
        grid.addWidget(self.card_severe, 0, 1)
        grid.addWidget(self.card_area, 1, 0)
        grid.addWidget(self.card_volume, 1, 1)
        grid.addWidget(self.card_depth, 2, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        root.addLayout(grid)

        detail_title = QLabel('任务摘要')
        detail_title.setObjectName('sectionTitle')
        root.addWidget(detail_title)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['摘要字段', '值'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.table, 1)

        self.setStyleSheet(
            """
            QFrame { background-color: #0b1624; border: 1px solid #173150; border-radius: 18px; }
            QLabel#sectionTitle { color: #eaf2ff; font-size: 18px; font-weight: 800; }
            QLabel#sectionSubtitle { color: #93a8c2; font-size: 12px; }
            QTableWidget { background-color: #08111d; color: #e0ecff; border-radius: 12px; border: 1px solid #17314d; gridline-color: #17314d; alternate-background-color: #0c1725; }
            QHeaderView::section { background-color: #10253d; color: #dce9fb; border: none; padding: 8px; font-weight: 700; }
            """
        )

    def reset(self) -> None:
        self.card_count.set_value('0')
        self.card_severe.set_value('0')
        self.card_area.set_value('0.0000')
        self.card_volume.set_value('0.0000')
        self.card_depth.set_value('0.00')
        self.table.setRowCount(0)

    def update_from_summary(self, summary: TaskSummary) -> None:
        self.card_count.set_value(str(summary.unique_damage_count))
        self.card_severe.set_value(str(summary.severe_count))
        self.card_area.set_value(f'{summary.estimated_total_area_m2:.4f}')
        self.card_volume.set_value(f'{summary.estimated_total_volume_m3:.4f}')
        self.card_depth.set_value(f'{summary.average_depth_cm:.2f}')
        self.table.setRowCount(0)
        display_map = {
            'task_name': '任务名',
            'unique_damage_count': '唯一病害数',
            'severity_distribution': '等级分布',
            'estimated_total_area_m2': '总受损面积(m²)',
            'estimated_total_volume_m3': '总填补体积(m³)',
            'average_depth_cm': '平均深度(cm)',
            'source': '输入源',
            'processed_frames': '处理帧数',
            'total_frames': '总帧数',
            'output_video': '输出视频',
        }
        for key, value in summary.raw_summary.items():
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(display_map.get(str(key), str(key))))
            self.table.setItem(row, 1, QTableWidgetItem(str(value)))
