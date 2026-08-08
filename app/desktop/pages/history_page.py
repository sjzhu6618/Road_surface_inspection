import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.desktop.widgets.info_card import InfoCard
from app.services.statistics_service import StatisticsService

try:
    import matplotlib as mpl

    mpl.rcParams["font.sans-serif"] = [
        "Microsoft YaHei UI", "Microsoft YaHei", "SimHei",
        "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans",
    ]
    mpl.rcParams["axes.unicode_minus"] = False
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
    FigureCanvas = None
    Figure = None


class ChartCard(QFrame):
    def __init__(self, title: str, subtitle: str, figsize=(5, 3)):
        super().__init__()
        self.setObjectName("chartCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setObjectName("chartTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("chartSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)

        if HAVE_MPL:
            self.figure = Figure(figsize=figsize, dpi=96)
            self.figure.patch.set_facecolor("#0b1624")
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.addWidget(self.canvas, 1)
        else:
            self.figure = None
            self.canvas = QTextEdit()
            self.canvas.setReadOnly(True)
            self.canvas.setObjectName("chartFallback")
            layout.addWidget(self.canvas, 1)

    def clear(self, message: str = "暂无历史数据"):
        if HAVE_MPL:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_facecolor("#0b1624")
            ax.text(0.5, 0.5, message, ha="center", va="center",
                    color="#8da6c5", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            self.canvas.draw_idle()
        else:
            self.canvas.setText(message)


class HistoryPage(QFrame):
    def __init__(self, config_service=None, open_result_file=None):
        super().__init__()
        self.config_service = config_service
        self.open_result_file = open_result_file
        self.stats_service = StatisticsService()
        self.output_dir = Path("runs/predict")
        self.entries: List[Dict[str, Any]] = []
        self._build_ui()
        self.refresh_data()

    # ------------------------------------------------------------------ #
    #  UI 构建                                                              #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { background: #0b1624; width: 8px; border-radius: 4px; }"
            "QScrollBar::handle:vertical { background: #244466; border-radius: 4px; min-height: 30px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }"
        )
        inner = QWidget()
        inner.setObjectName("scrollInner")
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        root = QVBoxLayout(inner)
        root.setContentsMargins(0, 6, 0, 12)
        root.setSpacing(10)

        title = QLabel("历史统计数据大屏")
        title.setObjectName("pageTitle")
        desc = QLabel(
            "跨任务累计指标 · 几何特征分布 · 风险结构对比 · 历史趋势追踪"
            " （从数据库读取）"
        )
        desc.setObjectName("pageSubtitle")
        desc.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(desc)

        toolbar = QFrame()
        toolbar.setObjectName("toolbar")
        tbl = QHBoxLayout(toolbar)
        tbl.setContentsMargins(16, 10, 16, 10)
        tbl.setSpacing(10)
        self.refresh_button = QPushButton("刷新统计")
        self.open_output_button = QPushButton("打开输出目录")
        self.open_selected_button = QPushButton("在结果中心查看选中")
        self.open_selected_button.setEnabled(False)
        tbl.addWidget(self.refresh_button)
        tbl.addWidget(self.open_output_button)
        tbl.addStretch(1)
        tbl.addWidget(self.open_selected_button)
        root.addWidget(toolbar)

        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(10)
        self.card_total_tasks = InfoCard("历史任务总数", "0", "#38bdf8", "从数据库读取")
        self.card_total_count = InfoCard("累计识别病害", "0", "#22c55e", "所有任务 count 之和")
        self.card_total_severe = InfoCard("累计重度目标", "0", "#fb7185", "所有任务 severe 之和")
        self.card_total_area = InfoCard("累计受损面积", "0 m²", "#f59e0b", "所有任务 area 之和")
        self.card_total_volume = InfoCard("累计填补体积", "0 m³", "#a78bfa", "所有任务 volume 之和")
        self.card_max_depth = InfoCard("平均深度", "0 cm", "#34d399", "所有任务 depth 平均值")
        for card in [
            self.card_total_tasks, self.card_total_count, self.card_total_severe,
            self.card_total_area, self.card_total_volume, self.card_max_depth,
        ]:
            kpi_row.addWidget(card, 1)
        root.addLayout(kpi_row)

        mid_splitter = QSplitter(Qt.Horizontal)
        mid_splitter.setChildrenCollapsible(False)
        mid_splitter.setHandleWidth(6)

        self.type_chart = ChartCard("任务类型分布", "图片 / 视频任务占比", figsize=(4, 3))
        self.scatter_chart = ChartCard("面积 vs 体积分布", "各任务几何损伤特征散点", figsize=(5, 3))
        self.radar_chart = ChartCard("多任务风险雷达", "最近 5 个任务风险结构对比", figsize=(4, 3))

        mid_splitter.addWidget(self.type_chart)
        mid_splitter.addWidget(self.scatter_chart)
        mid_splitter.addWidget(self.radar_chart)
        mid_splitter.setStretchFactor(0, 3)
        mid_splitter.setStretchFactor(1, 4)
        mid_splitter.setStretchFactor(2, 3)
        mid_splitter.setSizes([370, 490, 370])
        mid_splitter.setMinimumHeight(320)
        root.addWidget(mid_splitter)

        self.trend_chart = ChartCard(
            "历史趋势（双轴）",
            "左轴：病害数 / 重度数（整数） 右轴：受损面积 / 填补体积（浮点）",
            figsize=(12, 3),
        )
        self.trend_chart.setMinimumHeight(280)
        root.addWidget(self.trend_chart)

        table_card = QFrame()
        table_card.setObjectName("card")
        tc_layout = QVBoxLayout(table_card)
        tc_layout.setContentsMargins(14, 12, 14, 12)
        tc_layout.setSpacing(8)

        table_hdr = QHBoxLayout()
        table_title = QLabel("历史任务列表")
        table_title.setObjectName("sectionTitle")
        table_desc = QLabel("双击行跳转结果中心 · 点击列头排序")
        table_desc.setObjectName("sectionDesc")
        table_hdr.addWidget(table_title)
        table_hdr.addStretch(1)
        table_hdr.addWidget(table_desc)
        tc_layout.addLayout(table_hdr)

        self.table = QTableWidget(0, 10)
        self.table.setObjectName("historyTable")
        self.table.setHorizontalHeaderLabels([
            "时间", "任务名", "类型", "病害数", "重度", "中度", "轻度",
            "面积(m²)", "体积(m³)", "深度(cm)",
        ])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for col in range(2, 10):
            hh.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(False)
        self.table.setSortingEnabled(True)
        self.table.setMinimumHeight(200)
        tc_layout.addWidget(self.table, 1)
        table_card.setMinimumHeight(280)
        root.addWidget(table_card)

        self.refresh_button.clicked.connect(self.refresh_data)
        self.open_output_button.clicked.connect(self._open_output_dir)
        self.open_selected_button.clicked.connect(self._open_selected_result)
        self.table.itemDoubleClicked.connect(lambda *_: self._open_selected_result())
        self.table.itemSelectionChanged.connect(self._sync_selected_button)

        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet("""
            QWidget#scrollInner { background: transparent; }
            QLabel#pageTitle {
                font-size: 22px; font-weight: 800; color: #f0f8ff;
            }
            QLabel#pageSubtitle {
                font-size: 12px; color: #7a96b8;
            }
            QFrame#toolbar, QFrame#card, QFrame#chartCard {
                background-color: #0b1624;
                border: 1px solid #173150;
                border-radius: 16px;
            }
            QLabel#sectionTitle, QLabel#chartTitle {
                font-size: 15px; font-weight: 800; color: #e8f3ff;
            }
            QLabel#sectionDesc, QLabel#chartSubtitle {
                font-size: 11px; color: #6a8aac;
            }
            QPushButton {
                min-height: 36px;
                border-radius: 10px;
                background-color: #102740;
                color: #eaf3ff;
                border: 1px solid #24476d;
                padding: 0 14px;
                font-size: 13px;
                font-weight: 700;
            }
            QPushButton:hover { background-color: #153557; }
            QPushButton:disabled {
                color: #6d86a3;
                background-color: #0b1724;
                border-color: #1a2d44;
            }
            QTableWidget#historyTable {
                background-color: #070f1c;
                color: #dbeafe;
                border: none;
                gridline-color: #132438;
                font-size: 12px;
                border-radius: 10px;
            }
            QHeaderView::section {
                background-color: #0e2236;
                color: #a8c8f0;
                border: none;
                border-bottom: 1px solid #173150;
                padding: 6px 8px;
                font-weight: 700;
                font-size: 12px;
            }
            QTableWidget#historyTable::item { padding: 5px 8px; }
            QTableWidget#historyTable::item:selected {
                background-color: #143556;
                color: #ffffff;
            }
            QTextEdit#chartFallback {
                background-color: #0d1a2c;
                color: #dbeafe;
                border: 1px solid #18314f;
                border-radius: 10px;
                padding: 8px;
                font-size: 12px;
            }
            QSplitter::handle { background-color: #07101c; border-radius: 3px; }
            QSplitter::handle:hover { background-color: #173150; }
        """)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_data()

    def _get_output_dir(self) -> Path:
        if self.config_service:
            cfg = self.config_service.load()
            return Path(cfg.get("last_output_dir") or "runs/predict")
        return Path("runs/predict")

    def _open_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output_dir.resolve())))

    def _sync_selected_button(self):
        self.open_selected_button.setEnabled(self.table.currentRow() >= 0)

    def _open_selected_result(self):
        row = self.table.currentRow()
        if row < 0:
            return
        task_id_item = self.table.item(row, 0)
        if not task_id_item:
            return
        task_id = task_id_item.data(Qt.UserRole)
        if task_id and self.open_result_file:
            self.open_result_file(task_id)

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

    def _collect_entries_from_db(self) -> List[Dict[str, Any]]:
        """从数据库读取所有任务数据"""
        try:
            tasks = self.stats_service.get_recent_tasks(limit=1000)
            entries = []
            for task in tasks:
                entry = {
                    "task_id": task["id"],
                    "task_name": task["task_name"],
                    "filename": task["source_path"],
                    "task_type": task["task_type"],
                    "time": task["created_at"],
                    "count": self._safe_int(task["total_detections"], 0),
                    "severe": self._safe_int(task["severe_count"], 0),
                    "area": self._safe_float(task["total_area_m2"], 0.0),
                    "volume": self._safe_float(task["total_volume_m3"], 0.0),
                    "depth": self._safe_float(task["avg_depth_cm"], 0.0),
                    "severity": {
                        "轻度": self._safe_int(task["mild_count"], 0),
                        "中度": self._safe_int(task["moderate_count"], 0),
                        "重度": self._safe_int(task["severe_count"], 0),
                    },
                    "source": task["source_path"],
                }
                entries.append(entry)
            return entries
        except Exception as e:
            print(f"从数据库读取数据失败: {e}")
            return []

    def refresh_data(self):
        """从数据库刷新数据"""
        self.output_dir = self._get_output_dir()
        self.entries = self._collect_entries_from_db()

        if not self.entries:
            print("数据库中暂无数据，请先执行检测任务")
            self.card_total_tasks.set_value("0")
            self.card_total_count.set_value("0")
            self.card_total_severe.set_value("0")
            self.card_total_area.set_value("0 m²")
            self.card_total_volume.set_value("0 m³")
            self.card_max_depth.set_value("0 cm")
            self.type_chart.clear("暂无历史任务")
            self.scatter_chart.clear("暂无面积-体积数据")
            self.radar_chart.clear("暂无风险雷达数据")
            self.trend_chart.clear("暂无趋势数据")
            self.table.setRowCount(0)
            return

        total_count = sum(e["count"] for e in self.entries)
        total_severe = sum(e["severe"] for e in self.entries)
        total_area = sum(e["area"] for e in self.entries)
        total_volume = sum(e["volume"] for e in self.entries)

        depth_vals = [e["depth"] for e in self.entries if e["depth"] > 0]
        avg_depth = sum(depth_vals) / len(depth_vals) if depth_vals else 0.0

        self.card_total_tasks.set_value(str(len(self.entries)))
        self.card_total_tasks.set_note("从数据库读取的任务数")
        self.card_total_count.set_value(str(total_count))
        self.card_total_count.set_note("所有历史任务 count 之和")
        self.card_total_severe.set_value(str(total_severe))
        self.card_total_severe.set_note("所有历史任务 severe 之和")
        self.card_total_area.set_value(f"{total_area:.3f} m²")
        self.card_total_area.set_note("所有历史任务受损面积累计")
        self.card_total_volume.set_value(f"{total_volume:.4f} m³")
        self.card_total_volume.set_note("所有历史任务填补体积累计")
        self.card_max_depth.set_value(f"{avg_depth:.2f} cm")
        self.card_max_depth.set_note("所有任务平均深度")

        self._render_type_chart()
        self._render_scatter_chart()
        self._render_radar_chart()
        self._render_trend_chart()
        self._populate_table()
        self._sync_selected_button()

    def _render_type_chart(self):
        if not self.entries:
            self.type_chart.clear("暂无历史任务")
            return
        image_count = sum(1 for e in self.entries if e["task_type"] == "单图")
        video_count = sum(1 for e in self.entries if e["task_type"] == "视频流")
        values = [image_count, video_count]
        if HAVE_MPL:
            self.type_chart.figure.clear()
            ax = self.type_chart.figure.add_subplot(111)
            ax.set_facecolor("#0b1624")
            wedges, texts, autotexts = ax.pie(
                values,
                labels=["图片任务", "视频任务"],
                colors=["#22c55e", "#a78bfa"],
                startangle=90,
                wedgeprops={"width": 0.50, "edgecolor": "#0b1624", "linewidth": 1.5},
                textprops={"color": "#c8dff5", "fontsize": 9},
                autopct="%1.0f%%",
                pctdistance=0.75,
            )
            for at in autotexts:
                at.set_color("#f0f8ff")
                at.set_fontsize(9)
                at.set_fontweight("bold")
            ax.text(0, 0.08, "历史\n任务", ha="center", va="center",
                    color="#e8f3ff", fontsize=11, fontweight="bold")
            ax.text(0, -0.18, str(len(self.entries)), ha="center", va="center",
                    color="#7fd0ff", fontsize=20, fontweight="bold")
            self.type_chart.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            self.type_chart.canvas.draw_idle()
        else:
            self.type_chart.canvas.setText(f"图片任务：{image_count}\n视频任务：{video_count}")

    def _render_scatter_chart(self):
        if not self.entries:
            self.scatter_chart.clear("暂无面积-体积数据")
            return
        if HAVE_MPL:
            self.scatter_chart.figure.clear()
            ax = self.scatter_chart.figure.add_subplot(111)
            ax.set_facecolor("#0b1624")
            for e in self.entries:
                color = "#fb7185" if e["severe"] > 0 else "#38bdf8"
                size = max(30, e["count"] * 8)
                ax.scatter(e["area"], e["volume"], c=color, s=size,
                           alpha=0.82, edgecolors="#0b1624", linewidth=0.5, zorder=3)
            top3 = sorted(self.entries, key=lambda x: x["area"], reverse=True)[:3]
            for e in top3:
                ax.annotate(
                    e["task_name"][:8], (e["area"], e["volume"]),
                    fontsize=7, color="#c8dff5",
                    xytext=(4, 4), textcoords="offset points",
                )
            ax.set_xlabel("受损面积 (m²)", color="#7a96b8", fontsize=8)
            ax.set_ylabel("填补体积 (m³)", color="#7a96b8", fontsize=8)
            ax.tick_params(colors="#7a96b8", labelsize=7)
            ax.grid(color="#132438", linestyle="--", alpha=0.5, zorder=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#fb7185",
                       markersize=7, label="含重度"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#38bdf8",
                       markersize=7, label="无重度"),
            ]
            leg = ax.legend(handles=legend_elements, loc="upper left",
                            frameon=False, fontsize=8)
            for t in leg.get_texts():
                t.set_color("#c8dff5")
            self.scatter_chart.figure.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.14)
            self.scatter_chart.canvas.draw_idle()
        else:
            lines = [f"{e['task_name']}: 面积={e['area']:.4f} 体积={e['volume']:.4f}"
                     for e in self.entries]
            self.scatter_chart.canvas.setText("\n".join(lines))

    def _render_radar_chart(self):
        if not self.entries:
            self.radar_chart.clear("暂无风险雷达数据")
            return
        try:
            import numpy as np
        except ImportError:
            self.radar_chart.canvas.setText("需要 numpy 才能显示雷达图")
            return
        recent = self.entries[-5:]
        categories = ["轻度", "中度", "重度"]
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        if HAVE_MPL:
            self.radar_chart.figure.clear()
            ax = self.radar_chart.figure.add_subplot(111, polar=True)
            ax.set_facecolor("#0b1624")
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids(
                np.degrees(angles[:-1]), categories,
                color="#c8dff5", fontsize=9,
            )
            ax.spines["polar"].set_color("#173150")
            ax.tick_params(colors="#6a8aac", labelsize=7)
            ax.yaxis.grid(color="#173150", linestyle="--", alpha=0.5)
            ax.xaxis.grid(color="#173150", linestyle="--", alpha=0.5)

            palette = ["#38bdf8", "#22c55e", "#f59e0b", "#fb7185", "#a78bfa"]
            for i, e in enumerate(recent):
                vals = [e["severity"].get(k, 0) for k in categories]
                vals += vals[:1]
                color = palette[i % len(palette)]
                ax.plot(angles, vals, color=color, linewidth=1.8,
                        label=e["task_name"][:10])
                ax.fill(angles, vals, color=color, alpha=0.10)

            leg = ax.legend(
                loc="upper right", bbox_to_anchor=(1.35, 1.15),
                frameon=False, fontsize=8,
            )
            for t in leg.get_texts():
                t.set_color("#c8dff5")
            self.radar_chart.figure.subplots_adjust(left=0.05, right=0.80, top=0.90, bottom=0.05)
            self.radar_chart.canvas.draw_idle()
        else:
            lines = []
            for e in recent:
                s = e["severity"]
                lines.append(
                    f"{e['task_name']}: 轻={s.get('轻度', 0)} 中={s.get('中度', 0)} 重={s.get('重度', 0)}"
                )
            self.radar_chart.canvas.setText("\n".join(lines))

    def _render_trend_chart(self):
        if not self.entries:
            self.trend_chart.clear("暂无趋势数据")
            return
        recent = self.entries[-10:]
        labels = [e["task_name"][:10] for e in recent]
        count_vals = [e["count"] for e in recent]
        severe_vals = [e["severe"] for e in recent]
        area_vals = [e["area"] for e in recent]
        volume_vals = [e["volume"] for e in recent]

        if HAVE_MPL:
            self.trend_chart.figure.clear()
            ax1 = self.trend_chart.figure.add_subplot(111)
            ax1.set_facecolor("#0b1624")
            ax2 = ax1.twinx()

            if len(recent) == 1:
                ax1.scatter([0], count_vals, color="#38bdf8", s=60, zorder=5, label="病害数")
                ax1.scatter([0], severe_vals, color="#fb7185", s=60, zorder=5,
                            marker="s", label="重度数")
                ax2.scatter([0], area_vals, color="#22c55e", s=60, zorder=5,
                            marker="^", label="面积(m²)")
                ax2.scatter([0], volume_vals, color="#f59e0b", s=60, zorder=5,
                            marker="D", label="体积(m³)")
                ax1.set_xticks([0])
                ax1.set_xticklabels(labels, color="#c8dff5", fontsize=9)
            else:
                l1, = ax1.plot(labels, count_vals, "o-", color="#38bdf8", linewidth=2,
                               label="病害数", markersize=5)
                l2, = ax1.plot(labels, severe_vals, "s-", color="#fb7185", linewidth=2,
                               label="重度数", markersize=5)
                l3, = ax2.plot(labels, area_vals, "^--", color="#22c55e", linewidth=2,
                               label="面积(m²)", markersize=5)
                l4, = ax2.plot(labels, volume_vals, "D--", color="#f59e0b", linewidth=2,
                               label="体积(m³)", markersize=5)
                ax1.tick_params(axis="x", colors="#c8dff5", labelsize=8, rotation=25)
                self.trend_chart.figure.legend(
                    [l1, l2, l3, l4],
                    ["病害数", "重度数", "面积(m²)", "体积(m³)"],
                    loc="upper right", bbox_to_anchor=(0.99, 0.98),
                    frameon=True, fontsize=8,
                    facecolor="#0d1a2c", edgecolor="#173150",
                    labelcolor=["#38bdf8", "#fb7185", "#22c55e", "#f59e0b"],
                )

            ax1.set_ylabel("目标数量", color="#7ecfff", fontsize=9)
            ax2.set_ylabel("面积 / 体积", color="#a3e8a3", fontsize=9)
            ax1.tick_params(axis="y", colors="#7ecfff", labelsize=8)
            ax2.tick_params(axis="y", colors="#a3e8a3", labelsize=8)
            ax1.grid(axis="y", color="#132438", linestyle="--", alpha=0.6)
            for spine in ax1.spines.values():
                spine.set_visible(False)
            for spine in ax2.spines.values():
                spine.set_visible(False)
            self.trend_chart.figure.subplots_adjust(left=0.10, right=0.90, top=0.88, bottom=0.15)
            self.trend_chart.canvas.draw_idle()
        else:
            lines = [
                f"{e['task_name']}: 病害={e['count']} 面积={e['area']:.4f}"
                for e in recent
            ]
            self.trend_chart.canvas.setText("\n".join(lines))

    def _populate_table(self):
        rows = sorted(self.entries, key=lambda x: x["time"], reverse=True)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for row, entry in enumerate(rows):
            sev = entry["severity"]
            time_str = entry["time"].strftime("%Y-%m-%d %H:%M") if hasattr(entry["time"], 'strftime') else str(
                entry["time"])
            col_data = [
                (time_str, None),
                (entry["task_name"], None),
                (entry["task_type"], None),
                (entry["count"], entry["count"]),
                (entry["severe"], entry["severe"]),
                (sev.get("中度", 0), sev.get("中度", 0)),
                (sev.get("轻度", 0), sev.get("轻度", 0)),
                (f"{entry['area']:.4f}", entry["area"]),
                (f"{entry['volume']:.4f}", entry["volume"]),
                (f"{entry['depth']:.2f}", entry["depth"]),
            ]
            is_severe = entry["severe"] > 0
            for col, (display, sort_val) in enumerate(col_data):
                item = QTableWidgetItem(str(display))
                if col == 0:
                    item.setData(Qt.UserRole, entry.get("task_id"))
                if sort_val is not None:
                    item.setData(Qt.DisplayRole, sort_val)
                if is_severe:
                    item.setBackground(QColor("#2a0d10"))
                    if col == 4:
                        item.setForeground(QColor("#fb7185"))
                self.table.setItem(row, col, item)
        self.table.setSortingEnabled(True)
