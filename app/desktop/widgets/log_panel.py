from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QProgressBar, QSizePolicy


class LogPanel(QFrame):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        top = QHBoxLayout()
        title = QLabel('任务日志')
        title.setObjectName('sectionTitle')
        self.status_chip = QLabel('空闲')
        self.status_chip.setMinimumWidth(84)
        self.clear_button = QPushButton('清空日志')
        self.clear_button.setMinimumWidth(110)
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(self.status_chip)
        top.addWidget(self.clear_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setMinimumHeight(12)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout.addLayout(top)
        layout.addWidget(self.progress)
        layout.addWidget(self.text_edit, 1)

        self.clear_button.clicked.connect(self.text_edit.clear)
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet(
            """
            QFrame { background-color: #0b1624; border: 1px solid #173150; border-radius: 18px; }
            QLabel#sectionTitle { color: #eaf2ff; font-size: 18px; font-weight: 800; }
            QTextEdit { background-color: #08111d; color: #dbeafe; border-radius: 12px; border: 1px solid #17314d; font-size: 12px; }
            QPushButton { min-height: 34px; border-radius: 10px; background-color: #102740; color: #e2edff; border: 1px solid #23476d; font-weight: 700; }
            QPushButton:hover { background-color: #163557; }
            QProgressBar { background: #08111d; border: 1px solid #17314d; border-radius: 8px; min-height: 12px; }
            QProgressBar::chunk { border-radius: 7px; background: #22c55e; }
            """
        )
        self._set_chip('#334155', '#dbeafe')

    def _set_chip(self, bg: str, fg: str):
        self.status_chip.setStyleSheet(
            f'background-color: {bg}; color: {fg}; border-radius: 10px; padding: 7px 12px; font-weight: 700; font-size: 12px;'
        )

    def set_running(self):
        self._set_chip('#123052', '#7dd3fc')
        self.status_chip.setText('运行中')
        self.progress.setRange(0, 0)

    def set_idle(self):
        self._set_chip('#334155', '#dbeafe')
        self.status_chip.setText('空闲')
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

    def set_finished(self):
        self._set_chip('#164e3b', '#86efac')
        self.status_chip.setText('完成')
        self.progress.setRange(0, 100)
        self.progress.setValue(100)

    def set_failed(self):
        self._set_chip('#4c0519', '#fda4af')
        self.status_chip.setText('失败')
        self.progress.setRange(0, 100)
        self.progress.setValue(100)

    def append(self, text: str) -> None:
        self.text_edit.append(text)
