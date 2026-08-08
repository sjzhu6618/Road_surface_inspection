from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout


class InfoCard(QFrame):
    def __init__(self, title: str, value: str = '--', accent: str = '#38bdf8', note: str = ''):
        super().__init__()
        self._accent = accent
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        self.title_label = QLabel(title)
        self.title_label.setObjectName('cardTitle')
        self.title_label.setWordWrap(True)
        self.value_label = QLabel(value)
        self.value_label.setObjectName('cardValue')
        self.note_label = QLabel(note)
        self.note_label.setObjectName('cardNote')
        self.note_label.setWordWrap(True)
        self.note_label.setVisible(bool(note))

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.note_label)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.setMinimumHeight(104)
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: #0d1928;
                border: 1px solid #1a324d;
                border-left: 4px solid {self._accent};
                border-radius: 16px;
            }}
            QLabel#cardTitle {{ color: #93a8c2; font-size: 12px; font-weight: 600; }}
            QLabel#cardValue {{ color: #ecf5ff; font-size: 18px; font-weight: 800; }}
            QLabel#cardNote {{ color: #7dd3fc; font-size: 11px; }}
            """
        )

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)

    def set_note(self, note: str) -> None:
        self.note_label.setVisible(bool(note))
        self.note_label.setText(note)
