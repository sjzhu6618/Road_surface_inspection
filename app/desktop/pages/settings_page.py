from pathlib import Path
import sys
from PySide6.QtWidgets import QCheckBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout


class SettingsPage(QFrame):
    def __init__(self, config_service):
        super().__init__()
        self.config_service = config_service
        cfg = self.config_service.load()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)
        title = QLabel('系统设置')
        title.setStyleSheet('font-size: 22px; font-weight: 700; color: white;')
        desc = QLabel('统一维护默认输出目录、Python 解释器路径、主题偏好与最近一次任务配置。')
        desc.setStyleSheet('font-size: 13px; color: #9fb6cf;')
        root.addWidget(title)
        root.addWidget(desc)
        form_wrap = QFrame()
        form_wrap.setStyleSheet('background-color: #0d1a2c; border: 1px solid #18314f; border-radius: 14px;')
        form = QFormLayout(form_wrap)
        form.setContentsMargins(16, 16, 16, 16)
        form.setVerticalSpacing(12)
        self.output_edit = QLineEdit(cfg.get('last_output_dir', 'runs/predict'))
        self.theme_check = QCheckBox('启用深色主题')
        self.theme_check.setChecked(cfg.get('theme', 'dark') == 'dark')
        self.auto_save_check = QCheckBox('保存最近一次任务配置')
        self.auto_save_check.setChecked(True)

        # Python 解释器路径（打包后使用）
        py_row = QHBoxLayout()
        py_row.setSpacing(8)
        self.python_edit = QLineEdit(cfg.get('python_executable', '') or sys.executable)
        self.python_edit.setStyleSheet('min-height: 32px; background-color: #08111d; color: #e2e8f0; border: 1px solid #244466; border-radius: 8px; padding: 0 8px;')
        py_browse = QPushButton('浏览')
        py_browse.setStyleSheet('min-height: 32px; min-width: 60px; border-radius: 8px; background-color: #123052; color: white; border: 1px solid #254a72;')
        py_browse.clicked.connect(self._choose_python)
        py_row.addWidget(self.python_edit, 1)
        py_row.addWidget(py_browse)

        form.addRow('默认输出目录', self.output_edit)
        form.addRow('Python 解释器', py_row)
        form.addRow('', self.theme_check)
        form.addRow('', self.auto_save_check)
        root.addWidget(form_wrap)
        self.save_button = QPushButton('保存设置')
        self.save_button.setStyleSheet('min-height: 36px; border-radius: 10px; background-color: #123052; color: white;')
        self.save_button.clicked.connect(self._save)
        root.addWidget(self.save_button)
        root.addStretch(1)

    def _choose_python(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择 Python 解释器', '', '可执行文件 (*.exe);;所有文件 (*)')
        if path:
            self.python_edit.setText(path)

    def _save(self) -> None:
        cfg = self.config_service.load()
        cfg['last_output_dir'] = self.output_edit.text().strip() or 'runs/predict'
        cfg['theme'] = 'dark' if self.theme_check.isChecked() else 'light'
        cfg['python_executable'] = self.python_edit.text().strip()
        self.config_service.save(cfg)
