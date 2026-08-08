import sys
from pathlib import Path

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from app.desktop.main_window import MainWindow
from app.services.config_service import AppConfigService

from app.services.db_manager import init_database

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName('多任务感知路面损伤多维量化评估系统')
    app.setFont(QFont('Microsoft YaHei UI', 10))
    config_service = AppConfigService(Path.cwd() / 'configs' / 'app_config.json')

    init_database()

    window = MainWindow(config_service=config_service)
    window.resize(1500, 940)
    window.show()
    sys.exit(app.exec())
