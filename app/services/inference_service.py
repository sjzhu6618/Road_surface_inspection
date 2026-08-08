from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal

from app.core.paths import ROOT_DIR
from app.core.schemas import InferenceConfig
from app.services.command_builder import CommandBuilder
from app.services.result_service import ResultService
from app.workers.inference_worker import InferenceWorker


class InferenceService(QObject):
    log = Signal(str)
    status = Signal(str)
    finished = Signal(object)
    failed = Signal(str)
    started = Signal()

    def __init__(self, config_service=None, parent=None):
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[InferenceWorker] = None
        self._config_service = config_service

    def start_task(self, config: InferenceConfig) -> None:
        if self._thread and self._thread.isRunning():
            self.failed.emit('当前已有推理任务在运行，请先停止。')
            return

        python_exe = None
        if self._config_service:
            cfg = self._config_service.load()
            python_exe = cfg.get('python_executable') or None

        try:
            cmd = CommandBuilder.build(config, python_executable=python_exe)
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self._thread = QThread()
        self._worker = InferenceWorker(command=cmd, working_directory=str(ROOT_DIR))
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log.emit)
        self._worker.status.connect(self.status.emit)
        self._worker.finished.connect(lambda code: self._on_completed(code, config))
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self.started.emit()
        self._thread.start()

    def stop_task(self) -> None:
        if self._worker:
            self._worker.stop()
            self.status.emit('正在请求停止任务...')

    def _on_completed(self, return_code: int, config: InferenceConfig) -> None:
        if return_code == -2:
            self.status.emit('任务已停止，尝试读取已有结果...')
            try:
                result_service = ResultService(Path(config.output_dir))
                result_service.set_source_filename(config.source)
                summary = result_service.build_summary()
                self.finished.emit(summary)
            except Exception:
                self.finished.emit(None)
            return

        if return_code != 0:
            self.failed.emit(f'推理进程退出码异常: {return_code}')
            return

        result_service = ResultService(Path(config.output_dir))
        result_service.set_source_filename(config.source)
        summary = result_service.build_summary()
        self.finished.emit(summary)

    def _on_failed(self, message: str) -> None:
        self.failed.emit(message)

    def _cleanup(self) -> None:
        self._worker = None
        self._thread = None
