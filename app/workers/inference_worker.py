import subprocess
from typing import List, Optional

from PySide6.QtCore import QObject, Signal


class InferenceWorker(QObject):
    log = Signal(str)
    status = Signal(str)
    finished = Signal(int)
    failed = Signal(str)

    def __init__(self, command: List[str], working_directory: str):
        super().__init__()
        self.command = command
        self.working_directory = working_directory
        self._process: Optional[subprocess.Popen] = None
        self._stop_requested = False

    def run(self) -> None:
        try:
            self.status.emit('推理任务已启动')
            self.log.emit('执行命令：' + ' '.join(self.command))
            self._process = subprocess.Popen(
                self.command,
                cwd=self.working_directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='ignore',
            )
            assert self._process.stdout is not None
            for line in self._process.stdout:
                if self._stop_requested:
                    break
                line = line.rstrip('\n')
                if line:
                    self.log.emit(line)
            if self._stop_requested and self._process and self._process.poll() is None:
                self._process.kill()
                self.finished.emit(-2)
                return
            code = self._process.wait()
            self.status.emit('推理任务已结束')
            self.finished.emit(code)
        except Exception as exc:
            self.failed.emit(f'推理执行失败：{exc}')

    def stop(self) -> None:
        self._stop_requested = True
        if self._process and self._process.poll() is None:
            self._process.terminate()
