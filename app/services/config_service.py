import json
from pathlib import Path
from typing import Any, Dict


class AppConfigService:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.config_path.exists():
            self.save(self.default_config())

    def default_config(self) -> Dict[str, Any]:
        return {
            'window': {'width': 1480, 'height': 920},
            'last_model_path': '',
            'last_source_path': '',
            'last_output_dir': 'runs/predict',
            'theme': 'dark',
            'python_executable': '',
        }

    def load(self) -> Dict[str, Any]:
        try:
            return json.loads(self.config_path.read_text(encoding='utf-8'))
        except Exception:
            return self.default_config()

    def save(self, data: Dict[str, Any]) -> None:
        self.config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
