from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / 'configs'
OUTPUT_DIR = ROOT_DIR / 'runs' / 'predict'
PREDICT_SCRIPT = ROOT_DIR / 'predict.py'
