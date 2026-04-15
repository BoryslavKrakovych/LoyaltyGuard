from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True, parents=True)

DEFAULT_REFERENCE_DATE = '2025-12-31'
LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.60
RANDOM_STATE = 42
