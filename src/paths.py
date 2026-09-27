"""Project paths; large and temporary files default to drive E."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ASG06_DATA_DIR", str(ROOT.parent / "ASG_06_data")))
CACHE = DATA / "cache"
for folder in [CACHE / "tmp", ROOT / "results", ROOT / "logs"]:
    folder.mkdir(parents=True, exist_ok=True)
for key in ["TMP", "TEMP", "TMPDIR"]:
    os.environ[key] = str(CACHE / "tmp")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("PIP_CACHE_DIR", str(CACHE / "pip"))
os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
os.environ.setdefault("KERAS_HOME", str(CACHE / "keras"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
