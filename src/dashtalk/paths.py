from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
WAREHOUSE = DATA / "dashtalk.duckdb"
SEMANTIC = ROOT / "semantic"
CONFIG = ROOT / "config"
LOGS = ROOT / "logs"
EVALS = ROOT / "evals"
RESULTS = ROOT / "results"
