"""Central configuration for the Bosch Production Line Performance project.

Paths default to the repo layout but can be overridden with environment
variables so the same code runs unchanged on Kaggle / Colab:

    BOSCH_DATA_DIR       e.g. /kaggle/input/bosch-production-line-performance
    BOSCH_ARTIFACTS_DIR  e.g. /kaggle/working/artifacts
    BOSCH_CACHE_DIR      where engineered-feature parquet files are written
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# --- Paths ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("BOSCH_DATA_DIR", REPO_ROOT / "data-set"))
ARTIFACTS_DIR = Path(os.environ.get("BOSCH_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
CACHE_DIR = Path(os.environ.get("BOSCH_CACHE_DIR", REPO_ROOT / "cache"))

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Raw competition files (train_* are labelled; test_* are unlabelled and untouched).
TRAIN_NUMERIC = DATA_DIR / "train_numeric.csv"
TRAIN_DATE = DATA_DIR / "train_date.csv"
TRAIN_CATEGORICAL = DATA_DIR / "train_categorical.csv"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"

# Artifact files consumed by the FastAPI service.
MODEL_PATH = ARTIFACTS_DIR / "model.txt"          # LightGBM Booster text format
MODEL_PKL_PATH = ARTIFACTS_DIR / "model.pkl"      # fallback: any joblib-pickled estimator
FEATURE_LIST_PATH = ARTIFACTS_DIR / "feature_list.json"   # ordered engineered features
RAW_COLUMNS_PATH = ARTIFACTS_DIR / "raw_columns.json"     # canonical raw columns per block
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# --- Schema constants -------------------------------------------------------
ID_COL = "Id"
TARGET_COL = "Response"
RANDOM_STATE = 42

# --- Column-name parsing ----------------------------------------------------
# Bosch columns look like:  L0_S0_F0  (numeric/categorical feature)
#                           L0_S0_D1  (date feature)
# i.e. L{line}_S{station}_{F|D}{index}
_COL_RE = re.compile(r"^L(?P<line>\d+)_S(?P<station>\d+)_[FD](?P<index>\d+)$")


def parse_column(col: str) -> dict | None:
    """Return {'line': int, 'station': int, 'index': int} for a feature column.

    Returns None for non-feature columns such as 'Id' / 'Response'.
    """
    m = _COL_RE.match(col)
    if not m:
        return None
    return {
        "line": int(m.group("line")),
        "station": int(m.group("station")),
        "index": int(m.group("index")),
    }


def station_of(col: str) -> str | None:
    """'L0_S0_F0' -> 'S0'. None for non-feature columns."""
    p = parse_column(col)
    return f"S{p['station']}" if p else None


def line_of(col: str) -> str | None:
    """'L0_S0_F0' -> 'L0'. None for non-feature columns."""
    p = parse_column(col)
    return f"L{p['line']}" if p else None


def group_columns_by(cols, key="station") -> dict[str, list[str]]:
    """Group feature columns by 'station' or 'line'. Non-feature cols are skipped.

    Returns an insertion-ordered dict, e.g. {'S0': [...], 'S1': [...]}.
    """
    fn = station_of if key == "station" else line_of
    groups: dict[str, list[str]] = {}
    for c in cols:
        g = fn(c)
        if g is not None:
            groups.setdefault(g, []).append(c)
    return groups
