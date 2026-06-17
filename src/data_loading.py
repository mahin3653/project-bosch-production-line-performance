"""Memory-aware loading for the Bosch CSVs.

The three train files are wide (968 / 1156 / 2140 feature columns) and tall
(~1.18M rows, ~15 GB combined). Two ideas keep this tractable:

  * **float32 downcasting** for numeric and date files (values are small).
  * **Id-filtered chunked reads** so we can pull an arbitrary subset of rows
    (e.g. all positives + a slice of negatives) without holding a whole file
    in memory.

Engineered features are cached to parquet via save_parquet / load_parquet so a
notebook re-run does not reprocess 15 GB.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# --- Lightweight scans ------------------------------------------------------
def read_ids_and_target(path: Path | str = config.TRAIN_NUMERIC) -> pd.DataFrame:
    """Read just Id + Response from train_numeric (the only file with the label)."""
    return pd.read_csv(path, usecols=[config.ID_COL, config.TARGET_COL])


def header_columns(path: Path | str) -> list[str]:
    """Column names of a CSV without reading any data rows."""
    return pd.read_csv(path, nrows=0).columns.tolist()


# --- Sampling ---------------------------------------------------------------
def sample_ids(
    id_target: pd.DataFrame,
    frac: float = 0.1,
    keep_all_positives: bool = True,
    random_state: int = config.RANDOM_STATE,
) -> np.ndarray:
    """Return a sorted array of Ids to load.

    With keep_all_positives (default) every failure row is kept and `frac` of
    the negatives is sampled — useful for EDA / prototyping on a class so rare
    that a plain random sample would contain almost no positives.
    """
    pos = id_target[id_target[config.TARGET_COL] == 1]
    neg = id_target[id_target[config.TARGET_COL] == 0]
    neg_s = neg.sample(frac=frac, random_state=random_state)
    keep = pd.concat([pos, neg_s]) if keep_all_positives else id_target.sample(
        frac=frac, random_state=random_state
    )
    return np.sort(keep[config.ID_COL].to_numpy())


# --- Filtered / downcast reads ----------------------------------------------
def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    return df


def load_by_ids(
    path: Path | str,
    ids: np.ndarray | set | None = None,
    usecols: list[str] | None = None,
    numeric: bool = True,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Load rows of a Bosch CSV, optionally restricted to a set of Ids.

    numeric=True downcasts float64 -> float32 (numeric & date files).
    numeric=False keeps values as-is (categorical file -> object/category).
    Pass ids=None to read the full file (chunked + downcast).
    """
    id_set = None if ids is None else set(np.asarray(ids).tolist())
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        if id_set is not None:
            chunk = chunk[chunk[config.ID_COL].isin(id_set)]
        if chunk.empty:
            continue
        if numeric:
            chunk = _downcast_numeric(chunk)
        parts.append(chunk)
    if not parts:
        return pd.read_csv(path, nrows=0)
    out = pd.concat(parts, ignore_index=True)
    return out


def load_all_blocks(
    ids: np.ndarray | None = None,
    include_categorical: bool = False,
    chunksize: int = 100_000,
) -> dict[str, pd.DataFrame]:
    """Load numeric (+ Response), date, and optionally categorical blocks for `ids`.

    Returns {'numeric': df, 'date': df, 'categorical': df?} each indexed by Id.
    """
    blocks = {
        "numeric": load_by_ids(config.TRAIN_NUMERIC, ids, numeric=True, chunksize=chunksize),
        "date": load_by_ids(config.TRAIN_DATE, ids, numeric=True, chunksize=chunksize),
    }
    if include_categorical:
        cat = load_by_ids(config.TRAIN_CATEGORICAL, ids, numeric=False, chunksize=chunksize)
        # categorical values are sparse strings; store as 'category' to save memory
        for c in cat.columns:
            if c != config.ID_COL:
                cat[c] = cat[c].astype("category")
        blocks["categorical"] = cat
    for df in blocks.values():
        df.set_index(config.ID_COL, inplace=True)
    return blocks


# --- Parquet cache ----------------------------------------------------------
def save_parquet(df: pd.DataFrame, name: str) -> Path:
    """Persist an (engineered) DataFrame to the cache dir as parquet."""
    path = config.CACHE_DIR / f"{name}.parquet"
    df.to_parquet(path)
    return path


def load_parquet(name: str) -> pd.DataFrame | None:
    """Load a cached parquet by name, or None if it does not exist."""
    path = config.CACHE_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def memory_mb(df: pd.DataFrame) -> float:
    """Deep memory footprint of a DataFrame in MB (for the 'reduce memory' step)."""
    return df.memory_usage(deep=True).sum() / 1024 ** 2
