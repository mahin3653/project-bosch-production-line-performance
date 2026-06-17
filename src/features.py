"""Shared feature engineering for the Bosch project.

This is the single source of truth for transforms. The training notebook and
the FastAPI service both call `build_features(...)`, so a part scored in
production goes through the *identical* pipeline used at training time.

All inputs/outputs are DataFrames indexed by `Id`. A "block" is one of the raw
files restricted to feature columns:
    numeric / date  -> L{line}_S{station}_F|D{idx}, float
    categorical     -> L{line}_S{station}_F{idx}, string (optional)

Parity contract
---------------
At training time the blocks contain every canonical column (sparse rows have
NaN where a part skipped a station). At inference, reindex the incoming sparse
record to those same canonical columns (see `reindex_to_canonical`) *before*
calling build_features, then `align_to_feature_list` to lock column order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --- helpers ----------------------------------------------------------------
def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Columns that parse as Bosch features (exclude Id/Response if present)."""
    return [c for c in df.columns if config.parse_column(c) is not None]


def reindex_to_canonical(df: pd.DataFrame, canonical_cols: list[str]) -> pd.DataFrame:
    """Ensure `df` has exactly `canonical_cols` (missing -> NaN), preserving index.

    Used at inference so a sparse incoming record is expanded to the full column
    universe seen during training before features are built.
    """
    return df.reindex(columns=canonical_cols)


# --- feature blocks ---------------------------------------------------------
def date_features(date_df: pd.DataFrame) -> pd.DataFrame:
    """Production-flow timing features from the date block."""
    cols = _feature_cols(date_df)
    d = date_df[cols]
    start, end = d.min(axis=1), d.max(axis=1)
    data = {
        "date_start": start,
        "date_end": end,
        "date_duration": end - start,
        "date_recorded_count": d.notna().sum(axis=1),  # how many timestamps recorded
    }
    # per-line timing
    for line, line_cols in config.group_columns_by(cols, key="line").items():
        ls, le = d[line_cols].min(axis=1), d[line_cols].max(axis=1)
        data[f"{line}_date_start"] = ls
        data[f"{line}_date_end"] = le
        data[f"{line}_date_duration"] = le - ls
    return pd.DataFrame(data, index=date_df.index)


def station_numeric_features(num_df: pd.DataFrame) -> pd.DataFrame:
    """Per-station aggregations of numeric measurements (mean/std/min/max/count)."""
    cols = _feature_cols(num_df)
    data = {}
    for station, st_cols in config.group_columns_by(cols, key="station").items():
        sub = num_df[st_cols]
        data[f"{station}_num_mean"] = sub.mean(axis=1)
        data[f"{station}_num_std"] = sub.std(axis=1)
        data[f"{station}_num_min"] = sub.min(axis=1)
        data[f"{station}_num_max"] = sub.max(axis=1)
        data[f"{station}_num_count"] = sub.notna().sum(axis=1)
    return pd.DataFrame(data, index=num_df.index)


def missingness_features(num_df: pd.DataFrame, date_df: pd.DataFrame) -> pd.DataFrame:
    """Sparsity / station-visit features. NaN is interpreted, not dropped."""
    num_cols = _feature_cols(num_df)
    data = {"numeric_nonnull_total": num_df[num_cols].notna().sum(axis=1)}

    # per-line numeric non-null counts
    for line, line_cols in config.group_columns_by(num_cols, key="line").items():
        data[f"{line}_numeric_nonnull"] = num_df[line_cols].notna().sum(axis=1)

    # station-visit flags derived from the date block (a station is "visited"
    # if it recorded any timestamp)
    date_cols = _feature_cols(date_df)
    for station, st_cols in config.group_columns_by(date_cols, key="station").items():
        data[f"{station}_visited"] = date_df[st_cols].notna().any(axis=1).astype(np.int8)
    return pd.DataFrame(data, index=num_df.index)


def path_features(date_df: pd.DataFrame) -> pd.DataFrame:
    """The route a part takes through the stations (from recorded timestamps)."""
    date_cols = _feature_cols(date_df)
    d = date_df[date_cols]
    station_groups = config.group_columns_by(date_cols, key="station")
    visited = pd.DataFrame(
        {st: d[st_cols].notna().any(axis=1) for st, st_cols in station_groups.items()},
        index=date_df.index,
    )

    # deterministic fingerprint of the *set* of visited stations:
    #   path_id = sum over visited stations of 2**station_index  (exact in float64)
    powers = np.array([2 ** int(s[1:]) for s in visited.columns], dtype=np.float64)
    data = {
        "path_length": visited.sum(axis=1).astype(np.int16),
        "path_id": visited.to_numpy(dtype=np.float64) @ powers,
    }

    # first / last station by timestamp; guard rows with no timestamp at all
    # (idxmin/idxmax on an all-NA row is deprecated and would raise).
    first = pd.Series(np.nan, index=d.index)
    last = pd.Series(np.nan, index=d.index)
    has_any = d.notna().any(axis=1)
    if bool(has_any.any()):
        sub = d[has_any]
        to_st = lambda lab: lab.map(
            lambda c: config.parse_column(c)["station"] if isinstance(c, str) else np.nan)
        first.loc[has_any] = to_st(sub.idxmin(axis=1))
        last.loc[has_any] = to_st(sub.idxmax(axis=1))
    data["first_station"] = first
    data["last_station"] = last
    return pd.DataFrame(data, index=date_df.index)


def categorical_features(cat_df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight categorical sparsity features (optional block)."""
    cols = _feature_cols(cat_df)
    data = {"categorical_nonnull_total": cat_df[cols].notna().sum(axis=1)}
    for line, line_cols in config.group_columns_by(cols, key="line").items():
        data[f"{line}_categorical_nonnull"] = cat_df[line_cols].notna().sum(axis=1)
    return pd.DataFrame(data, index=cat_df.index)


def interaction_features(feat: pd.DataFrame) -> pd.DataFrame:
    """A few crosses between the strongest signals (duration, path, sparsity)."""
    data = {}
    dur, plen, nnn = feat.get("date_duration"), feat.get("path_length"), feat.get("numeric_nonnull_total")
    if dur is not None and plen is not None:
        data["duration_x_pathlen"] = dur * plen
        data["duration_per_station"] = dur / (plen.astype(np.float64) + 1.0)
    if nnn is not None and plen is not None:
        data["numeric_per_station"] = nnn / (plen.astype(np.float64) + 1.0)
    return pd.DataFrame(data, index=feat.index)


# --- top-level assembly -----------------------------------------------------
def build_features(
    numeric: pd.DataFrame,
    date: pd.DataFrame,
    categorical: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the full engineered feature table (indexed by Id).

    Blocks must already be restricted to the canonical column universe and
    share the same Id index. Drops the target column if present in `numeric`.
    """
    blocks = [
        date_features(date),
        station_numeric_features(numeric),
        missingness_features(numeric, date),
        path_features(date),
    ]
    if categorical is not None:
        blocks.append(categorical_features(categorical))

    feat = pd.concat(blocks, axis=1)
    feat = pd.concat([feat, interaction_features(feat)], axis=1)
    return feat


def align_to_feature_list(feat: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """Reorder/restrict engineered columns to exactly what the model expects.

    In the happy path (canonical inputs) this only reorders. Genuinely-absent
    columns are filled with NaN, which the tree models handle natively.
    """
    return feat.reindex(columns=feature_list)
