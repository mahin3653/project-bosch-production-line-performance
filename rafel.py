"""Bosch Production Line Performance — single-file pipeline (train + serve).

Predict whether a manufactured part fails end-of-line quality control
(`Response` = 1) under extreme class imbalance (<1% failures). Everything lives
in this one module: configuration, memory-aware data loading, feature
engineering, model training (Phases I+II), and the FastAPI service (Phase III).

Train/serve parity is automatic: both paths call the same in-file
`build_features`, so a part scored in production goes through the identical
pipeline used at training.

Usage
-----
    # Phase I + II — train and write artifacts/
    python rafel.py train                                   # cloud-sized defaults
    python rafel.py train --sample-frac 0.1 --no-categorical   # low-RAM / local
    python rafel.py train --sample-frac 0.05 --no-categorical --models lgbm

    # Phase III — serve predictions
    python rafel.py serve                  # or: uvicorn bosch:app
    #   GET  /health
    #   POST /predict  {"id": 1, "features": {"L0_S0_F0": 0.03, ...}}

Paths can be overridden with env vars so the same file runs on Kaggle / Colab:
    BOSCH_DATA_DIR  BOSCH_ARTIFACTS_DIR  BOSCH_CACHE_DIR
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================================
# Configuration & column parsing
# =====================================================================
REPO_ROOT = Path(__file__).resolve().parent


def _detect_data_dir() -> Path:
    """Honor BOSCH_DATA_DIR, else auto-detect Kaggle's input dir, else ./data-set."""
    env = os.environ.get("BOSCH_DATA_DIR")
    if env:
        return Path(env)
    for cand in ["/kaggle/input/bosch-production-line-performance", REPO_ROOT / "data-set"]:
        if Path(cand).exists():
            return Path(cand)
    return REPO_ROOT / "data-set"


DATA_DIR = _detect_data_dir()
ARTIFACTS_DIR = Path(os.environ.get("BOSCH_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
CACHE_DIR = Path(os.environ.get("BOSCH_CACHE_DIR", REPO_ROOT / "cache"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Raw competition files (train_* are labelled; test_* are unlabelled and untouched).
TRAIN_NUMERIC = DATA_DIR / "train_numeric.csv"
TRAIN_DATE = DATA_DIR / "train_date.csv"
TRAIN_CATEGORICAL = DATA_DIR / "train_categorical.csv"

# Artifacts consumed by the serving path.
MODEL_PKL_PATH = ARTIFACTS_DIR / "model.pkl"
FEATURE_LIST_PATH = ARTIFACTS_DIR / "feature_list.json"
RAW_COLUMNS_PATH = ARTIFACTS_DIR / "raw_columns.json"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

ID_COL = "Id"
TARGET_COL = "Response"
RANDOM_STATE = 42

# Columns look like  L0_S0_F0 (numeric/categorical) or L0_S0_D1 (date):
#   L{line}_S{station}_{F|D}{index}
_COL_RE = re.compile(r"^L(?P<line>\d+)_S(?P<station>\d+)_[FD](?P<index>\d+)$")


def parse_column(col: str) -> dict | None:
    """Return {'line','station','index'} for a feature column, else None (Id/Response)."""
    m = _COL_RE.match(col)
    if not m:
        return None
    return {"line": int(m.group("line")), "station": int(m.group("station")),
            "index": int(m.group("index"))}


def station_of(col: str) -> str | None:
    """'L0_S0_F0' -> 'S0'. None for non-feature columns."""
    p = parse_column(col)
    return f"S{p['station']}" if p else None


def line_of(col: str) -> str | None:
    """'L0_S0_F0' -> 'L0'. None for non-feature columns."""
    p = parse_column(col)
    return f"L{p['line']}" if p else None


def group_columns_by(cols, key="station") -> dict[str, list[str]]:
    """Group feature columns by 'station' or 'line' (insertion-ordered)."""
    fn = station_of if key == "station" else line_of
    groups: dict[str, list[str]] = {}
    for c in cols:
        g = fn(c)
        if g is not None:
            groups.setdefault(g, []).append(c)
    return groups


# =====================================================================
# Memory-aware data loading
# =====================================================================
def read_ids_and_target(path=TRAIN_NUMERIC) -> pd.DataFrame:
    """Read just Id + Response from train_numeric (the only file with the label)."""
    return pd.read_csv(path, usecols=[ID_COL, TARGET_COL])


def sample_ids(id_target: pd.DataFrame, frac: float = 0.1,
               keep_all_positives: bool = True,
               random_state: int = RANDOM_STATE) -> np.ndarray:
    """Sorted array of Ids to load: all failures + `frac` of passing parts.

    Failures are so rare a plain random sample would contain almost none, so we
    keep every positive and subsample only the negatives.
    """
    pos = id_target[id_target[TARGET_COL] == 1]
    neg = id_target[id_target[TARGET_COL] == 0]
    neg_s = neg.sample(frac=frac, random_state=random_state)
    keep = pd.concat([pos, neg_s]) if keep_all_positives else id_target.sample(
        frac=frac, random_state=random_state)
    return np.sort(keep[ID_COL].to_numpy())


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    return df


def load_by_ids(path, ids=None, usecols=None, numeric=True,
                chunksize: int = 100_000) -> pd.DataFrame:
    """Load rows of a Bosch CSV, optionally restricted to a set of Ids.

    numeric=True downcasts float64 -> float32 (numeric & date files); the
    filter+chunk loop means we never hold a whole 2-3 GB file in memory.
    """
    id_set = None if ids is None else set(np.asarray(ids).tolist())
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        if id_set is not None:
            chunk = chunk[chunk[ID_COL].isin(id_set)]
        if chunk.empty:
            continue
        if numeric:
            chunk = _downcast_numeric(chunk)
        parts.append(chunk)
    if not parts:
        return pd.read_csv(path, nrows=0)
    return pd.concat(parts, ignore_index=True)


def load_all_blocks(ids=None, include_categorical: bool = False,
                    chunksize: int = 100_000) -> dict[str, pd.DataFrame]:
    """Load numeric (+ Response), date, optionally categorical blocks for `ids`.

    Returns {'numeric','date','categorical'?} each indexed by Id.
    """
    blocks = {
        "numeric": load_by_ids(TRAIN_NUMERIC, ids, numeric=True, chunksize=chunksize),
        "date": load_by_ids(TRAIN_DATE, ids, numeric=True, chunksize=chunksize),
    }
    if include_categorical:
        cat = load_by_ids(TRAIN_CATEGORICAL, ids, numeric=False, chunksize=chunksize)
        for c in cat.columns:
            if c != ID_COL:
                cat[c] = cat[c].astype("category")
        blocks["categorical"] = cat
    for df in blocks.values():
        df.set_index(ID_COL, inplace=True)
    return blocks


def save_parquet(df: pd.DataFrame, name: str) -> Path:
    path = CACHE_DIR / f"{name}.parquet"
    df.to_parquet(path)
    return path


def load_parquet(name: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def memory_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1024 ** 2


# =====================================================================
# Feature engineering  (the train/serve parity contract)
# =====================================================================
# Inputs/outputs are DataFrames indexed by Id. At inference a sparse incoming
# record is expanded to the same canonical columns seen in training (see
# reindex_to_canonical) *before* build_features, then align_to_feature_list
# locks the exact column order the model expects.
def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Columns that parse as Bosch features (exclude Id/Response if present)."""
    return [c for c in df.columns if parse_column(c) is not None]


def reindex_to_canonical(df: pd.DataFrame, canonical_cols: list[str]) -> pd.DataFrame:
    """Ensure `df` has exactly `canonical_cols` (missing -> NaN), preserving index."""
    return df.reindex(columns=canonical_cols)


def date_features(date_df: pd.DataFrame) -> pd.DataFrame:
    """Production-flow timing features from the date block."""
    cols = _feature_cols(date_df)
    d = date_df[cols]
    start, end = d.min(axis=1), d.max(axis=1)
    data = {
        "date_start": start,
        "date_end": end,
        "date_duration": end - start,
        "date_recorded_count": d.notna().sum(axis=1),
    }
    for line, line_cols in group_columns_by(cols, key="line").items():
        ls, le = d[line_cols].min(axis=1), d[line_cols].max(axis=1)
        data[f"{line}_date_start"] = ls
        data[f"{line}_date_end"] = le
        data[f"{line}_date_duration"] = le - ls
    return pd.DataFrame(data, index=date_df.index)


def station_numeric_features(num_df: pd.DataFrame) -> pd.DataFrame:
    """Per-station aggregations of numeric measurements (mean/std/min/max/count)."""
    cols = _feature_cols(num_df)
    data = {}
    for station, st_cols in group_columns_by(cols, key="station").items():
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
    for line, line_cols in group_columns_by(num_cols, key="line").items():
        data[f"{line}_numeric_nonnull"] = num_df[line_cols].notna().sum(axis=1)
    date_cols = _feature_cols(date_df)
    for station, st_cols in group_columns_by(date_cols, key="station").items():
        data[f"{station}_visited"] = date_df[st_cols].notna().any(axis=1).astype(np.int8)
    return pd.DataFrame(data, index=num_df.index)


def path_features(date_df: pd.DataFrame) -> pd.DataFrame:
    """The route a part takes through the stations (from recorded timestamps)."""
    date_cols = _feature_cols(date_df)
    d = date_df[date_cols]
    station_groups = group_columns_by(date_cols, key="station")
    visited = pd.DataFrame(
        {st: d[st_cols].notna().any(axis=1) for st, st_cols in station_groups.items()},
        index=date_df.index,
    )
    # deterministic fingerprint of the visited-station set (exact in float64)
    powers = np.array([2 ** int(s[1:]) for s in visited.columns], dtype=np.float64)
    data = {
        "path_length": visited.sum(axis=1).astype(np.int16),
        "path_id": visited.to_numpy(dtype=np.float64) @ powers,
    }
    # first / last station by timestamp; guard rows with no timestamp at all
    first = pd.Series(np.nan, index=d.index)
    last = pd.Series(np.nan, index=d.index)
    has_any = d.notna().any(axis=1)
    if bool(has_any.any()):
        sub = d[has_any]
        to_st = lambda lab: lab.map(
            lambda c: parse_column(c)["station"] if isinstance(c, str) else np.nan)
        first.loc[has_any] = to_st(sub.idxmin(axis=1))
        last.loc[has_any] = to_st(sub.idxmax(axis=1))
    data["first_station"] = first
    data["last_station"] = last
    return pd.DataFrame(data, index=date_df.index)


def categorical_features(cat_df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight categorical sparsity features (optional block)."""
    cols = _feature_cols(cat_df)
    data = {"categorical_nonnull_total": cat_df[cols].notna().sum(axis=1)}
    for line, line_cols in group_columns_by(cols, key="line").items():
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


def build_features(numeric: pd.DataFrame, date: pd.DataFrame,
                   categorical: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assemble the full engineered feature table (indexed by Id).

    Blocks must already be restricted to the canonical column universe and share
    the same Id index. Drops the target column if present in `numeric`.
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
    """Reorder/restrict engineered columns to exactly what the model expects."""
    return feat.reindex(columns=feature_list)


# =====================================================================
# Training — Phases I + II
# =====================================================================
ALL_MODELS = ["logreg", "rf", "xgb", "lgbm"]


def best_threshold(y_true, proba):
    """Decision threshold that maximizes MCC (the competition metric)."""
    from sklearn.metrics import matthews_corrcoef
    ts = np.linspace(0.01, 0.99, 99)
    mccs = [matthews_corrcoef(y_true, (proba >= t).astype(int)) for t in ts]
    i = int(np.argmax(mccs))
    return float(ts[i]), float(mccs[i])


def run_training(args) -> None:
    """Load -> engineer -> select -> train models -> tune MCC threshold -> save."""
    import joblib
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score

    t0 = time.time()
    print(f"Data dir   : {DATA_DIR}")
    print(f"Sample frac: {args.sample_frac:.0%} of passing parts | "
          f"categorical: {args.categorical} | top_k: {args.top_k} | models: {args.models}")

    # --- I.1  class distribution & class-balanced sample --------------------
    id_target = read_ids_and_target()
    n = len(id_target)
    pos = int(id_target[TARGET_COL].sum())
    print(f"\n[1] Parts: {n:,} | failures: {pos:,} ({100 * pos / n:.3f}%) | "
          f"imbalance {(n - pos) / max(pos, 1):.0f}:1")
    ids = sample_ids(id_target, frac=args.sample_frac, keep_all_positives=True)
    print(f"    working sample: {len(ids):,} parts (all failures kept)")

    # --- I.2  load blocks ---------------------------------------------------
    blocks = load_all_blocks(ids, include_categorical=args.categorical)
    numeric, date = blocks["numeric"], blocks["date"]
    categorical = blocks.get("categorical")
    y = numeric[TARGET_COL].astype("int8")
    numeric = numeric.drop(columns=[TARGET_COL])
    loaded = [("numeric", numeric), ("date", date)]
    if categorical is not None:
        loaded.append(("categorical", categorical))
    print("[2] loaded blocks:")
    for name, b in loaded:
        print(f"      {name:12s} {b.shape}  ~{memory_mb(b):,.0f} MB")

    # --- I.7  feature engineering (parquet-cached) --------------------------
    cache_name = f"features_train_{args.sample_frac}_{'cat' if args.categorical else 'nocat'}"
    feat = None if args.no_cache else load_parquet(cache_name)
    if feat is None:
        feat = build_features(numeric, date, categorical)
        save_parquet(feat, cache_name)
        print(f"[7] engineered features: {feat.shape} (cached -> {cache_name}.parquet)")
    else:
        print(f"[7] engineered features: {feat.shape} (loaded from cache)")

    # --- I.8  selection: drop constants, rank by LightGBM gain --------------
    X = feat.replace([np.inf, -np.inf], np.nan)
    const = X.nunique(dropna=True)
    X = X.drop(columns=const[const <= 1].index)
    print(f"[8] dropped {int((const <= 1).sum())} constant/empty features -> {X.shape[1]} remain")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=RANDOM_STATE)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    ranker = lgb.LGBMClassifier(n_estimators=200, scale_pos_weight=spw,
                                random_state=RANDOM_STATE, verbosity=-1)
    ranker.fit(X_tr, y_tr)
    imp = pd.Series(ranker.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected = imp.head(min(args.top_k, X.shape[1])).index.tolist()
    X_tr_s, X_val_s = X_tr[selected], X_val[selected]
    print(f"    selected {len(selected)} features; top 5: {selected[:5]}")

    # --- II  train requested models -----------------------------------------
    results, fitted = [], {}

    def evaluate(name, model, proba):
        t, mcc = best_threshold(y_val, proba)
        pred = (proba >= t).astype(int)
        row = dict(model=name, roc_auc=float(roc_auc_score(y_val, proba)),
                   f1=float(f1_score(y_val, pred)), mcc=float(mcc), threshold=float(t))
        results.append(row)
        fitted[name] = model
        print(f"    {name:20s} AUC={row['roc_auc']:.4f}  F1={row['f1']:.4f}  "
              f"MCC={row['mcc']:.4f} @ t={t:.2f}")

    print("[II] training models:")
    if "logreg" in args.models:
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        lr = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=-999)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        lr.fit(X_tr_s, y_tr)
        evaluate("LogisticRegression", lr, lr.predict_proba(X_val_s)[:, 1])

    if "rf" in args.models:
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import RandomForestClassifier
        rf = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=-999)),
            ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                           n_jobs=-1, random_state=RANDOM_STATE)),
        ])
        rf.fit(X_tr_s, y_tr)
        evaluate("RandomForest", rf, rf.predict_proba(X_val_s)[:, 1])

    if "xgb" in args.models:
        import xgboost as xgb
        xgbc = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric="auc", n_jobs=-1, random_state=RANDOM_STATE)
        xgbc.fit(X_tr_s, y_tr)
        evaluate("XGBoost", xgbc, xgbc.predict_proba(X_val_s)[:, 1])

    if "lgbm" in args.models:
        lgbm = lgb.LGBMClassifier(
            n_estimators=800, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            random_state=RANDOM_STATE, verbosity=-1)
        lgbm.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], eval_metric="auc",
                 callbacks=[lgb.early_stopping(50, verbose=False)])
        evaluate("LightGBM", lgbm, lgbm.predict_proba(X_val_s)[:, 1])

    if not results:
        sys.exit("No models trained — check --models.")

    # --- comparison, best model, persist artifacts --------------------------
    comparison = (pd.DataFrame(results)
                  .sort_values("mcc", ascending=False).reset_index(drop=True))
    print("\n[II] model comparison (sorted by MCC):")
    print(comparison.to_string(index=False,
          formatters={"roc_auc": "{:.4f}".format, "f1": "{:.4f}".format,
                      "mcc": "{:.4f}".format, "threshold": "{:.2f}".format}))
    best = comparison.iloc[0]
    best_name, best_t = best["model"], float(best["threshold"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}  (MCC={best['mcc']:.4f} @ threshold={best_t:.2f})")

    cat_cols = _feature_cols(categorical) if categorical is not None else []
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PKL_PATH)
    json.dump(selected, open(FEATURE_LIST_PATH, "w"))
    json.dump({"numeric": _feature_cols(numeric), "date": _feature_cols(date),
               "categorical": cat_cols}, open(RAW_COLUMNS_PATH, "w"))
    json.dump({"threshold": best_t, "metric": "mcc", "model": best_name},
              open(THRESHOLD_PATH, "w"))
    json.dump(comparison.to_dict(orient="records"), open(METRICS_PATH, "w"), indent=2)
    print("Saved artifacts:", *[p.name for p in [
        MODEL_PKL_PATH, FEATURE_LIST_PATH, RAW_COLUMNS_PATH, THRESHOLD_PATH, METRICS_PATH]])
    print(f"Done in {time.time() - t0:.0f}s.")


# =====================================================================
# Serving — Phase III  (FastAPI)
# =====================================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """One part to score.

    `features` is a sparse map of raw Bosch columns to values, e.g.
    ``{"L0_S0_F0": 0.123, "L0_S0_D1": 82.24, "L0_S1_F25": "T1"}``. Only the
    stations the part actually visited need be supplied; the rest are NaN.
    """
    id: int = Field(default=0, description="Part Id (optional, for traceability)")
    features: dict[str, float | str | None] = Field(
        default_factory=dict, description="Sparse raw measurement map for the part")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": 1,
                "features": {
                    "L0_S0_F0": 0.03, "L0_S0_F2": -0.05, "L0_S0_D1": 82.24,
                    "L0_S1_F25": "T1", "L3_S33_F3855": 0.12, "L3_S33_D3856": 1618.7,
                },
            }
        }
    }


class PredictResponse(BaseModel):
    id: int
    response: int = Field(description="1 = predicted failure, 0 = predicted pass")
    probability: float = Field(description="Predicted failure probability")
    threshold: float = Field(description="Decision threshold (MCC-optimized)")


class Predictor:
    """Loads artifacts and turns a sparse raw part record into a prediction."""

    def __init__(self):
        import joblib
        self.model = joblib.load(MODEL_PKL_PATH)
        self.feature_list: list[str] = json.loads(FEATURE_LIST_PATH.read_text())
        self.raw_columns: dict[str, list[str]] = json.loads(RAW_COLUMNS_PATH.read_text())
        self.threshold: float = float(json.loads(THRESHOLD_PATH.read_text())["threshold"])
        self._sets = {k: set(v) for k, v in self.raw_columns.items()}

    def _record_to_blocks(self, record: dict, part_id: int):
        """Split one sparse record into canonical numeric/date/categorical frames."""
        routed = {"numeric": {}, "date": {}, "categorical": {}}
        for col, val in record.items():
            for block, members in self._sets.items():
                if col in members:
                    routed[block][col] = val
                    break  # a column belongs to exactly one block
        idx = pd.Index([part_id], name=ID_COL)
        blocks = {}
        for block in ("numeric", "date", "categorical"):
            df = pd.DataFrame([routed[block]], index=idx)
            df = reindex_to_canonical(df, self.raw_columns[block])
            # Match the training loader's dtype exactly: numeric/date are float32
            # there. Trees split on thresholds, so a float32-vs-float64 gap could
            # flip a split — cast so inference features are identical.
            if block in ("numeric", "date"):
                df = df.astype("float32")
            blocks[block] = df
        return blocks

    def predict_one(self, record: dict, part_id: int = 0) -> dict:
        blocks = self._record_to_blocks(record, part_id)
        feat = build_features(blocks["numeric"], blocks["date"], blocks["categorical"])
        X = align_to_feature_list(feat, self.feature_list)
        proba = float(self.model.predict_proba(X)[:, 1][0])
        return {"id": part_id, "response": int(proba >= self.threshold),
                "probability": proba, "threshold": self.threshold}


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Load artifacts once and reuse across requests."""
    return Predictor()


app = FastAPI(
    title="Bosch Production Line Performance API",
    description="Predicts whether a manufactured part fails end-of-line QC.",
    version="1.0.0",
)


@app.get("/health")
def health():
    """Liveness + readiness: confirms the model artifacts loaded."""
    try:
        p = get_predictor()
        return {"status": "ok", "n_features": len(p.feature_list), "threshold": p.threshold}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"model not ready: {exc}")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Score one part from its sparse raw measurement map."""
    try:
        predictor = get_predictor()
        result = predictor.predict_one(req.features, part_id=req.id)
        return PredictResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"prediction failed: {exc}")


# =====================================================================
# CLI
# =====================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bosch",
        description="Bosch Production Line Performance — train (Phases I+II) or serve (III).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="Run Phase I+II training and write artifacts/.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pt.add_argument("--sample-frac", type=float, default=0.3,
                    help="Fraction of PASSING parts to sample; all failures kept.")
    cat = pt.add_mutually_exclusive_group()
    cat.add_argument("--include-categorical", dest="categorical", action="store_true",
                     help="Load the 2140-column categorical block (memory-heavy).")
    cat.add_argument("--no-categorical", dest="categorical", action="store_false",
                     help="Skip the categorical block (recommended for low-RAM/local).")
    pt.set_defaults(categorical=True)
    pt.add_argument("--top-k", type=int, default=80,
                    help="Features to keep after LightGBM-importance ranking.")
    pt.add_argument("--test-size", type=float, default=0.2,
                    help="Validation fraction for the stratified split.")
    pt.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
                    help="Subset of models to train.")
    pt.add_argument("--no-cache", action="store_true",
                    help="Recompute features instead of reusing the parquet cache.")

    ps = sub.add_parser("serve", help="Run the FastAPI prediction service.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ps.add_argument("--host", default="0.0.0.0")
    ps.add_argument("--port", type=int, default=8000)
    ps.add_argument("--reload", action="store_true", help="Auto-reload (development).")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        run_training(args)
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("bosch:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
