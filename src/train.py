#!/usr/bin/env python3
"""Headless trainer for Phases I + II — the notebook pipeline without Jupyter.

Runs the identical steps as `notebooks/bosch_production_line_performance.ipynb`
(load -> engineer -> select -> train 4 models -> MCC-threshold tune -> compare ->
save artifacts), but as a plain CLI. It imports the same `src/` modules the
notebook and the API use, so results match and train/serve parity is preserved.

Examples
--------
# Full run (cloud-sized, same defaults as the notebook):
    python scripts/train.py

# Local-friendly run (fits a ~15 GB machine with little free RAM):
    python scripts/train.py --sample-frac 0.1 --no-categorical

# Quick smoke (LightGBM only, tiny sample):
    python scripts/train.py --sample-frac 0.05 --no-categorical --models lgbm

Artifacts (model.pkl, feature_list.json, raw_columns.json, threshold.json,
metrics.json) are written to BOSCH_ARTIFACTS_DIR (default ./artifacts), exactly
as the notebook does — so the FastAPI service / Docker image can serve them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make the repo root importable whether run as `python scripts/train.py` or `-m`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Point at the data the same way the notebook does, unless already overridden.
import os
for _cand in ["/kaggle/input/bosch-production-line-performance", str(ROOT / "data-set")]:
    if Path(_cand).exists():
        os.environ.setdefault("BOSCH_DATA_DIR", _cand)
        break
os.environ.setdefault("BOSCH_CACHE_DIR", str(ROOT / "cache"))
os.environ.setdefault("BOSCH_ARTIFACTS_DIR", str(ROOT / "artifacts"))

import numpy as np
import pandas as pd

from src import config, data_loading as dl, features as fe


ALL_MODELS = ["logreg", "rf", "xgb", "lgbm"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train Bosch failure-prediction models (Phases I+II), no notebook.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sample-frac", type=float, default=0.3,
                   help="Fraction of PASSING parts to sample; all failures are always kept.")
    cat = p.add_mutually_exclusive_group()
    cat.add_argument("--include-categorical", dest="categorical", action="store_true",
                     help="Load the 2140-column categorical block (memory-heavy).")
    cat.add_argument("--no-categorical", dest="categorical", action="store_false",
                     help="Skip the categorical block (recommended for low-RAM/local runs).")
    p.set_defaults(categorical=True)
    p.add_argument("--top-k", type=int, default=80,
                   help="Number of features to keep after LightGBM-importance ranking.")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Validation fraction for the stratified split.")
    p.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
                   help="Subset of models to train.")
    p.add_argument("--no-cache", action="store_true",
                   help="Recompute features instead of reusing the parquet cache.")
    return p.parse_args(argv)


def best_threshold(y_true, proba):
    """Decision threshold that maximizes MCC (the competition metric)."""
    from sklearn.metrics import matthews_corrcoef
    ts = np.linspace(0.01, 0.99, 99)
    mccs = [matthews_corrcoef(y_true, (proba >= t).astype(int)) for t in ts]
    i = int(np.argmax(mccs))
    return float(ts[i]), float(mccs[i])


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    print(f"Data dir   : {config.DATA_DIR}")
    print(f"Sample frac: {args.sample_frac:.0%} of passing parts | "
          f"categorical: {args.categorical} | top_k: {args.top_k} | models: {args.models}")

    # --- Phase I.1 — class distribution & class-balanced sample -------------
    id_target = dl.read_ids_and_target()
    n = len(id_target)
    pos = int(id_target[config.TARGET_COL].sum())
    print(f"\n[1] Parts: {n:,} | failures: {pos:,} ({100 * pos / n:.3f}%) | "
          f"imbalance {(n - pos) / max(pos, 1):.0f}:1")
    sample_ids = dl.sample_ids(id_target, frac=args.sample_frac, keep_all_positives=True)
    print(f"    working sample: {len(sample_ids):,} parts (all failures kept)")

    # --- Phase I.2 — load blocks for the sample -----------------------------
    blocks = dl.load_all_blocks(sample_ids, include_categorical=args.categorical)
    numeric, date = blocks["numeric"], blocks["date"]
    categorical = blocks.get("categorical")  # None when --no-categorical
    y = numeric[config.TARGET_COL].astype("int8")
    numeric = numeric.drop(columns=[config.TARGET_COL])
    loaded = [("numeric", numeric), ("date", date)]
    if categorical is not None:
        loaded.append(("categorical", categorical))
    print("[2] loaded blocks:")
    for name, b in loaded:
        print(f"      {name:12s} {b.shape}  ~{dl.memory_mb(b):,.0f} MB")

    # --- Phase I.7 — feature engineering (cached to parquet) ----------------
    cache_name = f"features_train_{args.sample_frac}_{'cat' if args.categorical else 'nocat'}"
    feat = None if args.no_cache else dl.load_parquet(cache_name)
    if feat is None:
        feat = fe.build_features(numeric, date, categorical)
        dl.save_parquet(feat, cache_name)
        print(f"[7] engineered features: {feat.shape} (cached -> {cache_name}.parquet)")
    else:
        print(f"[7] engineered features: {feat.shape} (loaded from cache)")

    # --- Phase I.8 — selection: drop constants, rank by LightGBM gain -------
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    X = feat.replace([np.inf, -np.inf], np.nan)
    const = X.nunique(dropna=True)
    X = X.drop(columns=const[const <= 1].index)
    print(f"[8] dropped {int((const <= 1).sum())} constant/empty features -> {X.shape[1]} remain")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=config.RANDOM_STATE)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    ranker = lgb.LGBMClassifier(n_estimators=200, scale_pos_weight=spw,
                                random_state=config.RANDOM_STATE, verbosity=-1)
    ranker.fit(X_tr, y_tr)
    imp = pd.Series(ranker.feature_importances_, index=X.columns).sort_values(ascending=False)
    selected = imp.head(min(args.top_k, X.shape[1])).index.tolist()
    X_tr_s, X_val_s = X_tr[selected], X_val[selected]
    print(f"    selected {len(selected)} features; top 5: {selected[:5]}")

    # --- Phase II — train the requested models ------------------------------
    from sklearn.metrics import roc_auc_score, f1_score

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
                                           n_jobs=-1, random_state=config.RANDOM_STATE)),
        ])
        rf.fit(X_tr_s, y_tr)
        evaluate("RandomForest", rf, rf.predict_proba(X_val_s)[:, 1])

    if "xgb" in args.models:
        import xgboost as xgb
        xgbc = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric="auc", n_jobs=-1, random_state=config.RANDOM_STATE)
        xgbc.fit(X_tr_s, y_tr)
        evaluate("XGBoost", xgbc, xgbc.predict_proba(X_val_s)[:, 1])

    if "lgbm" in args.models:
        lgbm = lgb.LGBMClassifier(
            n_estimators=800, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            random_state=config.RANDOM_STATE, verbosity=-1)
        lgbm.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], eval_metric="auc",
                 callbacks=[lgb.early_stopping(50, verbose=False)])
        evaluate("LightGBM", lgbm, lgbm.predict_proba(X_val_s)[:, 1])

    if not results:
        sys.exit("No models trained — check --models.")

    # --- comparison & best model -------------------------------------------
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

    # --- persist artifacts (identical to the notebook) ----------------------
    import joblib
    cat_cols = fe._feature_cols(categorical) if categorical is not None else []
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, config.MODEL_PKL_PATH)
    json.dump(selected, open(config.FEATURE_LIST_PATH, "w"))
    json.dump({"numeric": fe._feature_cols(numeric),
               "date": fe._feature_cols(date),
               "categorical": cat_cols},
              open(config.RAW_COLUMNS_PATH, "w"))
    json.dump({"threshold": best_t, "metric": "mcc", "model": best_name},
              open(config.THRESHOLD_PATH, "w"))
    json.dump(comparison.to_dict(orient="records"),
              open(config.METRICS_PATH, "w"), indent=2)
    print("Saved artifacts:", *[p.name for p in [
        config.MODEL_PKL_PATH, config.FEATURE_LIST_PATH, config.RAW_COLUMNS_PATH,
        config.THRESHOLD_PATH, config.METRICS_PATH]])
    print(f"Done in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
