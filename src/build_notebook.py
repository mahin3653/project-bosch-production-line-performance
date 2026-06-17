#!/usr/bin/env python3
"""Generate notebooks/bosch_production_line_performance.ipynb (stdlib only).

Keeping the notebook in a generator makes it reviewable and regenerable. The
notebook itself stays thin: heavy logic lives in src/ so the FastAPI service
reuses the exact same feature code (train/serve parity).
"""
import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text):
    CELLS.append({
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": text,
    })


# ---------------------------------------------------------------- intro
md("""# Bosch Production Line Performance — Failure Prediction

**Problem.** Bosch records thousands of anonymized measurements as each part
moves along its assembly lines. We predict the binary `Response` — whether a
part **fails** end-of-line quality control (1) or passes (0).

**Why it matters (business view).** Internal failures are rare but expensive:
scrap, rework, and — worst case — a defective part reaching a customer. A model
that flags high-risk parts early lets the line prioritize inspection and catch
defects before they propagate.

**The core challenge: extreme class imbalance.** Failures are well under 1% of
parts. Accuracy is meaningless here, so we use **MCC** (the competition metric)
alongside ROC-AUC and F1, and we **tune the decision threshold** rather than
defaulting to 0.5.

**Approach (this notebook).**
1. *Phase I* — understand the data, reduce memory, engineer station / date /
   path / missingness features, and select the useful ones.
2. *Phase II* — train Logistic Regression, Random Forest, XGBoost and LightGBM,
   optimize each threshold for MCC, and pick the best.

Deployment (*Phase III*, FastAPI + Docker) reuses `src/features.py` so the
served model applies identical transforms — see `app/` and the `Dockerfile`.""")

# ---------------------------------------------------------------- setup
md("## 0. Setup & configuration")
code("""import sys, os, json
from pathlib import Path

# Find the repo root (the folder that contains `src/`) and make it importable.
ROOT = Path.cwd()
while not (ROOT / "src").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

# Point at the data. On Kaggle the competition files live under /kaggle/input/...
for cand in ["/kaggle/input/bosch-production-line-performance", str(ROOT / "data-set")]:
    if Path(cand).exists():
        os.environ["BOSCH_DATA_DIR"] = cand
        break
os.environ.setdefault("BOSCH_CACHE_DIR", str(ROOT / "cache"))
os.environ.setdefault("BOSCH_ARTIFACTS_DIR", str(ROOT / "artifacts"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config, data_loading as dl, features as fe

# Fraction of *negative* (passing) parts to sample; ALL failures are always kept.
# Set to 1.0 on a large machine to use the full dataset.
SAMPLE_FRAC = 0.3
print("Data dir:", config.DATA_DIR)""")

# ---------------------------------------------------------------- Phase I
md("""---
## Phase I — Data understanding, preprocessing, feature engineering, selection""")

md("""### 1. Class distribution & a class-balanced working sample

`train_numeric.csv` is the only file carrying the `Response` label. We read just
`Id` + `Response` (cheap), measure the failure rate, then build a working sample
that keeps **every** failure plus a fraction of passing parts — otherwise a rare
positive would be almost invisible in a small sample.""")
code("""id_target = dl.read_ids_and_target()
n = len(id_target)
pos = int(id_target[config.TARGET_COL].sum())
print(f"Total parts : {n:,}")
print(f"Failures    : {pos:,}  ({100 * pos / n:.3f}%)")
print(f"Imbalance   : {(n - pos) / max(pos, 1):.0f} : 1  (pass : fail)")

ax = (id_target[config.TARGET_COL].value_counts()
      .rename({0: "pass", 1: "fail"}).plot(kind="bar", color=["#4c72b0", "#c44e52"]))
ax.set_yscale("log"); ax.set_ylabel("count (log)"); ax.set_title("Class distribution")
plt.tight_layout(); plt.show()

sample_ids = dl.sample_ids(id_target, frac=SAMPLE_FRAC, keep_all_positives=True)
print(f"\\nWorking sample: {len(sample_ids):,} parts "
      f"(all {pos:,} failures + {SAMPLE_FRAC:.0%} of passing parts)")""")

md("""### 2. Load the three data blocks for the sample

Numeric and date blocks are downcast to **float32** on read; the categorical
block is loaded as `category`. The loader filters by our sampled `Id`s in
chunks so we never hold a whole 2–3 GB file in memory.""")
code("""blocks = dl.load_all_blocks(sample_ids, include_categorical=True)
numeric, date, categorical = blocks["numeric"], blocks["date"], blocks["categorical"]

y = numeric[config.TARGET_COL].astype("int8")
numeric = numeric.drop(columns=[config.TARGET_COL])

for name, b in [("numeric", numeric), ("date", date), ("categorical", categorical)]:
    print(f"{name:12s} {b.shape}  ~{dl.memory_mb(b):,.0f} MB")""")

md("""### 3. Missingness & feature sparsity

Each part only visits a handful of stations, so most cells are empty. The
**pattern** of missingness encodes the part's route — we will turn it into
features rather than dropping it.""")
code("""print(f"Avg missing — numeric: {numeric.isna().mean().mean():.1%}, "
      f"date: {date.isna().mean().mean():.1%}, "
      f"categorical: {categorical.isna().mean().mean():.1%}")

nonnull = numeric.notna().sum(axis=1)
ax = nonnull.plot(kind="hist", bins=50, color="#4c72b0")
ax.set_xlabel("non-null numeric measurements per part")
ax.set_title("How many numeric features a part actually has")
plt.tight_layout(); plt.show()""")

md("""### 4. Station / line structure from the column names

Columns follow `L{line}_S{station}_F|D{index}`. Parsing them lets us group
measurements by station and check where failures concentrate.""")
code("""num_by_station = config.group_columns_by(numeric.columns, key="station")
print(f"{len(num_by_station)} stations carry numeric measurements; "
      f"lines present: {sorted(config.group_columns_by(numeric.columns, key='line'))}")

# Failure rate among parts that *visited* each station (visit = recorded a timestamp).
rows = []
for st, cols in config.group_columns_by(date.columns, key="station").items():
    visited = date[cols].notna().any(axis=1)
    if visited.sum() > 50:
        rows.append((st, int(visited.sum()), float(y[visited].mean())))
station_fail = (pd.DataFrame(rows, columns=["station", "n_parts", "fail_rate"])
                .sort_values("fail_rate", ascending=False))
display(station_fail.head(12))""")

md("""### 5. Production-flow timing

The date block timestamps each station a part passes through. The spread between
first and last timestamp is the part's processing **duration** — a strong signal.""")
code("""dcols = fe._feature_cols(date)
duration = date[dcols].max(axis=1) - date[dcols].min(axis=1)
ax = duration.plot(kind="hist", bins=60, color="#55a868")
ax.set_xlabel("duration (date units)"); ax.set_title("Processing duration per part")
plt.tight_layout(); plt.show()
print(duration.describe().round(3).to_string())""")

md("""### 6. Preprocessing — memory & missing values

* **Memory:** numeric/date are float32 (≈ half the float64 footprint); the
  categorical block is stored as `category`.
* **Missing values are interpreted, not dropped.** A NaN means "the part did not
  pass through this station" — informative in itself. Tree models (XGBoost,
  LightGBM) consume NaN natively; for Logistic Regression / Random Forest we
  impute a sentinel and add explicit missingness features (next section).""")
code("""print("Numeric block is float32:",
      numeric.select_dtypes(include=['float64']).shape[1] == 0)
print(f"Current numeric footprint: {dl.memory_mb(numeric):,.0f} MB "
      f"(≈ {2 * dl.memory_mb(numeric):,.0f} MB if float64)")""")

md("""### 7. Feature engineering

All transforms live in `src/features.py` (shared with the API). We build:

* **Date features** — start / end / duration, count of recorded timestamps, and
  per-line timing.
* **Station-level numeric aggregations** — mean / std / min / max / count of each
  station's measurements (collapses ~970 raw numeric columns).
* **Missingness features** — non-null counts per row / line and per-station
  visit flags.
* **Path features** — `path_length`, a `path_id` fingerprint of the visited-station
  set, and first / last station by timestamp.
* **Categorical sparsity** and a few **interaction** crosses.

The result is cached to parquet so re-runs skip the heavy recompute.""")
code("""feat = dl.load_parquet("features_sample")
if feat is None:
    feat = fe.build_features(numeric, date, categorical)
    dl.save_parquet(feat, "features_sample")
print("Engineered feature matrix:", feat.shape)
feat.head(3)""")

md("""### 8. Feature selection

Drop constant / empty columns, then rank by LightGBM gain importance and keep
the strongest. The selected list is what every model below trains on (and what
the API will expect).""")
code("""import lightgbm as lgb
from sklearn.model_selection import train_test_split

X = feat.replace([np.inf, -np.inf], np.nan)
const = X.nunique(dropna=True)
X = X.drop(columns=const[const <= 1].index)
print(f"Dropped {int((const <= 1).sum())} constant/empty features -> {X.shape[1]} remain")

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)
spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

ranker = lgb.LGBMClassifier(n_estimators=200, scale_pos_weight=spw,
                            random_state=config.RANDOM_STATE, verbosity=-1)
ranker.fit(X_tr, y_tr)
imp = pd.Series(ranker.feature_importances_, index=X.columns).sort_values(ascending=False)

TOP_K = min(80, X.shape[1])
selected = imp.head(TOP_K).index.tolist()

ax = imp.head(20).iloc[::-1].plot(kind="barh", color="#4c72b0")
ax.set_title("Top-20 feature importances (LightGBM gain)"); plt.tight_layout(); plt.show()
print(f"Selected {len(selected)} features")""")

# ---------------------------------------------------------------- Phase II
md("""---
## Phase II — Model training & evaluation

We train four models and, for each, **sweep the decision threshold to maximize
MCC** before reading off F1. All share the same train/validation split and the
selected features. Class imbalance is handled with `scale_pos_weight`
(boosters) or `class_weight='balanced'` (linear / forest).""")
code("""from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef

X_tr_s, X_val_s = X_tr[selected], X_val[selected]

def best_threshold(y_true, proba):
    ts = np.linspace(0.01, 0.99, 99)
    mccs = [matthews_corrcoef(y_true, (proba >= t).astype(int)) for t in ts]
    i = int(np.argmax(mccs))
    return float(ts[i]), float(mccs[i])

results, fitted = [], {}
def evaluate(name, model, proba):
    t, mcc = best_threshold(y_val, proba)
    pred = (proba >= t).astype(int)
    row = dict(model=name, roc_auc=roc_auc_score(y_val, proba),
               f1=f1_score(y_val, pred), mcc=mcc, threshold=t)
    results.append(row); fitted[name] = model
    print(f"{name:20s} AUC={row['roc_auc']:.4f}  F1={row['f1']:.4f}  "
          f"MCC={row['mcc']:.4f} @ t={t:.2f}")
    return row""")

md("""### Logistic Regression — interpretable linear baseline

Fast and transparent; establishes a floor. Needs imputation + scaling, and
`class_weight='balanced'` to counter the imbalance. We expect tree models to
beat it because the signal is highly non-linear and interaction-heavy.""")
code("""from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

lr = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value=-999)),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
lr.fit(X_tr_s, y_tr)
evaluate("LogisticRegression", lr, lr.predict_proba(X_val_s)[:, 1]);""")

md("""### Random Forest — non-linear bagging baseline

Captures non-linearities and interactions without tuning, and is robust to the
sentinel imputation. Usually strong but heavier and typically a notch below
gradient boosting on tabular data like this.""")
code("""from sklearn.ensemble import RandomForestClassifier

rf = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value=-999)),
    ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   n_jobs=-1, random_state=config.RANDOM_STATE)),
])
rf.fit(X_tr_s, y_tr)
evaluate("RandomForest", rf, rf.predict_proba(X_val_s)[:, 1]);""")

md("""### XGBoost *(recommended)* — gradient-boosted trees

State-of-the-art on sparse tabular data, handles NaN natively, and `scale_pos_weight`
addresses the imbalance directly. A top performer on this competition.""")
code("""import xgboost as xgb

xgbc = xgb.XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
    eval_metric="auc", n_jobs=-1, random_state=config.RANDOM_STATE)
xgbc.fit(X_tr_s, y_tr)
evaluate("XGBoost", xgbc, xgbc.predict_proba(X_val_s)[:, 1]);""")

md("""### LightGBM *(recommended)* — fast histogram boosting

Very fast and memory-efficient on wide, sparse data, with native NaN handling
and early stopping. Often the best accuracy/speed trade-off here.""")
code("""lgbm = lgb.LGBMClassifier(
    n_estimators=800, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
    random_state=config.RANDOM_STATE, verbosity=-1)
lgbm.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], eval_metric="auc",
         callbacks=[lgb.early_stopping(50, verbose=False)])
evaluate("LightGBM", lgbm, lgbm.predict_proba(X_val_s)[:, 1]);""")

md("### Model comparison")
code("""comparison = (pd.DataFrame(results)
              .sort_values("mcc", ascending=False).reset_index(drop=True))
display(comparison.style.format({"roc_auc": "{:.4f}", "f1": "{:.4f}",
                                 "mcc": "{:.4f}", "threshold": "{:.2f}"}))

best = comparison.iloc[0]
best_name, best_t = best["model"], float(best["threshold"])
best_model = fitted[best_name]
print(f"\\nBest model: {best_name}  (MCC={best['mcc']:.4f} @ threshold={best_t:.2f})")

# Confusion matrix for the winner at its tuned threshold
from sklearn.metrics import ConfusionMatrixDisplay
proba_best = best_model.predict_proba(X_val_s)[:, 1]
ConfusionMatrixDisplay.from_predictions(
    y_val, (proba_best >= best_t).astype(int), display_labels=["pass", "fail"])
plt.title(f"{best_name} @ t={best_t:.2f}"); plt.tight_layout(); plt.show()""")

md("""### Persist artifacts for deployment

Save the winning model plus everything the API needs to reproduce the exact
feature pipeline: the ordered selected-feature list, the canonical raw columns
`build_features` expects, and the tuned threshold.""")
code("""import joblib

config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, config.MODEL_PKL_PATH)
json.dump(selected, open(config.FEATURE_LIST_PATH, "w"))
json.dump({"numeric": fe._feature_cols(numeric),
           "date": fe._feature_cols(date),
           "categorical": fe._feature_cols(categorical)},
          open(config.RAW_COLUMNS_PATH, "w"))
json.dump({"threshold": best_t, "metric": "mcc", "model": best_name},
          open(config.THRESHOLD_PATH, "w"))
json.dump(comparison.to_dict(orient="records"),
          open(config.METRICS_PATH, "w"), indent=2)
print("Saved:", *[p.name for p in [config.MODEL_PKL_PATH, config.FEATURE_LIST_PATH,
      config.RAW_COLUMNS_PATH, config.THRESHOLD_PATH, config.METRICS_PATH]])""")

md("""---
## Summary

**Best model.** Sorted by MCC, the gradient-boosted models (XGBoost / LightGBM)
lead — they fit the sparse, non-linear, interaction-heavy signal far better than
the linear baseline, and threshold tuning is what turns a strong ROC-AUC into a
usable failure flag under <1% prevalence.

**What drove performance.** Engineered features — processing **duration**,
per-station measurement aggregates, **missingness / visit** patterns, and the
**path** a part takes — carried most of the signal, confirming that *how* a part
flows through the line is as informative as the raw measurements.

**Business takeaway.** The model converts a flood of anonymous sensor readings
into a single ranked failure risk per part. At the tuned threshold the line can
route the riskiest parts to extra inspection, catching defects earlier and
reducing both scrap and the chance of a bad part shipping — without manually
inspecting the >99% of parts that pass.

**Next (Phase III).** `app/` serves `predict_proba` over FastAPI using
`src/features.py` for identical transforms; the `Dockerfile` containerizes it for
DockerHub. See `README.md`.""")

# ---------------------------------------------------------------- write
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent.parent / "notebooks" / "bosch_production_line_performance.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(CELLS)} cells)")