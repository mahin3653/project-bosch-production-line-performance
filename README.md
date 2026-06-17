# Bosch Production Line Performance — Failure Prediction

Predict whether a manufactured part fails end-of-line quality control
(`Response` = 1) from thousands of anonymized line measurements
([Kaggle competition](https://www.kaggle.com/competitions/bosch-production-line-performance/overview)).
Binary classification under **extreme class imbalance** (failures are <1% of parts),
so the headline metric is **MCC** alongside ROC-AUC and F1.

Everything lives in a single file, **`bosch.py`**, driven by flags:

| Phase | Deliverable | Command |
|-------|-------------|---------|
| I — Preprocessing / FE / Selection | memory reduction, feature engineering, selection | `python bosch.py train` |
| II — Model training | LogReg, RandomForest, XGBoost, LightGBM; threshold-tuned for MCC | `python bosch.py train` |
| III — Deployment | FastAPI service + Docker image | `uvicorn bosch:app` / `Dockerfile` |

## Project layout

```
bosch.py         config + data loading + feature engineering + training + FastAPI app
data-set/        raw competition CSVs (train_* labelled; test_* untouched)
docs/            project brief PDFs
artifacts/       model.pkl, feature_list.json, raw_columns.json, threshold.json (produced by training)
Dockerfile  requirements.txt  requirements-serve.txt
```

`bosch.py` defines `build_features(...)` once and both training and the API call
it, so a part scored in production goes through the identical pipeline used at
training (train/serve parity).

## 1. Train (Phases I + II)

```bash
pip install -r requirements.txt

python bosch.py train                                      # full run (cloud-sized defaults)
python bosch.py train --sample-frac 0.1 --no-categorical   # low-RAM / local machine
python bosch.py train --sample-frac 0.05 --no-categorical --models lgbm   # quick check
```

It auto-detects the data directory (`/kaggle/input/bosch-production-line-performance`
on Kaggle, else `data-set/`), trains the requested models, tunes each decision
threshold for MCC, prints an ROC-AUC / F1 / MCC comparison, and writes the
deployment artifacts into `artifacts/`.

Flags: `--sample-frac` (fraction of passing parts; all failures always kept),
`--include-categorical/--no-categorical` (the categorical block is the memory
hog — skip it on a small machine), `--top-k`, `--test-size`, `--models`
(`logreg rf xgb lgbm`), `--no-cache`. Run `python bosch.py train --help` for details.

> **Local machines:** the categorical block (~2140 columns) dominates memory; use
> `--no-categorical` and a small `--sample-frac` (e.g. `0.1`). The first run scans
> the numeric/date CSVs in chunks (slow once), then caches engineered features to
> parquet for fast re-runs.

> The data is ~15 GB. Numeric/date blocks are read as float32 and filtered by Id
> in chunks; engineered features are cached to parquet so re-runs are fast.
> Per the brief, the unlabelled `test_*.csv` files are never used — evaluation is
> on a stratified validation split.

## 2. Serve (Phase III) — local

After training has produced `artifacts/`:

```bash
pip install -r requirements-serve.txt
python bosch.py serve            # or: uvicorn bosch:app --reload
# -> http://localhost:8000/docs
```

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "id": 1,
  "features": {"L0_S0_F0": 0.03, "L0_S0_D1": 82.24, "L0_S1_F25": "T1"}
}'
# -> {"id":1,"response":0,"probability":0.0123,"threshold":0.27}
```

`features` is a **sparse** map of raw `L#_S#_F#` / `_D#` columns — only the
stations a part actually visited need be supplied; the rest are treated as
not-visited.

## 3. Docker + DockerHub

```bash
docker build -t <dockerhub-user>/bosch-pl-api:latest .
docker run -p 8000:8000 <dockerhub-user>/bosch-pl-api:latest
# verify, then publish:
docker login
docker push <dockerhub-user>/bosch-pl-api:latest
```

The image bundles `bosch.py` and `artifacts/` (raw data and caches are excluded
via `.dockerignore`) and installs only the serving dependencies.
