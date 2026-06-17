# Bosch Production Line Performance — Failure Prediction

Predict whether a manufactured part fails end-of-line quality control
(`Response` = 1) from thousands of anonymized line measurements
([Kaggle competition](https://www.kaggle.com/competitions/bosch-production-line-performance/overview)).
Binary classification under **extreme class imbalance** (failures are <1% of parts),
so the headline metric is **MCC** alongside ROC-AUC and F1.

This repo delivers all three project phases:

| Phase | Deliverable | Where |
|-------|-------------|-------|
| I — Preprocessing / FE / Selection | EDA, memory reduction, datefeature engineering, selection | `notebooks/…ipynb` (+ `src/`) |
| II — Model training | LogReg, RandomForest, XGBoost, LightGBM; threshold-tunenotebooks/bosch_production_line_performance.ipynbd for MCC | same notebook |
| III — Deployment | FastAPI service + Docker image | `app/`, `Dockerfile` |

## Project layout

```
data-set/        raw competition CSVs (train_* labelled; test_* untouched)
docs/            project brief PDFs + the read_pdf.py reader
src/
  config.py        paths, column-name parsing (L#_S#_F|D#), constants
  data_loading.py  float32 / chunked loaders, Id sampling, parquet cache
  features.py      SHARED feature transforms (notebook + API import this)
notebooks/
  bosch_production_line_performance.ipynb   graded Phase I+II deliverable
scripts/
  train.py            headless Phase I+II trainer (no Jupyter; writes artifacts/)
  build_notebook.py   regenerates the notebook
app/             FastAPI service (main.py, predict.py, schema.py)
artifacts/       model.pkl, feature_list.json, raw_columns.json, threshold.json (produced by the notebook)
Dockerfile  requirements*.txt
```

`src/features.py` is the single source of truth for feature engineering — the
notebook and the API both call `build_features(...)`, so a part scored in
production goes through the identical pipeline used at training.

## 1. Train (Phases I + II)

```bash
pip install -r requirements.txt
jupyter notebook notebooks/bosch_production_line_performance.ipynb
```

Run all cells. The notebook auto-detects the data directory
(`/kaggle/input/bosch-production-line-performance` on Kaggle, else `data-set/`).
Tune `SAMPLE_FRAC` in the setup cell — it keeps **all** failures and samples that
fraction of passing parts (default `0.3`; set `1.0` to use the full data on a
large machine). Running the notebook writes the deployment artifacts into
`artifacts/`.

### Without Jupyter (headless script)

`scripts/train.py` runs the identical pipeline (same `src/` modules, same models,
same artifacts) from the command line:

```bash
python scripts/train.py                                      # full run (cloud-sized defaults)
python scripts/train.py --sample-frac 0.1 --no-categorical   # low-RAM / local machine
python scripts/train.py --sample-frac 0.05 --no-categorical --models lgbm   # quick check
```

Flags: `--sample-frac` (fraction of passing parts; all failures always kept),
`--include-categorical/--no-categorical` (the categorical block is the memory
hog — skip it on a small machine), `--top-k`, `--test-size`, `--models`
(`logreg rf xgb lgbm`), `--no-cache`. It writes the same `artifacts/` the
notebook does, so serving / Docker work identically. Run with `--help` for details.

> **Local machines:** the categorical block (~2140 columns) dominates memory; use
> `--no-categorical` and a small `--sample-frac` (e.g. `0.1`). The first run scans
> the numeric/date CSVs in chunks (slow once), then caches engineered features to
> parquet for fast re-runs.

> The data is ~15 GB. Numeric/date blocks are read as float32 and filtered by Id
> in chunks; engineered features are cached to parquet so re-runs are fast.
> Per the brief, the unlabelled `test_*.csv` files are never used — evaluation is
> on a stratified validation split.

## 2. Serve (Phase III) — local

After the notebook has produced `artifacts/`:

```bash
pip install -r requirements-serve.txt
uvicorn app.main:app --reload          # http://localhost:8000/docs
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

The image bundles `src/`, `app/`, and `artifacts/` (raw data and caches are
excluded via `.dockerignore`), and installs only the serving dependencies.
