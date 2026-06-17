"""FastAPI service for Bosch production-line failure prediction.

Run locally:   uvicorn app.main:app --reload
Docs (Swagger): http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.predict import get_predictor
from app.schema import PredictRequest, PredictResponse

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
    except Exception as exc:  # artifacts missing / unreadable
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
