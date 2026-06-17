"""Pydantic request / response models for the prediction API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """One part to score.

    `features` is a sparse map of raw Bosch columns to values, e.g.
    ``{"L0_S0_F0": 0.123, "L0_S0_D1": 82.24, "L0_S1_F25": "T1"}``.
    Only the stations the part actually visited need be supplied; everything
    else is treated as not-visited (NaN).
    """
    id: int = Field(default=0, description="Part Id (optional, for traceability)")
    features: dict[str, float | str | None] = Field(
        default_factory=dict, description="Sparse raw measurement map for the part"
    )

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
