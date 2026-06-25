"""Pydantic request/response schemas for /predict.

Fields match the sulianova Cardiovascular Disease Dataset schema
(11 features + binary `cardio` target). `age` is in years (source data
is in days; converted at load time in `data/cardio.py`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """Single-patient feature payload; 11 sulianova cardio-risk features."""

    age: int | None = Field(None, ge=0, le=120, description="age in years")
    gender: int | None = Field(None, ge=1, le=2, description="1=female, 2=male")
    height: float | None = Field(None, ge=50, le=250, description="cm")
    weight: float | None = Field(None, ge=20, le=300, description="kg")
    ap_hi: int | None = Field(None, ge=60, le=300, description="systolic BP, mmHg")
    ap_lo: int | None = Field(None, ge=30, le=200, description="diastolic BP, mmHg")
    cholesterol: int | None = Field(
        None, ge=1, le=3, description="1=normal, 2=above normal, 3=well above normal"
    )
    gluc: int | None = Field(
        None, ge=1, le=3, description="1=normal, 2=above normal, 3=well above normal"
    )
    smoke: int | None = Field(None, ge=0, le=1)
    alco: int | None = Field(None, ge=0, le=1, description="alcohol intake")
    active: int | None = Field(None, ge=0, le=1, description="physical activity")


class ShapEntry(BaseModel):
    """Single SHAP contribution for one feature."""

    feature: str
    value: float | int | None
    shap: float


class PredictionResponse(BaseModel):
    """Response schema for /predict."""

    probability: float
    cls: int = Field(..., alias="class")
    threshold: float
    shap_top5: list[ShapEntry]
    model_version: str
    model_name: str
    request_id: str

    class Config:
        """Pydantic configuration - allow populating `cls` by field or alias."""

        populate_by_name = True
