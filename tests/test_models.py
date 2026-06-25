"""Factory tests - LGBM passes NaN through, RF gets median imputer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from cardio_risk_rf.data.cardio import FEATURES
from cardio_risk_rf.models.factory import build_baseline, build_main


def _toy(n: int = 50, nan_rate: float = 0.1, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f: rng.normal(size=n) for f in FEATURES})
    mask = rng.random(X.shape) < nan_rate
    X = X.mask(mask)
    y = pd.Series((rng.random(n) < 0.2).astype(int), name="TenYearCHD")
    return X, y


def test_build_main_is_pipeline_and_accepts_nan() -> None:
    model = build_main(scale_pos_weight=1.0, random_state=42)
    assert isinstance(model, Pipeline)
    X, y = _toy()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.isfinite(probs).all()


def test_build_baseline_imputes_medians_and_accepts_nan() -> None:
    model = build_baseline(random_state=42)
    assert isinstance(model, Pipeline)
    X, y = _toy()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.isfinite(probs).all()


def test_build_main_uses_scale_pos_weight() -> None:
    spw = 5.7
    model = build_main(scale_pos_weight=spw, random_state=42)
    est = model.named_steps["clf"]
    assert est.scale_pos_weight == spw
