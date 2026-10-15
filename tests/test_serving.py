"""FastAPI smoke: /health, /predict, /predict?model=baseline, bad payload.

Uses isolated temporary synthetic artifacts, preserving any operator checkpoints.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cardio_risk_rf.data.cardio import FEATURES
from cardio_risk_rf.models.factory import build_baseline, build_main
from cardio_risk_rf.serving import routes
from cardio_risk_rf.serving.main import app


@pytest.fixture(autouse=True, scope="module")
def _dummy_artefacts(tmp_path_factory):
    """Fit temporary pipelines and restore the in-process serving configuration."""
    directory = tmp_path_factory.mktemp("api-models")
    rng = np.random.default_rng(0)
    n = 200
    data = {f: rng.normal(size=n) for f in FEATURES}
    x = pd.DataFrame(data)
    y = pd.Series((rng.random(n) < 0.5).astype(int))

    main = build_main(scale_pos_weight=1.0, random_state=42)
    main.fit(x, y)
    main_path = directory / "cardio_risk_lgbm.joblib"
    joblib.dump(main, main_path)

    base = build_baseline(random_state=42)
    base.fit(x, y)
    baseline_path = directory / "cardio_risk_rf.joblib"
    joblib.dump(base, baseline_path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            routes, "_MODEL_FILES", {"main": str(main_path), "baseline": str(baseline_path)}
        )
        patch.setattr(routes, "_MODELS", {})
        yield


def _payload() -> dict:
    return {
        "age": 58,
        "gender": 2,
        "height": 175.0,
        "weight": 82.0,
        "ap_hi": 145,
        "ap_lo": 95,
        "cholesterol": 2,
        "gluc": 1,
        "smoke": 1,
        "alco": 0,
        "active": 1,
    }


def test_health_returns_200() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200


def test_predict_main_returns_contract() -> None:
    client = TestClient(app)
    r = client.post("/predict", json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["class"] in (0, 1)
    assert body["threshold"] == 0.5
    assert len(body["shap_top5"]) == 5
    assert body["model_name"] == "cardio_risk_lgbm"


def test_predict_baseline_query_param() -> None:
    client = TestClient(app)
    r = client.post("/predict?model=baseline", json=_payload())
    assert r.status_code == 200
    assert r.json()["model_name"] == "cardio_risk_rf"


def test_predict_all_null_returns_422() -> None:
    client = TestClient(app)
    empty = {k: None for k in _payload().keys()}
    r = client.post("/predict", json=empty)
    assert r.status_code == 422
