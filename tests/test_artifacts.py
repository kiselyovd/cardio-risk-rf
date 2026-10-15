"""Canonical bundles, plain aliases and feature contracts must agree."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from cardio_risk_rf.artifacts import export_pipeline_alias, file_sha256, load_pipeline
from cardio_risk_rf.data.cardio import FEATURES
from cardio_risk_rf.inference.predict import predict
from cardio_risk_rf.models.factory import build_main


def fitted_bundle(path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(120, 11)), columns=FEATURES)
    target = (frame["age"] > 0).astype(int)
    model = build_main(scale_pos_weight=1, n_estimators=8, n_jobs=1)
    model.fit(frame, target)
    joblib.dump({"model": model, "best_params": {}, "cv_pr_auc": 0.5}, path)
    return frame


def test_plain_alias_preserves_original_bundle_predictions(tmp_path: Path) -> None:
    canonical, alias = tmp_path / "main.joblib", tmp_path / "sklearn_model.joblib"
    frame = fitted_bundle(canonical)
    before = file_sha256(canonical)
    receipt = export_pipeline_alias(canonical, alias)
    assert file_sha256(canonical) == before
    assert receipt["canonical_sha256"] == before
    assert receipt["max_prediction_difference"] == 0
    a, b = load_pipeline(canonical, before), load_pipeline(alias, receipt["alias_sha256"])
    np.testing.assert_array_equal(a.predict_proba(frame), b.predict_proba(frame))
    assert not isinstance(joblib.load(alias), dict)


def test_wrong_hash_fails_before_deserialization(tmp_path: Path) -> None:
    path = tmp_path / "not-a-pickle.joblib"
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="SHA256"):
        load_pipeline(path, "0" * 64)


def test_inference_requires_canonical_features_and_years(tmp_path: Path) -> None:
    path = tmp_path / "main.joblib"
    fitted_bundle(path)
    model = load_pipeline(path)
    with pytest.raises(ValueError, match="Unknown"):
        predict(model, {"TenYearCHD": 0})
    with pytest.raises(ValueError, match="years"):
        predict(model, {"age": 18000})
    with pytest.raises(ValueError, match="At least"):
        predict(model, {})
