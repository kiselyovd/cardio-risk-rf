"""Load trusted author-owned bundles and export a compatible Hub Pipeline alias."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from .data.cardio import FEATURES


def file_sha256(path: Path) -> str:
    """Hash exact artifact bytes before deserializing a known model."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_pipeline(path: Path | str, expected_sha256: str | None = None) -> Pipeline:
    """Accept the canonical dict bundle or its plain fitted Pipeline alias."""
    path = Path(path)
    if expected_sha256 and file_sha256(path) != expected_sha256:
        raise ValueError("Model artifact SHA256 mismatch")
    payload = joblib.load(path)
    model = payload.get("model") if isinstance(payload, dict) else payload
    if not isinstance(model, Pipeline):
        raise ValueError("Expected a fitted sklearn Pipeline or model bundle")
    if list(getattr(model, "feature_names_in_", [])) != FEATURES:
        raise ValueError("Artifact feature schema differs from the 11-feature cardio contract")
    if list(model.classes_) != [0, 1]:
        raise ValueError("Artifact target must be binary cardio, classes [0,1]")
    return model


def export_pipeline_alias(source: Path, destination: Path) -> dict[str, Any]:
    """Export the main estimator for Hub's sklearn_model.joblib loading convention."""
    model = load_pipeline(source)
    if not isinstance(model.named_steps.get("clf"), LGBMClassifier):
        raise ValueError("Main inference alias requires the LightGBM pipeline")
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination, compress=3)
    alias = load_pipeline(destination)
    # Synthetic interface probes verify serialization only, not medical quality.
    probes = pd.DataFrame(
        [
            [45, 1, 165, 65, 120, 80, 1, 1, 0, 0, 1],
            [60, 2, 175, 85, 160, 100, 2, 2, 1, 0, 0],
            [55, 1, 160, 70, 135, 85, 3, 1, 0, 1, 1],
        ],
        columns=FEATURES,
    )
    difference = float(np.max(np.abs(model.predict_proba(probes) - alias.predict_proba(probes))))
    if difference > 1e-12:
        raise ValueError("Inference alias predictions differ from the canonical bundle")
    return {
        "canonical_sha256": file_sha256(source),
        "alias_sha256": file_sha256(destination),
        "max_prediction_difference": difference,
        "features": FEATURES,
        "alias_payload": "sklearn.pipeline.Pipeline",
    }
