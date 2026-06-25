"""Build the two production sklearn Pipelines: main (LGBM) + baseline (RF)."""

from __future__ import annotations

from typing import Any

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

__all__ = ["build_baseline", "build_main"]


def build_main(
    *,
    scale_pos_weight: float,
    random_state: int = 42,
    **lgbm_overrides: Any,
) -> Pipeline:
    """LightGBM classifier; passthrough preprocessing - native NaN handling."""
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "average_precision",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "class_weight": None,
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
    }
    params.update(lgbm_overrides)
    return Pipeline(steps=[("clf", LGBMClassifier(**params))])


def build_baseline(
    *,
    random_state: int = 42,
    **rf_overrides: Any,
) -> Pipeline:
    """RandomForest baseline; median imputation because RF cannot split on NaN."""
    params: dict[str, Any] = {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": random_state,
        "n_jobs": -1,
    }
    params.update(rf_overrides)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(**params)),
        ]
    )
