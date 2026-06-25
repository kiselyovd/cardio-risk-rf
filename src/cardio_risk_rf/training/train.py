"""Training orchestration for main (LightGBM + Optuna) and baseline (RF + GridSearchCV)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import optuna
import pandas as pd
from lightgbm import early_stopping
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from ..data.cardio import FEATURES, TARGET
from ..models.factory import build_baseline, build_main
from ..utils import get_logger

log = get_logger(__name__)


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].copy(), df[TARGET].astype(int).copy()


def _scale_pos_weight(y: pd.Series) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("No positive samples in training target.")
    return n_neg / n_pos


def train_main(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    out_path: Path,
    optuna_trials: int = 50,
    seed: int = 42,
) -> Path:
    X_train, y_train = _xy(train_df)  # noqa: N806 - ML convention for feature matrix
    X_val, y_val = _xy(val_df)  # noqa: N806 - ML convention for feature matrix
    spw = _scale_pos_weight(y_train)
    log.info("lgbm_scale_pos_weight", value=spw)

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_estimators": 2000,
        }
        pipe = build_main(scale_pos_weight=spw, random_state=seed, **params)
        pipe.fit(
            X_train,
            y_train,
            clf__eval_set=[(X_val, y_val)],
            clf__eval_metric="average_precision",
            clf__callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
        )
        probs = pipe.predict_proba(X_val)[:, 1]
        return float(average_precision_score(y_val, probs))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
    log.info("optuna_best", pr_auc=study.best_value, params=study.best_params)

    final_params = dict(study.best_params)
    final_params["n_estimators"] = 2000
    model = build_main(scale_pos_weight=spw, random_state=seed, **final_params)
    model.fit(
        X_train,
        y_train,
        clf__eval_set=[(X_val, y_val)],
        clf__eval_metric="average_precision",
        clf__callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "best_params": final_params, "cv_pr_auc": study.best_value}, out_path
    )
    return out_path


def train_baseline(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    out_path: Path,
    cv_folds: int = 5,
    seed: int = 42,
) -> Path:
    X_train, y_train = _xy(train_df)  # noqa: N806 - ML convention for feature matrix

    base = build_baseline(random_state=seed)
    grid = {
        "clf__n_estimators": [200, 500, 1000],
        "clf__max_depth": [None, 8, 16],
        "clf__min_samples_leaf": [1, 3, 5],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        base, grid, cv=cv, scoring="average_precision", n_jobs=-1, refit=True, verbose=0
    )
    search.fit(X_train, y_train)
    log.info("rf_grid_best", pr_auc=search.best_score_, params=search.best_params_)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": search.best_estimator_,
            "best_params": search.best_params_,
            "cv_pr_auc": search.best_score_,
        },
        out_path,
    )
    return out_path
