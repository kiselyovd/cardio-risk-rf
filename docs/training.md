# Training

End-to-end training is orchestrated by `scripts/train_all.py`, which runs the baseline and main model in sequence, produces evaluation metrics, the calibration plot, and the global SHAP summary. The full run is CPU-only - no GPU is required or used at any stage.

## Prerequisites

```bash
uv sync --all-groups
bash scripts/sync_data.sh
uv run python -m cardio_risk_rf.data.prepare --raw data/raw/framingham.csv --out data/processed
```

`sync_data.sh` pulls the Framingham CSV from Kaggle, with an HF Datasets mirror (`kiselyovd/framingham`) as fallback. `prepare.py` splits 70/15/15 stratified by `TenYearCHD` and writes `train.parquet`, `val.parquet`, `test.parquet` under `data/processed/`.

## Commands

Full training pipeline (baseline + main + evaluation + SHAP):

```bash
uv run python scripts/train_all.py
```

Individual stages:

```bash
# Main model only (LightGBM + Optuna)
uv run python -m cardio_risk_rf.training.train model=lgbm

# Baseline only (RandomForest + GridSearchCV)
uv run python -m cardio_risk_rf.training.train model=rf

# Evaluation (metrics + calibration plot)
uv run python -m cardio_risk_rf.evaluation.evaluate

# Global SHAP summary from main model
uv run python -m cardio_risk_rf.explain --model artifacts/main/cardio_risk_lgbm.joblib
```

## Hyperparameters

**LightGBM (main) - Optuna TPE, 50 trials, seed=42.** Search ranges (see `configs/model/lgbm.yaml`):

- `num_leaves`: 16-128
- `learning_rate`: 0.01-0.2 (log)
- `max_depth`: -1 or 3-12
- `min_child_samples`: 5-60
- `reg_alpha`: 1e-8-10 (log)
- `reg_lambda`: 1e-8-10 (log)
- `feature_fraction`: 0.6-1.0
- `bagging_fraction`: 0.6-1.0

Class imbalance: `scale_pos_weight = N_neg / N_pos` computed on train. Early stopping on val ROC-AUC with patience 30. No feature scaling or imputation - LightGBM handles NaN natively.

**RandomForest (baseline) - GridSearchCV, 5-fold stratified.** Grid (see `configs/model/rf.yaml`):

- `n_estimators`: [200, 500]
- `max_depth`: [None, 8, 16]
- `min_samples_leaf`: [1, 5, 10]
- `class_weight`: `"balanced"`

Pipeline: `SimpleImputer(strategy="median")` → `RandomForestClassifier`. No `StandardScaler` (tree models are scale-invariant).

## Outputs

- `artifacts/main/cardio_risk_lgbm.joblib` - main model pipeline.
- `artifacts/baseline/cardio_risk_rf.joblib` - baseline pipeline.
- `reports/metrics_summary.json` - ROC-AUC / PR-AUC / F1 / Brier on test (n≈636) for both models.
- `reports/calibration.png` - reliability diagram on val.
- `reports/shap_summary.png` - global SHAP summary bar + beeswarm on main model.

## GPU notes

None - both LightGBM and scikit-learn RandomForest run on CPU. A typical Optuna 50-trial run completes in ~2-4 min on a modern laptop CPU. No CUDA, no `torch`, no accelerator config.

### Framingham mirror upload (one-time, operator only)

```bash
hf repo create kiselyovd/framingham --repo-type dataset
hf upload kiselyovd/framingham data/raw/framingham.csv --repo-type dataset
```
