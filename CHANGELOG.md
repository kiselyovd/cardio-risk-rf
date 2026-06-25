# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-04-17

Initial release - Milestone 4 of the `kiselyovd` ML portfolio refactor.

### Added

- **Main dataset:** [sulianova Cardiovascular Disease](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) - 70 000 patients × 11 clinical features, **balanced** target `cardio` at 50/50. Features: age (years), gender, height, weight, systolic/diastolic BP (mmHg), cholesterol & glucose (ordinal 1-2-3), smoking/alcohol/activity flags.
- **Main model:** LightGBM classifier tuned with Optuna (50 trials, TPE sampler, early stopping on val). Native NaN handling. `scale_pos_weight` for mild imbalance.
- **Baseline model:** RandomForest classifier with `SimpleImputer(strategy="median")` + `GridSearchCV` (5-fold stratified), `class_weight="balanced"`.
- **Secondary benchmark cohort:** Framingham Heart Study 10-Year CHD (4 240 rows, prospective prediction target) - kept at `data/raw/framingham.csv`, used in `notebooks/02_benchmark.ipynb` for cross-cohort comparison.
- **Data acquisition:** `scripts/sync_data.sh` fetches sulianova from Kaggle (`sulianova/cardiovascular-disease-dataset`) with an HF Datasets fallback mirror. Supports `~/.kaggle/kaggle.json`, `KAGGLE_USERNAME`/`KAGGLE_KEY` and `KAGGLE_API_TOKEN` env-var auth.
- **Evaluation:** ROC-AUC / PR-AUC / F1 / Brier + calibration plot on val, F1-optimal threshold search (`reports/metrics_thresholded.json`).
- **Explainability:** SHAP global summary (beeswarm + importance CSV) on val; local `shap_top5` on each `/predict` call. Top drivers identified: `ap_hi` (systolic BP), `age`, `cholesterol`, `weight`, `ap_lo`.
- **Serving:** FastAPI `POST /predict?model={main|baseline}` returning `{probability, class, threshold, shap_top5, model_version, model_name, request_id}`; `GET /health`; `GET /metrics` (Prometheus). Null values in input are imputed (main: LGBM native NaN, baseline: median imputer); input coerced to `float` dtype.
- **Docker:** multi-stage `python:3.13-slim` runtime, no torch / CUDA.
- **CI/CD:** matrix pytest (3.12 + 3.13), ruff + mypy + deptry + bandit + interrogate + actionlint + codespell, MkDocs Material auto-deploy, GHCR docker push on tag, GitHub Release with auto-generated notes.
- **HF Hub:** `kiselyovd/cardio-risk-rf` publishing via `scripts/publish_to_hf.py` with Jinja-rendered model card (pills, widget, `model-index`).
- **Quality gates:** full backport from M2 (brain-mri-segmentation) - pinned tool versions, pre-commit hooks, interrogate ≥30%.
- **Notebooks:** `01_eda.ipynb` (distribution, correlations, missing-value comparison), `02_benchmark.ipynb` (cross-cohort: sulianova vs Framingham with same pipeline).
- **Legacy archival:** original 49-row biomarker dataset preserved at `docs/legacy/original_dataset.csv` with provenance note.

### Metrics (held-out test, n=10 501, balanced 50/50)

| Model | ROC-AUC | PR-AUC | F1 @ 0.5 | F1 @ t\* | Brier | t\* |
|---|---|---|---|---|---|---|
| **LightGBM** (main) | 79.8% | 78.1% | 71.9% | 73.8% | 0.182 | 0.33 |
| RandomForest (baseline) | 79.5% | 77.9% | 70.8% | 73.2% | 0.184 | 0.41 |

### Design decisions

- Swapped the original 49-row `coro-detect` biomarker CSV and the initially-selected Framingham dataset (4 240 rows, ROC-AUC 0.66) for sulianova (70 000 rows) after brainstorming - the larger balanced dataset yields practically meaningful F1 numbers without `scale_pos_weight` tricks, and the classical `ap_hi/age/cholesterol` feature set is clinically familiar to README readers.
- Switched from a single `framingham.py` loader to a dedicated `cardio.py` canonical loader, with `framingham.py` retained for the secondary benchmark notebook.

### Known limitations

- sulianova `cardio` is a cross-sectional diagnosis (did the patient have CVD at examination), **not** a prospective 10-year risk - interpret downstream accordingly. `/predict` probability ≠ future incidence probability.
- Docker image is ~900 MB (tabular deps including matplotlib, optuna, shap, pandas, pyarrow). A follow-up will split base vs. training deps for a slim <250 MB serving image.
- Training uses only sulianova native columns; no feature engineering (BMI-from-HW, pulse pressure from `ap_hi-ap_lo`, etc.) - left for follow-up to keep v0.1.0 focused.
