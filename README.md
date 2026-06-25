# cardio-risk-rf

[![CI](https://img.shields.io/github/actions/workflow/status/kiselyovd/cardio-risk-rf/test.yml?branch=main&style=for-the-badge&label=CI&logo=github)](https://github.com/kiselyovd/cardio-risk-rf/actions/workflows/test.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-526CFE?style=for-the-badge&logo=materialformkdocs&logoColor=white)](https://kiselyovd.github.io/cardio-risk-rf/)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/kiselyovd/cardio-risk-rf/badges/coverage.json&style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/kiselyovd/cardio-risk-rf/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![HF Hub](https://img.shields.io/badge/🤗%20HF%20Hub-model-FFD21E?style=for-the-badge)](https://huggingface.co/kiselyovd/cardio-risk-rf)

![cardio-risk-rf banner](docs/images/banner.jpg)

Production-grade tabular cardiovascular-risk classifier on the Framingham Heart Study (4240 patients, 10-year CHD). Main model **LightGBM** with native NaN handling and SHAP explainability; baseline **RandomForest** with median imputation. Hydra-configured, Optuna-tuned, evaluated with ROC-AUC / PR-AUC / F1 / Brier + calibration plot, served by FastAPI as `/predict` with local SHAP top-5, distributed through Hugging Face Hub and MkDocs Material.

> **Part of the [kiselyovd ML portfolio](https://github.com/kiselyovd#ml-portfolio)** - production-grade ML projects sharing one [cookiecutter template](https://github.com/kiselyovd/ml-project-template).

📖 [English docs](https://kiselyovd.github.io/cardio-risk-rf/) • 🇷🇺 [Русский README](README.ru.md) • 🤗 [HF Hub model](https://huggingface.co/kiselyovd/cardio-risk-rf)

## Dataset

[sulianova Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset). 70 000 rows × 11 clinical features, **balanced** target `cardio` (50/50). Features: age, gender, height/weight, systolic/diastolic BP, cholesterol & glucose (ordinal 1-2-3), smoking/alcohol/activity. Fetched by `scripts/sync_data.sh` from Kaggle with an HF Datasets fallback mirror. Split 70/15/15 stratified by target (`train=48999, val=10500, test=10501`). The original [Framingham Heart Study](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset) (4240 rows, 10-year prospective CHD target) is kept as a secondary cohort in `notebooks/02_benchmark.ipynb`.

The original 49-row dataset from the `coro-detect` author is archived at `docs/legacy/original_dataset.csv` - see `docs/legacy/README.md` for rationale.

## Results

Filled in from `reports/metrics_summary.json` once the v0.1.0 run completes.

| Model | ROC-AUC | PR-AUC | F1 @ 0.5 | F1 @ t\* | Brier | t\* |
|---|---|---|---|---|---|---|
| **LightGBM** (main) | **79.8%** | **78.4%** | 71.7% | **73.9%** | 0.182 | 0.35 |
| RandomForest (baseline) | 79.5% | 77.9% | 70.8% | 73.2% | 0.184 | 0.41 |

Metrics on held-out test split (n=10 501, balanced target, sulianova Cardiovascular Disease Dataset). F1 reported at both the default 0.5 threshold and at the validation-set optimal threshold t\*. Calibration plot on val → `reports/calibration.png`; F1-optimal thresholds saved to `reports/metrics_thresholded.json`.

## Visualizations

![ROC and PR curves](docs/images/roc_pr.png)

ROC and Precision-Recall curves on the held-out test split - LightGBM (main) vs RandomForest (baseline).

![SHAP summary](docs/images/shap_summary.png)

Global SHAP summary for the main LightGBM model - per-feature impact on the 10-year CHD risk prediction.

![Calibration curve](docs/images/calibration.png)

Reliability (calibration) curve on the validation split - predicted probability vs observed frequency.

## Quick Start

```bash
uv sync --all-groups
bash scripts/sync_data.sh
uv run python -m cardio_risk_rf.data.prepare --raw data/raw/framingham.csv --out data/processed
uv run python scripts/train_all.py
```

## Serving

```bash
docker run --rm -p 8000:8000 ghcr.io/kiselyovd/cardio-risk-rf:v0.1.0
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d @data/widget/sample_patient.json
```

## License

MIT - see [LICENSE](LICENSE). Dataset under CC-BY-4.0 (Framingham Heart Study).
