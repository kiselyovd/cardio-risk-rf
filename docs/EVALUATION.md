# Data, artifacts and replay contract

The released files were inspected after SHA256 verification. Both are dict bundles with `model`, `best_params`, `cv_pr_auc`. The main Pipeline contains LGBMClassifier; the RF Pipeline contains median imputation and RandomForestClassifier. Both expose the same 11 input features, and the RF records 48,999 training rows.

## Exact source and split

The [original Sulianova CSV](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) is SHA256 `21a705d23381b0dfd6a6416da701b490744f1fc3b47e9ff3db3968c420ffa10c`, extracted from archive SHA256 `b78ff607480810c3e27de3e11d52c4962a3a81dbb3d69365e1cf3394a67c2b23` (dataset 107706/version256873).

The seed 42 stratified protocol requests 70/15/15 via float test sizes. Sklearn's ceiling gives 48,999/10,500/10,501 rows. This matches the saved RF's training count. The recovered recipe and source order are recorded; historical checkpoint training row IDs themselves remain unrecorded.

[Source receipt](https://github.com/kiselyovd/cardio-risk-rf/blob/main/reports/provenance/source.json) · [Exact row split](https://github.com/kiselyovd/cardio-risk-rf/blob/main/reports/provenance/split-rows.json) · [Replay](https://github.com/kiselyovd/cardio-risk-rf/blob/main/reports/replay-native.json).

## Inputs and target

`age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active`.

Age is rounded years after dividing source days by 365.25. Height is cm, weight kg, blood pressures mmHg. Other codes retain source encoding: gender 1/2; cholesterol/glucose 1/2/3; smoking/alcohol/activity 0/1. The model's binary class 1 is the source `cardio` label at medical examination. No prospective time horizon is attached.

The evaluator reports ROC-AUC, sklearn average precision (the legacy `pr_auc` key), F1/Brier and confusion counts at threshold 0.5. Main and RF use the same reconstructed test rows. Results describe this cohort; they are not a clinical validation or a causal feature analysis.

## Model file formats

Canonical `cardio_risk_lgbm.joblib` and `cardio_risk_rf.joblib` preserve their original bytes. `sklearn_model.joblib` is an explicit plain-Pipeline alias of the main estimator, with separate SHA256 and prediction-parity evidence. [Payload contract](https://github.com/kiselyovd/cardio-risk-rf/blob/main/reports/provenance/payload-contract.json) · [Alias verification](https://github.com/kiselyovd/cardio-risk-rf/blob/main/reports/provenance/alias.json).

The alias serialization matches the canonical model on every 10,501 test row, with zero probability difference. It does not create a second trained model. The source dataset license is Unknown in publisher metadata; code MIT is a separate component declaration.

## Repeat

```bash
uv sync --frozen --all-groups
uv run --frozen python scripts/fetch_data.py
uv run --frozen python -m cardio_risk_rf.evaluation.replay --csv data/raw/cardio.csv
```

For already-downloaded canonical files, pass `--artifact-dir /path/to/model-files`. The runner checks CSV and model hashes, records source/package identities and stores anonymized probability/target arrays. Fresh preparation writes train/val/test Parquet plus a split manifest into an empty destination.

Code fingerprints normalize CRLF to LF for portable source comparisons. Model, CSV and array digests always cover exact raw bytes.
