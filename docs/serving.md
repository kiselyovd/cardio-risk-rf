# Serving the released models

The API consumes the 11-feature Sulianova schema. `age` is rounded years; the response score refers to `cardio=1` at medical examination.

## Install model files

Use the exact pinned/checksummed bundles described in the [README](https://github.com/kiselyovd/cardio-risk-rf#run-the-published-artifact). The default paths are:

- `artifacts/main/cardio_risk_lgbm.joblib`
- `artifacts/baseline/cardio_risk_rf.joblib`

`CARDIO_MAIN_CKPT` and `CARDIO_BASELINE_CKPT` can point to already-cached verified files. They are operator-controlled paths, not upload fields. Bundles and plain Pipelines are accepted.

```bash
uv run --frozen uvicorn cardio_risk_rf.serving.main:app --port 8000
curl -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' -d @data/widget/sample_patient.json
```

Append `?model=baseline` to compare the RF estimator. Inputs are `age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active`; field descriptions and range validation live in `serving/schemas.py`. Null values use the fitted model's missing-value handling; an all-null record returns 422.

## Response

The JSON contains `probability`, `class`, `threshold`, `shap_top5`, `model_version`, `model_name`, and `request_id`. `CARDIO_MODEL_VERSION` identifies the deployed artifact; `CARDIO_THRESHOLD` defaults to 0.5. SHAP contributions describe fitted-model associations and retain their explainer output scale; they are not causal effects or clinical recommendations.

## Container

`docker compose up api` mounts `./artifacts:/app/artifacts:ro`. Weights are not in the image. Prepare or download the reviewed model files before launch; a successful build alone does not establish inference availability. The service runs on port 8000.

The training target is cross-sectional. The retained Framingham loader/legacy notes do not define this API's released model contract.
