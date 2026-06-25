# Serving

The main and baseline models are exposed behind a single FastAPI app (`cardio_risk_rf.serving.main:app`). Both are loaded lazily from `artifacts/{main,baseline}/*.joblib` and selected via a query parameter.

## Run

```bash
# local dev
uv run uvicorn cardio_risk_rf.serving.main:app --host 0.0.0.0 --port 8000

# docker
docker run --rm -p 8000:8000 ghcr.io/kiselyovd/cardio-risk-rf:v0.1.0
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe - returns `{"status": "ok", "version": <model_version>}`. |
| `POST` | `/predict` | Score a single patient. Query param `model` selects `main` (default) or `baseline`. |

`/predict` returns HTTP 422 if every feature is `null`, and HTTP 503 if the joblib checkpoint is missing from disk.

## Request - `PatientFeatures`

Schema source: `cardio_risk_rf.serving.schemas.PatientFeatures`. All fields are **optional** - missing values are forwarded as `NaN` into the pipeline (LightGBM main handles this natively; the baseline imputes with the train-set median).

```python
class PatientFeatures(BaseModel):
    male: int | None              # 0 or 1
    age: int | None               # years, 0-120
    education: float | None       # 1-4 (Framingham encoding)
    currentSmoker: int | None     # 0 or 1
    cigsPerDay: float | None      # cigarettes per day
    BPMeds: float | None          # 0 or 1 - on BP medication
    prevalentStroke: int | None   # 0 or 1
    prevalentHyp: int | None      # 0 or 1 - prevalent hypertension
    diabetes: int | None          # 0 or 1
    totChol: float | None         # total cholesterol (mg/dL)
    sysBP: float | None           # systolic BP (mmHg)
    diaBP: float | None           # diastolic BP (mmHg)
    BMI: float | None             # kg/m²
    heartRate: float | None       # bpm
    glucose: float | None         # mg/dL
```

## Response - `PredictionResponse`

```python
class ShapEntry(BaseModel):
    feature: str
    value: float | int | None
    shap: float


class PredictionResponse(BaseModel):
    probability: float            # P(TenYearCHD=1)
    cls: int                      # 0 or 1, serialised as "class"
    threshold: float              # decision threshold used (default 0.5)
    shap_top5: list[ShapEntry]    # 5 features with largest |SHAP| on this request
    model_version: str            # e.g. "v0.1.0"
    model_name: str               # "cardio_risk_lgbm" or "cardio_risk_rf"
    request_id: str               # 12-char UUID prefix for tracing
```

## curl example

```bash
curl -X POST 'localhost:8000/predict?model=main' \
  -H 'content-type: application/json' \
  -d @data/widget/sample_patient.json
```

Using the main model on the bundled sample patient, a typical response looks like:

```json
{
  "probability": 0.18,
  "class": 0,
  "threshold": 0.5,
  "shap_top5": [
    {"feature": "age", "value": 52, "shap": 0.42},
    {"feature": "sysBP", "value": 138, "shap": 0.21},
    {"feature": "cigsPerDay", "value": 10, "shap": 0.14},
    {"feature": "totChol", "value": 220, "shap": -0.08},
    {"feature": "BMI", "value": 24.1, "shap": -0.05}
  ],
  "model_version": "v0.1.0",
  "model_name": "cardio_risk_lgbm",
  "request_id": "a1b2c3d4e5f6"
}
```

To score the same patient with the baseline RandomForest, pass `?model=baseline`.

## Configuration

Environment variables read by `cardio_risk_rf.serving.routes`:

| Variable | Default | Purpose |
|---|---|---|
| `CARDIO_MAIN_CKPT` | `artifacts/main/cardio_risk_lgbm.joblib` | Main model path. |
| `CARDIO_BASELINE_CKPT` | `artifacts/baseline/cardio_risk_rf.joblib` | Baseline model path. |
| `CARDIO_MODEL_VERSION` | `v0.1.0` | Reported in `/health` and response body. |
| `CARDIO_THRESHOLD` | `0.5` | Decision threshold applied to `probability`. |
