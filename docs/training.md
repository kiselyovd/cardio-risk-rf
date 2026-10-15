# Training

The actual orchestrator is `scripts/train_all.py`. It trains the LightGBM main estimator with 50 Optuna trials and a RandomForest grid with 5-fold CV, using seed 42. Checkpoints are joblib bundles under `artifacts/main` and `artifacts/baseline`.

```bash
uv run --frozen python scripts/fetch_data.py
uv run --frozen python -m cardio_risk_rf.data.prepare --raw data/raw/cardio.csv --out data/processed
uv run --frozen python scripts/train_all.py
```

These commands create a new run. For reproducing published weights' evaluation, use the inference-only [replay](EVALUATION.md). The scaffold Hydra configuration files are retained for reference; the orchestrator's explicit parameters govern its current behavior.
