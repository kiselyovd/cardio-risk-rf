# Architecture

Two independent `sklearn.Pipeline` artefacts sharing the same `{prob, class, shap_top5}` contract.

```mermaid
flowchart LR
  CSV[Framingham CSV]:::external --> Prep[prepare.py<br/>70/15/15 stratified]:::code
  Prep --> Train[train.parquet]:::data
  Prep --> Val[val.parquet]:::data
  Prep --> Test[test.parquet]:::data
  Train --> LGBM[LightGBM<br/>+ Optuna 50]:::model
  Train --> RF[RandomForest<br/>+ GridSearchCV]:::model
  Val --> LGBM
  Val --> Calib[Calibration plot]:::artifact
  LGBM --> MainArt[artifacts/main/*.joblib]:::artifact
  RF --> BaseArt[artifacts/baseline/*.joblib]:::artifact
  MainArt --> API[FastAPI /predict]:::serve
  BaseArt --> API
  MainArt --> SHAP[Global SHAP<br/>reports/*]:::serve
  Test --> Score[compute_metrics]:::code
  MainArt --> Score
  BaseArt --> Score

  classDef external fill:#FFEBEE,stroke:#E53935,color:#B71C1C
  classDef data fill:#FFCDD2,stroke:#E53935,color:#B71C1C
  classDef code fill:#EF9A9A,stroke:#C62828,color:#B71C1C
  classDef model fill:#EF5350,stroke:#B71C1C,color:#fff
  classDef artifact fill:#E53935,stroke:#B71C1C,color:#fff
  classDef serve fill:#C62828,stroke:#B71C1C,color:#fff
```

Key design decisions:
- Main model uses **LightGBM native NaN handling** - no imputer in its pipeline. Missing values at inference (including `/predict`) are forwarded as-is.
- Baseline uses `SimpleImputer(strategy="median")` because RandomForest cannot split on NaN. No `StandardScaler` (tree models are scale-invariant).
- Class imbalance handled by `scale_pos_weight = N_neg / N_pos` (LGBM) and `class_weight="balanced"` (RF).
- Hyperparameter search: Optuna 50 trials for LGBM (TPE sampler, seed=42, early-stopping on val), GridSearchCV short grid for RF.
- Final metric table uses **test set only**; calibration plot uses **val** to avoid test-set contamination.
