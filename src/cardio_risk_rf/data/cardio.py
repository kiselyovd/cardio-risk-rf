"""sulianova Cardiovascular Disease dataset loader + stratified split.

Canonical main-pipeline dataset for cardio-risk-rf. 70000 patients,
binary target `cardio` at ~50/50 positive rate. See
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

Raw columns (source CSV):
    id;age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio

- `id` is dropped on load.
- `age` is converted from **days** (18000-23000) to integer **years**
  (32-65) during load so downstream pipelines, SHAP plots, and the
  serving API all work in a human-readable unit.
- `gender` is left as 1/2 (source encoding) - LightGBM and RandomForest
  treat it as a numeric feature without issue.
- `cholesterol` and `gluc` are ordinal (1=normal, 2=above normal,
  3=well above normal); kept as numeric for trees.
- `smoke`, `alco`, `active` are binary 0/1.
- `ap_hi`/`ap_lo` are systolic/diastolic BP (mmHg).
- Target `cardio` is 0 (no CVD) / 1 (CVD).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES: list[str] = [
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]
TARGET: str = "cardio"


def load_cardio(csv_path: str | Path) -> pd.DataFrame:
    """Read the sulianova Cardiovascular Disease CSV.

    The source file is `;`-separated; some Kaggle forks reshuffle to `,` -
    both are auto-detected. `id` is dropped. `age` is converted from days
    to integer years. Column order is stabilised to FEATURES + [TARGET].
    """
    path = Path(csv_path)
    with path.open("r", encoding="utf-8") as fh:
        first_line = fh.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    df = pd.read_csv(path, sep=sep)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    missing = {TARGET, *FEATURES} - set(df.columns)
    if missing:
        raise ValueError(f"sulianova CSV missing columns: {sorted(missing)}")

    df["age"] = (df["age"] / 365.25).round().astype("int64")
    df = df[[*FEATURES, TARGET]].copy()
    df[TARGET] = df[TARGET].astype("int64")
    return df


def split_stratified(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) with stratification on the target column."""
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio out of range: {train_ratio}")
    if not 0 < val_ratio < 1 - train_ratio:
        raise ValueError(f"val_ratio out of range: {val_ratio}")
    test_ratio = 1.0 - train_ratio - val_ratio

    train_df, rest = train_test_split(
        df,
        test_size=1 - train_ratio,
        stratify=df[TARGET],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        rest,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=rest[TARGET],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
