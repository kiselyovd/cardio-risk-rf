"""Tests for sulianova cardio loader + stratified split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cardio_risk_rf.data.cardio import FEATURES, TARGET, load_cardio, split_stratified


def test_original_70000_row_split_preserves_published_checkpoint_protocol() -> None:
    # The released RF records _n_samples=48999. Preserve the legacy sklearn
    # floating test_size/ceil recipe when reconstructing that experiment.
    frame = pd.DataFrame({TARGET: np.tile([0, 1], 35000), "row_id": np.arange(70000)})
    parts = split_stratified(frame, seed=42)
    assert [len(part) for part in parts] == [48999, 10500, 10501]
    rows = [set(part["row_id"]) for part in parts]
    assert not rows[0] & rows[1] and not rows[0] & rows[2] and not rows[1] & rows[2]
    assert len(set.union(*rows)) == 70000


def _toy_df(n: int = 200, pos_rate: float = 0.50, seed: int = 0) -> pd.DataFrame:
    """Synthesize a table with the sulianova schema for splits/loader tests."""
    rng = np.random.default_rng(seed)
    data = {
        "id": np.arange(n),
        "age": rng.integers(14000, 24000, n),  # raw: days
        "gender": rng.integers(1, 3, n),
        "height": rng.uniform(150, 200, n),
        "weight": rng.uniform(45, 130, n),
        "ap_hi": rng.integers(90, 200, n),
        "ap_lo": rng.integers(60, 120, n),
        "cholesterol": rng.integers(1, 4, n),
        "gluc": rng.integers(1, 4, n),
        "smoke": rng.integers(0, 2, n),
        "alco": rng.integers(0, 2, n),
        "active": rng.integers(0, 2, n),
        TARGET: (rng.random(n) < pos_rate).astype(int),
    }
    return pd.DataFrame(data)


def test_load_cardio_returns_expected_columns_and_age_in_years(tmp_path) -> None:
    df = _toy_df(100)
    path = tmp_path / "cardio.csv"
    df.to_csv(path, index=False)
    loaded = load_cardio(path)
    assert list(loaded.columns) == [*FEATURES, TARGET]
    assert "id" not in loaded.columns
    assert len(loaded) == 100
    assert loaded[TARGET].dtype == np.int64
    # age converted days → years; raw 14000-24000 → 38-66
    assert loaded["age"].min() >= 30
    assert loaded["age"].max() <= 80


def test_split_stratified_respects_ratios_and_positives() -> None:
    df = _toy_df(n=1000, pos_rate=0.50).drop(columns=["id"])
    train, val, test = split_stratified(df, seed=42)
    total = len(train) + len(val) + len(test)
    assert total == len(df)
    assert abs(len(train) / total - 0.70) < 0.01
    assert abs(len(val) / total - 0.15) < 0.01
    assert abs(len(test) / total - 0.15) < 0.01
    for part in (train, val, test):
        share = part[TARGET].mean()
        assert abs(share - 0.50) < 0.04, f"stratification broken: {share}"


def test_split_no_row_leak() -> None:
    df = _toy_df(n=500).drop(columns=["id"])
    df["row_id"] = range(len(df))
    train, val, test = split_stratified(df, seed=42)
    seen = set()
    for part in (train, val, test):
        for rid in part["row_id"].tolist():
            assert rid not in seen
            seen.add(rid)
    assert len(seen) == len(df)
