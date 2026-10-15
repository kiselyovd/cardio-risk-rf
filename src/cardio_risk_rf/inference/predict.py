"""Inference CLI - load a checkpoint and predict on input(s)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..artifacts import load_pipeline
from ..data.cardio import FEATURES
from ..utils import configure_logging, get_logger

log = get_logger(__name__)


def load_model(path: str | Path) -> Any:
    """Load a joblib checkpoint from disk."""
    return load_pipeline(path)


def predict(model: Any, features: dict[str, Any]) -> dict[str, Any]:
    """Run inference on a single feature mapping and return pred + class probabilities."""
    import pandas as pd

    unknown = set(features) - set(FEATURES)
    if unknown:
        raise ValueError(f"Unknown cardio features: {sorted(unknown)}")
    if not any(value is not None for value in features.values()):
        raise ValueError("At least one feature is required")
    if features.get("age") is not None and not 0 <= float(features["age"]) <= 120:
        raise ValueError("Model age is in rounded years, not source days")
    x = pd.DataFrame([{name: features.get(name) for name in FEATURES}]).astype(float)
    proba = model.predict_proba(x)[0].tolist()
    pred = int(model.predict(x)[0])
    return {"pred": pred, "proba": proba}


def main() -> None:
    """CLI entry point - parse args, load model, predict, print JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "--checkpoint", dest="checkpoint", required=True)
    parser.add_argument("--input", required=True, help="JSON object or path to a JSON feature file")
    args = parser.parse_args()
    configure_logging()
    model = load_model(args.checkpoint)
    text = (
        args.input if args.input.lstrip().startswith("{") else Path(args.input).read_text("utf-8")
    )
    features = json.loads(text)
    if not isinstance(features, dict):
        raise ValueError("Input must be a JSON object of cardio features")
    result = predict(model, features)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
