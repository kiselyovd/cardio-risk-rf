"""Inference CLI - load a checkpoint and predict on input(s)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

from ..utils import configure_logging, get_logger

log = get_logger(__name__)


def load_model(path: str | Path) -> Any:
    """Load a joblib checkpoint from disk."""
    return joblib.load(path)


def predict(model: Any, features: dict[str, Any]) -> dict[str, Any]:
    """Run inference on a single feature mapping and return pred + class probabilities."""
    import pandas as pd

    x = pd.DataFrame([features]).astype(float)
    proba = model.predict_proba(x)[0].tolist()
    pred = int(model.predict(x)[0])
    return {"pred": pred, "proba": proba}


def main() -> None:
    """CLI entry point - parse args, load model, predict, print JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    configure_logging()
    model = load_model(args.checkpoint)
    result = predict(model, args.input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
