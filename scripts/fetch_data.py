"""Retrieve and verify the original public Sulianova source without credentials."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

SOURCE = "https://www.kaggle.com/api/v1/datasets/download/sulianova/cardiovascular-disease-dataset"
ARCHIVE_SHA256 = "b78ff607480810c3e27de3e11d52c4962a3a81dbb3d69365e1cf3394a67c2b23"
CSV_SHA256 = "21a705d23381b0dfd6a6416da701b490744f1fc3b47e9ff3db3968c420ffa10c"


def verified_csv(archive: bytes) -> bytes:
    """Reject a changed archive or unexpected CSV payload before writing any data."""
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Original source archive changed; review its new provenance")
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        data = source.read("cardio_train.csv")
    if hashlib.sha256(data).hexdigest() != CSV_SHA256:
        raise ValueError("Original CSV hash mismatch")
    return data


def main() -> None:
    """Keep the historical local name cardio.csv while recording its original identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/raw/cardio.csv"))
    args = parser.parse_args()
    if args.out.exists():
        if hashlib.sha256(args.out.read_bytes()).hexdigest() != CSV_SHA256:
            raise ValueError("Existing data preserved: its hash differs from the original source")
        print("Original CSV already present and hash-verified")
        return
    # Fixed HTTPS publisher URL; no caller-controlled protocols or local-file access.
    with urllib.request.urlopen(SOURCE, timeout=90) as response:  # nosec B310
        data = verified_csv(response.read())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    receipt = {
        "source": SOURCE,
        "archive_sha256": ARCHIVE_SHA256,
        "csv_sha256": CSV_SHA256,
        "license_status": "Unknown (publisher metadata)",
        "target": "cardio at medical examination; no future time horizon",
    }
    args.out.with_suffix(".source.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Verified original CSV: {CSV_SHA256}")


if __name__ == "__main__":
    main()
