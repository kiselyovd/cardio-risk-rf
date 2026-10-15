"""CLI to produce data/processed/{train,val,test}.parquet from raw CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .cardio import load_cardio, split_stratified


def prepare_dataset(raw: Path, out: Path, seed: int = 42) -> dict[str, Any]:
    """Prepare exact legacy split membership without mixing existing destination files."""
    if out.exists() and any(p.name != ".gitkeep" for p in out.iterdir()):
        raise ValueError("Processed destination must be empty; existing files are preserved")
    df = load_cardio(raw)
    df["_source_row"] = range(len(df))
    train_df, val_df, test_df = split_stratified(df, seed=seed)
    frames = {"train": train_df, "val": val_df, "test": test_df}
    rows = {name: frame["_source_row"].astype(int).tolist() for name, frame in frames.items()}
    serialized = json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n"
    with raw.open("rb") as stream:
        csv_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "source_csv_sha256": csv_sha256,
        "protocol": "legacy sklearn float test_size ceil; requested 70/15/15",
        "sizes": {name: len(frame) for name, frame in frames.items()},
        "age_transform": "round(source_days / 365.25) integer years",
        "records_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "splits": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.drop(columns=["_source_row"]).to_parquet(out / f"{name}.parquet", index=False)
    (out / "split-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Prepared source={csv_sha256}, sizes={manifest['sizes']}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw/cardio.csv")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    prepare_dataset(Path(args.raw), Path(args.out), seed=args.seed)


if __name__ == "__main__":
    main()
