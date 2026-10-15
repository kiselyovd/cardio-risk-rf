"""Replay reviewed Hub artifacts on a hash-identified original CSV and split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from ..artifacts import file_sha256, load_pipeline
from ..data.cardio import FEATURES, TARGET, load_cardio, split_stratified
from .metrics import compute_metrics


def replay(
    csv: Path,
    manifest_path: Path,
    out: Path,
    artifact_dir: Path | None = None,
    source_revision: str | None = None,
) -> dict[str, Any]:
    """Record both original estimators, exact row membership and predictions."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["hub"]["repo_id"] != "kiselyovd/cardio-risk-rf" or not re.fullmatch(
        r"[0-9a-f]{40}", manifest["hub"]["revision"]
    ):
        raise ValueError("Replay requires the personal model's immutable reviewed revision")
    if file_sha256(csv) != manifest["dataset"]["csv_sha256"]:
        raise ValueError("CSV differs from the reviewed original source")
    frame = load_cardio(csv)
    frame["_source_row"] = np.arange(len(frame))
    train, val, test = split_stratified(frame, seed=42)
    folds = {
        name: part["_source_row"].astype(int).tolist()
        for name, part in (("train", train), ("val", val), ("test", test))
    }
    canonical = json.dumps(folds, sort_keys=True, separators=(",", ":")) + "\n"
    fold_hash = hashlib.sha256(canonical.encode()).hexdigest()
    report: dict[str, Any] = {
        "status": "replayed_on_reconstructed_original_split",
        "training_membership_verified": False,
        "source_revision": source_revision,
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "csv_sha256": file_sha256(csv),
        "split_rows_sha256": fold_hash,
        "split_sizes": {k: len(v) for k, v in folds.items()},
        "target": "cardio presence at medical examination; no prospective horizon",
        "features": FEATURES,
        "age_unit": "rounded years from source_days/365.25",
        "average_precision_definition": "sklearn.average_precision_score; legacy key pr_auc",
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "scikit-learn", "lightgbm", "joblib")
        },
        "models": {},
    }
    arrays = {"target": test[TARGET].to_numpy(), "source_row": test["_source_row"].to_numpy()}
    for artifact in manifest["hub"]["files"]:
        if artifact["role"] != "weights":
            continue
        tag = artifact["variant"]
        if artifact_dir:
            path = artifact_dir / artifact["path"]
        else:
            from huggingface_hub import hf_hub_download

            path = Path(
                hf_hub_download(
                    manifest["hub"]["repo_id"],
                    artifact["path"],
                    revision=manifest["hub"]["revision"],
                    token=False,
                )
            )
        model = load_pipeline(path, artifact["sha256"])
        if tag == "baseline" and model.named_steps["clf"]._n_samples != len(train):
            raise ValueError("RF training row count differs from reconstructed protocol")
        probs = model.predict_proba(test[FEATURES])[:, 1]
        record = compute_metrics(test[TARGET].to_numpy(), probs, threshold=0.5)
        record.update(
            {
                "average_precision": record["pr_auc"],
                "accuracy": float(accuracy_score(test[TARGET], probs >= 0.5)),
                "confusion_matrix": confusion_matrix(test[TARGET], probs >= 0.5).tolist(),
                "artifact": {
                    "file": artifact["path"],
                    "sha256": artifact["sha256"],
                    "hub_revision": manifest["hub"]["revision"],
                },
            }
        )
        report["models"][tag] = record
        arrays[f"{tag}_probabilities"] = probs
        print(f"Replayed {tag}: n={len(test)}, ROC-AUC={record['roc_auc']:.8f}")
    if set(report["models"]) != {"main", "baseline"}:
        raise ValueError("Manifest must identify the main and baseline weight variants")
    report["artifact"] = report["models"]["main"]["artifact"]
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = out.with_suffix(".npz")
    np.savez_compressed(
        arrays_path,
        target=arrays["target"],
        source_row=arrays["source_row"],
        main_probabilities=arrays["main_probabilities"],
        baseline_probabilities=arrays["baseline_probabilities"],
    )
    report["prediction_arrays"] = {"file": arrays_path.name, "sha256": file_sha256(arrays_path)}
    package_root = Path(__file__).resolve().parents[1]
    report["source_files_sha256"] = {
        p.relative_to(package_root).as_posix(): file_sha256(p)
        for p in sorted(package_root.rglob("*.py"))
    }
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    provenance = out.parent / "provenance"
    provenance.mkdir(exist_ok=True)
    (provenance / "split-rows.json").write_text(
        json.dumps(
            {
                "source_csv_sha256": report["csv_sha256"],
                "records_sha256": fold_hash,
                "splits": folds,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> None:
    """Run inference only; no model training or cloud writes occur."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/raw/cardio.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("artifact-manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/replay-native.json"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    if args.source_revision and not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("source-revision must be an immutable commit")
    replay(args.csv, args.manifest, args.out, args.artifact_dir, args.source_revision)


if __name__ == "__main__":
    main()
