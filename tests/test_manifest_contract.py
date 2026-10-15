"""The published score sources must reproduce from the saved native predictions."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from cardio_risk_rf.evaluation.replay import source_sha256

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "artifact_manifest", ROOT / "scripts/artifact_manifest.py"
)
assert SPEC and SPEC.loader
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def test_source_fingerprints_are_portable_across_checkout_line_endings(tmp_path: Path) -> None:
    unix, windows = tmp_path / "unix.py", tmp_path / "windows.py"
    unix.write_bytes(b"value = 1\n")
    windows.write_bytes(b"value = 1\r\n")
    assert source_sha256(unix) == source_sha256(windows)
    assert contract.digest(unix) != contract.digest(windows)


def test_native_scores_match_their_saved_arrays() -> None:
    manifest, _ = contract.validate(ROOT)
    report = json.loads((ROOT / manifest["evaluation"]["report"]).read_text())
    path = ROOT / "reports" / report["prediction_arrays"]["file"]
    assert contract.digest(path) == report["prediction_arrays"]["sha256"]
    with np.load(path, allow_pickle=False) as arrays:
        target = arrays["target"]
        assert len(target) == 10501
        for name in ("main", "baseline"):
            probs = arrays[f"{name}_probabilities"]
            actual = report["models"][name]
            assert actual["roc_auc"] == roc_auc_score(target, probs)
            assert actual["average_precision"] == average_precision_score(target, probs)
            assert actual["f1"] == f1_score(target, probs >= 0.5)


def test_missing_or_changed_report_cannot_pass_the_gate(tmp_path: Path) -> None:
    manifest = json.loads((ROOT / "artifact-manifest.json").read_text())
    (tmp_path / "artifact-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError):
        contract.validate(tmp_path)
    path = tmp_path / manifest["evaluation"]["report"]
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="report SHA256"):
        contract.validate(tmp_path)
