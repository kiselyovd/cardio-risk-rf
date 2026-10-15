"""Export a verified plain-Pipeline alias for Hugging Face sklearn clients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cardio_risk_rf.artifacts import export_pipeline_alias


def main() -> None:
    """Preserve the canonical bundle and export sklearn_model.joblib separately."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, default=Path("artifacts/main/cardio_risk_lgbm.joblib"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/hf_export"))
    args = parser.parse_args()
    receipt = export_pipeline_alias(args.main, args.out / "sklearn_model.joblib")
    (args.out / "alias-provenance.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
