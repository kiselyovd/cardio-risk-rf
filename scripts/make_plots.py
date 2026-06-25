"""Generate ROC and Precision-Recall curves from the real trained models.

Loads the main (LightGBM) and baseline (RandomForest) joblib pipelines from
``artifacts/``, runs ``predict_proba`` on the held-out test split, and renders a
clean 1x2 figure (ROC + PR) to ``reports/roc_pr.png``.

AUROC / PR-AUC printed here are computed from the live predictions and should
match ``reports/metrics.json`` (main ROC-AUC approximately 0.798).

Usage:
    uv run python scripts/make_plots.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
TARGET = "cardio"

MODELS = {
    "LightGBM (main)": ROOT / "artifacts" / "main" / "cardio_risk_lgbm.joblib",
    "RandomForest (baseline)": ROOT / "artifacts" / "baseline" / "cardio_risk_rf.joblib",
}
COLORS = {"LightGBM (main)": "#526CFE", "RandomForest (baseline)": "#9AA3B2"}


def main() -> None:
    test = pd.read_parquet(ROOT / "data" / "processed" / "test.parquet")
    y = test[TARGET].to_numpy()
    x = test.drop(columns=[TARGET])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    for name, path in MODELS.items():
        # joblib artifacts are this project's own trusted, locally-trained models.
        bundle = joblib.load(path)
        model = bundle["model"] if isinstance(bundle, dict) else bundle
        proba = model.predict_proba(x)[:, 1]

        auroc = roc_auc_score(y, proba)
        ap = average_precision_score(y, proba)
        print(f"{name}: ROC-AUC={auroc:.4f}  PR-AUC={ap:.4f}")

        lw = 2.4 if "main" in name else 1.8
        fpr, tpr, _ = roc_curve(y, proba)
        ax_roc.plot(fpr, tpr, label=f"{name} - AUROC {auroc:.3f}", color=COLORS[name], lw=lw)

        prec, rec, _ = precision_recall_curve(y, proba)
        ax_pr.plot(rec, prec, label=f"{name} - PR-AUC {ap:.3f}", color=COLORS[name], lw=lw)

    ax_roc.plot([0, 1], [0, 1], ls="--", color="#C7CCD6", lw=1.2, label="Chance")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curve")
    ax_roc.legend(loc="lower right", frameon=True)
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)

    pos_rate = float(y.mean())
    ax_pr.axhline(pos_rate, ls="--", color="#C7CCD6", lw=1.2, label=f"Baseline ({pos_rate:.2f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall curve")
    ax_pr.legend(loc="lower left", frameon=True)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)

    fig.suptitle(
        "cardio-risk-rf - held-out test (n=10501, balanced)", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()

    out = ROOT / "reports" / "roc_pr.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
