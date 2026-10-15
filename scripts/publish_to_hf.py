"""Upload trained artifacts to HuggingFace Hub.

Adapted from the M2 brain-mri-segmentation template for the M4
cardio-risk-rf tabular classifier. Uploads the LightGBM main artefact
and the RandomForest baseline, renders a Jinja2 model card from
``docs/model_card.md.j2``, and pushes to the HF Hub via ``HfApi``.

Supports ``--dry-run`` to render the model card locally without
touching HuggingFace.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from artifact_manifest import digest, render, validate
from huggingface_hub import HfApi
from jinja2 import Environment, FileSystemLoader

# Filenames shipped to the Hub. Main + baseline artefact.
MODEL_FILES: tuple[str, ...] = (
    "cardio_risk_lgbm.joblib",
    "cardio_risk_rf.joblib",
    "sklearn_model.joblib",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_metrics_table(summary: dict[str, Any]) -> str:
    """Render a Markdown table from the evaluation summary JSON.

    Falls back to ``TBD`` when the summary is empty (Task 18 will
    populate ``reports/metrics_summary.json``).
    """
    if not summary:
        return "TBD"
    scalar = {k: v for k, v in summary.items() if isinstance(v, (int, float, str))}
    if not scalar:
        return "TBD"
    rows = [f"| {k} | {v} |" for k, v in scalar.items()]
    return "| Metric | Value |\n|---|---|\n" + "\n".join(rows)


def _main_metrics_from(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the ``model-index`` metrics entries from the summary.

    Returns an empty list when metrics are unavailable so the Jinja2
    ``{% for m in main_metrics %}`` loop simply emits nothing. The
    plan prefers an empty loop over placeholder numbers.
    """
    if not summary:
        return []
    mapping = {
        "main_roc_auc": "roc_auc",
        "main_pr_auc": "pr_auc",
        "main_f1": "f1",
        "main_brier": "brier",
    }
    out: list[dict[str, Any]] = []
    for key, metric_type in mapping.items():
        if key in summary:
            out.append({"type": metric_type, "value": summary[key]})
    return out


def _widget_payload(widget_path: Path) -> str:
    """Load the sample-patient JSON and return a compact inline literal."""
    if not widget_path.exists():
        return "{}"
    data = json.loads(widget_path.read_text(encoding="utf-8"))
    return json.dumps(data, separators=(", ", ": "))


def _widget_structured(widget_path: Path) -> str:
    """Wrap each field value in a single-element list for HF tabular widget."""
    if not widget_path.exists():
        return "{}"
    data = json.loads(widget_path.read_text(encoding="utf-8"))
    wrapped = {k: [v] for k, v in data.items()}
    return json.dumps(wrapped, separators=(", ", ": "))


def render_model_card(
    template_path: Path,
    out_path: Path,
    **context: Any,
) -> None:
    """Render the Jinja2 model-card template to disk.

    ``autoescape`` is deliberately left at its default (``False``)
    because the template produces Markdown + YAML frontmatter, not
    HTML; HTML-escaping would mangle Markdown table pipes and YAML
    braces. Bandit's B701 warning is suppressed with a targeted
    ``# nosec`` for the same reason.
    """
    env = Environment(  # nosec B701 - renders Markdown/YAML, not HTML
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    tpl = env.get_template(template_path.name)
    out_path.write_text(tpl.render(**context), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish cardio-risk-rf artifacts to HuggingFace Hub.",
    )
    parser.add_argument("--repo-id", default="kiselyovd/cardio-risk-rf")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--metrics", default="reports/metrics_summary.json")
    parser.add_argument("--template", default="docs/model_card.md.j2")
    parser.add_argument("--widget-payload", default="data/widget/sample_patient.json")
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render model card to a local file without uploading to HuggingFace.",
    )
    parser.add_argument(
        "--dry-run-out",
        default="artifacts/model_card_preview.md",
        help="Destination for --dry-run rendered card.",
    )
    args = parser.parse_args()
    manifest, values = validate(Path.cwd())
    if args.repo_id != manifest["hub"]["repo_id"]:
        raise SystemExit("Target differs from reviewed artifact manifest")

    artifacts_dir = Path(args.artifacts)
    summary: dict[str, Any] = {}
    metrics_path = Path(args.metrics)
    if metrics_path.exists():
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))

    widget_payload = _widget_payload(Path(args.widget_payload))
    widget_structured = _widget_structured(Path(args.widget_payload))
    test_size = summary.get("test_size", "TBD")
    template_ctx = {
        "repo_id": args.repo_id,
        "widget_payload": widget_payload,
        "widget_structured": widget_structured,
        "main_metrics": _main_metrics_from(summary),
        "test_size": test_size,
        "metrics_table": _format_metrics_table(summary),
        "artifact_contract": render(manifest, values),
        "artifact_revision": manifest["hub"]["revision"],
        "main_artifact_sha256": next(
            item["sha256"] for item in manifest["hub"]["files"] if item.get("variant") == "main"
        ),
    }

    if args.dry_run:
        out_path = Path(args.dry_run_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        render_model_card(
            template_path=Path(args.template),
            out_path=out_path,
            **template_ctx,
        )
        print(f"[dry-run] Rendered model card to: {out_path}")
        print("[dry-run] Preview (first 40 lines):")
        preview_lines = out_path.read_text(encoding="utf-8").splitlines()[:40]
        for line in preview_lines:
            print(f"  {line}")
        print("[dry-run] No upload performed.")
        return

    if not artifacts_dir.exists():
        raise SystemExit(f"Artifacts dir not found: {artifacts_dir}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Flatten the per-model artifact layout: copy only the joblib
        # files listed in MODEL_FILES from anywhere under artifacts/.
        found: set[str] = set()
        expected = {item["path"]: item["sha256"] for item in manifest["hub"]["files"]}
        for item in artifacts_dir.rglob("*"):
            if item.is_file() and item.name in MODEL_FILES:
                if item.name not in expected or digest(item) != expected[item.name]:
                    raise SystemExit(f"Artifact differs from reviewed manifest: {item.name}")
                dest = tmp_path / item.name
                dest.write_bytes(item.read_bytes())
                found.add(item.name)
        missing = set(MODEL_FILES) - found
        if missing:
            raise SystemExit(f"Missing expected artifacts: {sorted(missing)}")

        render_model_card(
            template_path=Path(args.template),
            out_path=tmp_path / "README.md",
            **template_ctx,
        )

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        if api.whoami()["name"] != "kiselyovd":
            raise SystemExit("HF owner guard failed: expected kiselyovd")
        before = api.model_info(args.repo_id)
        if before.id != manifest["hub"]["repo_id"]:
            raise SystemExit("Unexpected target model repository")
        commit_message = f"Release {args.tag}" if args.tag else "Upload artifacts"
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(tmp_path),
            commit_message=commit_message,
            parent_commit=before.sha,
        )

    print(f"Published to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
