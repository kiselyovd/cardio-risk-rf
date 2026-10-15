#!/usr/bin/env bash
# Download the original, checksum-verified Sulianova CSV into the canonical path.
set -euo pipefail
ROOT="${CARDIO_REPO_ROOT:-$(pwd)}"
cd "$ROOT"
uv run python scripts/fetch_data.py --out data/raw/cardio.csv
