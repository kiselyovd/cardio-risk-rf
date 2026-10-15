.PHONY: setup data train evaluate serve test lint fmt typecheck docker docs publish-hf clean

setup:
	uv sync --all-groups
	uv run pre-commit install

data:
	uv run python -m cardio_risk_rf.data.prepare --raw data/raw/cardio.csv --out data/processed

train:
	uv run python scripts/train_all.py

evaluate:
	uv run python -m cardio_risk_rf.evaluation.evaluate --model artifacts/main/cardio_risk_lgbm.joblib --split data/processed/test.parquet --out reports/metrics.json

serve:
	uv run uvicorn cardio_risk_rf.serving.main:app --reload --port 8000
test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

typecheck:
	uv run mypy src

docker:
	docker compose build

docs:
	uv run mkdocs serve

publish-hf:
	uv run python scripts/publish_to_hf.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml dist build
