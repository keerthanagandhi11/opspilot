.PHONY: install run test lint samples docker

install:
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

run:
	.venv/bin/uvicorn app.main:app --app-dir backend --reload --port 8000 --no-proxy-headers

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

samples:
	.venv/bin/python scripts/generate_samples.py

docker:
	docker build -t opspilot . && docker run --rm -p 7860:7860 --env-file .env opspilot
