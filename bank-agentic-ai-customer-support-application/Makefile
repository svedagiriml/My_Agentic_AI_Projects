.PHONY: install test eval eval-online up down ingest token logs
install:      ## dev install
	pip install -r requirements/dev.txt && pip install -e libs/bankcore
test:         ## unit + integration (trajectory) tests, fully offline
	pytest -q
eval:         ## golden dataset, deterministic offline models, with quality gates
	python eval/run_eval.py --offline
eval-online:  ## golden dataset against real free LLMs + LLM-as-judge
	python eval/run_eval.py --online --judge --no-gate
up:           ## start the full local stack
	cp -n .env.example .env || true; docker compose up -d --build
observability:
	docker compose -f docker-compose.observability.yml up -d
down:
	docker compose down
ingest:       ## load the knowledge base into pgvector
	python scripts/ingest_kb.py --dsn postgresql://bank:bank@localhost:5432/bank
token:        ## dev customer token
	@curl -s -X POST localhost:8000/v1/auth/dev-token -H 'content-type: application/json' -d '{"subject":"USR-1001"}'
