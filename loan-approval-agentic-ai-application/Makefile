.PHONY: install test test-redis unit integration eval hitl lint run-local stack up down deploy demo

install:        ## install dev dependencies
	pip install -r requirements-dev.txt

lint:
	ruff check src eval tests scripts

unit:
	pytest tests/unit -q

integration:
	pytest tests/integration -q

test:           ## all tests (unit + integration + HTTP microservices)
	pytest -q

test-redis:     ## real-Redis backend tests (needs redis-server on :6390)
	REDIS_TEST_URL=redis://localhost:6390/15 pytest tests/integration/test_redis_backend.py -q

eval:           ## golden-dataset evaluation + quality gate -> eval/reports/
	python -m eval.run_eval

hitl:           ## human evaluation of end results (interactive), then summary
	python -m eval.hitl_review label && python -m eval.hitl_review summarize

run-local:      ## single process, in-memory infra, offline rules LLM
	PYTHONPATH=src:. LLM_CHAIN=$${LLM_CHAIN:-rules} uvicorn loanflow.supervisor.api:app --reload --port 8000

stack:          ## every microservice as its own local process (HTTP transport)
	PYTHONPATH=src:. python scripts/run_local_stack.py

demo:           ## submit the 8 sample applications against a running supervisor
	python scripts/demo.py

up:             ## full docker-compose stack (Redis, Postgres+PGVector, Ollama, Prometheus, 14 services)
	docker compose up --build -d

down:
	docker compose down

deploy:         ## Cloud Run (needs PROJECT_ID, REGION)
	./deploy/cloudrun/deploy.sh
