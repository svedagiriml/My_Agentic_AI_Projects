.PHONY: install test lint pdfs ingest up down eval report deploy
install:   ## local dev env
	pip install -r requirements/dev.txt -r requirements/retriever.txt -r requirements/ingestion.txt -r requirements/generator.txt -r requirements/gateway.txt
test:
	pytest
lint:
	ruff check .
pdfs:      ## download the knowledge-base PDFs
	python scripts/download_pdfs.py
ingest:    ## incremental CDC ingestion into Pinecone
	docker compose run --rm ingestion
up:
	docker compose up -d --build
down:
	docker compose down
eval:      ## RAGAS evaluation against running services
	pip install -r requirements/eval.txt && python -m evaluation.run_ragas
report:    ## cost / latency / cache report from /metrics
	python scripts/monitoring_report.py
deploy:    ## build + deploy all Cloud Run services
	bash deploy/gcp/02_build_and_deploy.sh && bash deploy/gcp/03_ingestion_cdc.sh
