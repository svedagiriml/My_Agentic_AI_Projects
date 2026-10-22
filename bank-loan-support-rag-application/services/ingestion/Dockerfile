# Build from repo root:  docker build -f services/ingestion/Dockerfile -t ingestion .
# Runs either as a Cloud Run Job (default CMD = one incremental CDC run) or as a
# Cloud Run service receiving Eventarc GCS events (override CMD with uvicorn).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8080 \
    HF_HOME=/app/.cache/huggingface TOKENIZERS_PARALLELISM=false
WORKDIR /app
RUN useradd --uid 10001 --create-home app
COPY requirements/ requirements/
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements/ingestion.txt
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')" \
 && chown -R app:app /app/.cache
COPY common/ common/
COPY services/__init__.py services/__init__.py
COPY services/ingestion/ services/ingestion/
COPY data/sources.yaml data/sources.yaml
RUN mkdir -p data/state data/markdown data/logs && chown -R app:app data
USER app
CMD ["python", "-m", "services.ingestion.main"]
