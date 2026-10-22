# Build from repo root:  docker build -f services/retriever/Dockerfile -t retriever .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8080 WEB_CONCURRENCY=1 \
    HF_HOME=/app/.cache/huggingface TOKENIZERS_PARALLELISM=false
WORKDIR /app
RUN useradd --uid 10001 --create-home app
COPY requirements/ requirements/
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements/retriever.txt
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
# Bake the free embedding model into the image: no download on cold start.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')" \
 && chown -R app:app /app/.cache
COPY common/ common/
COPY services/__init__.py services/__init__.py
COPY services/retriever/ services/retriever/
USER app
EXPOSE 8080
CMD exec uvicorn services.retriever.app:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --timeout-keep-alive 75 --no-access-log
