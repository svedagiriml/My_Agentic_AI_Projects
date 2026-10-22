# Build from repo root:  docker build -f services/generator/Dockerfile -t generator .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8080 WEB_CONCURRENCY=1
WORKDIR /app
RUN useradd --uid 10001 --create-home app
COPY requirements/ requirements/
RUN pip install -r requirements/generator.txt
COPY common/ common/
COPY services/__init__.py services/__init__.py
COPY services/generator/ services/generator/
USER app
EXPOSE 8080
# async uvicorn worker(s): every request handler is non-blocking, so one worker
# multiplexes many concurrent streams; scale out with Cloud Run instances.
CMD exec uvicorn services.generator.app:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --timeout-keep-alive 75 --no-access-log
