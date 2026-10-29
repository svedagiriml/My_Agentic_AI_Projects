# One hardened image for every service; SERVICE selects the entrypoint at runtime:
#   supervisor | agent | mcp | memory-worker
# Cloud Run sets $PORT (default 8080).
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements/ requirements/
ARG WITH_ML=false
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r requirements/base.txt \
 && if [ "$WITH_ML" = "true" ]; then /opt/venv/bin/pip install -r requirements/ml.txt; fi
COPY libs/ libs/
RUN /opt/venv/bin/pip install ./libs/bankcore

FROM python:3.12-slim AS runtime
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/services PORT=8080 KB_DIR=/app/data/kb
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY services/ services/
COPY data/ data/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod 0555 /entrypoint.sh && chown -R root:root /app && chmod -R a-w /app
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8080\")}/healthz')" || exit 1
ENTRYPOINT ["/entrypoint.sh"]
