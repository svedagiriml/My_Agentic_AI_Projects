# One image, many services. SERVICE_KIND selects the entrypoint:
#   supervisor | agent (+AGENT_NAME) | mcp (+MCP_SERVER)
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src:/app PORT=8080
RUN useradd --create-home --uid 10001 app
COPY --from=builder /install /usr/local
WORKDIR /app
COPY src ./src
COPY eval ./eval
COPY scripts/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"
ENTRYPOINT ["./entrypoint.sh"]
