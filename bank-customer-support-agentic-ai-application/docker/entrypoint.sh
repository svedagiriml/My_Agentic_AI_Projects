#!/bin/sh
set -eu
PORT="${PORT:-8080}"
WORKERS="${WEB_CONCURRENCY:-1}"   # Cloud Run scales instances; keep 1 worker + async concurrency
case "${SERVICE:-supervisor}" in
  supervisor)     APP="supervisor.app.main:app" ;;
  agent)          APP="subagent.app.main:app" ;;        # AGENT_NAME selects the sub-agent spec
  mcp)            APP="mcp_servers.main:app" ;;          # MCP_DOMAIN selects the MCP server
  memory-worker)  APP="memory_worker.app.main:app" ;;
  *) echo "unknown SERVICE=${SERVICE}"; exit 64 ;;
esac
exec uvicorn "$APP" --host 0.0.0.0 --port "$PORT" --workers "$WORKERS" \
     --timeout-keep-alive 65 --proxy-headers --forwarded-allow-ips='*' --no-server-header
