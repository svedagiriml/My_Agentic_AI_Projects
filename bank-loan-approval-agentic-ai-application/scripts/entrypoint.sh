#!/bin/sh
set -e
case "$SERVICE_KIND" in
  supervisor) exec uvicorn loanflow.supervisor.api:app --host 0.0.0.0 --port "$PORT" --timeout-graceful-shutdown 25 ;;
  agent)      exec uvicorn loanflow.agents.service:app --host 0.0.0.0 --port "$PORT" --timeout-graceful-shutdown 25 ;;
  mcp)        exec uvicorn "loanflow.mcp_servers.${MCP_SERVER}_server:app" --host 0.0.0.0 --port "$PORT" ;;
  *) echo "SERVICE_KIND must be supervisor|agent|mcp" >&2; exit 1 ;;
esac
