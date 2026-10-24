"""Launch every microservice locally as its own process (no Docker needed).

  python scripts/run_local_stack.py            # MCP servers :9101-9105, agents :8101-8107, supervisor :8000
  python scripts/run_local_stack.py --no-supervisor --base-mcp 19101 --base-agent 18101

This mirrors the Cloud Run topology: 5 MCP services + 7 agent services + 1 supervisor,
all talking over HTTP with internal auth (TRANSPORT_MODE=http).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MCP = ["document", "credit_bureau", "risk_engine", "compliance", "reporting"]
AGENTS = ["fetch_document", "validate_document", "check_credit", "assess_risk", "check_compliance",
          "compile_report", "compile_rejection"]
MCP_MODULE = {"document": "document_server", "credit_bureau": "credit_bureau_server", "risk_engine": "risk_engine_server",
              "compliance": "compliance_server", "reporting": "reporting_server"}


def stack_env(base_mcp: int, base_agent: int) -> dict[str, str]:
    return {
        "TRANSPORT_MODE": "http",
        "MCP_URLS": json.dumps({n: f"http://127.0.0.1:{base_mcp + i}" for i, n in enumerate(MCP)}),
        "AGENT_URLS": json.dumps({n: f"http://127.0.0.1:{base_agent + i}" for i, n in enumerate(AGENTS)}),
    }


def _uvicorn(module: str, port: int, env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", module, "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}", **env},
    )


def _wait(url: str, timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    raise RuntimeError(f"service did not become healthy: {url}")


def start(base_mcp: int = 9101, base_agent: int = 8101, supervisor_port: int | None = 8000) -> list[subprocess.Popen]:
    env = stack_env(base_mcp, base_agent)
    procs = [_uvicorn(f"loanflow.mcp_servers.{MCP_MODULE[n]}:app", base_mcp + i, {**env, "SERVICE_NAME": f"mcp-{n}"})
             for i, n in enumerate(MCP)]
    procs += [_uvicorn("loanflow.agents.service:app", base_agent + i, {**env, "AGENT_NAME": n, "SERVICE_NAME": f"agent-{n}"})
              for i, n in enumerate(AGENTS)]
    if supervisor_port:
        procs.append(_uvicorn("loanflow.supervisor.api:app", supervisor_port, {**env, "SERVICE_NAME": "supervisor"}))
    urls = [f"http://127.0.0.1:{base_mcp + i}/healthz" for i in range(len(MCP))]
    urls += [f"http://127.0.0.1:{base_agent + i}/healthz" for i in range(len(AGENTS))]
    if supervisor_port:
        urls.append(f"http://127.0.0.1:{supervisor_port}/healthz")
    try:
        for u in urls:
            _wait(u)
    except Exception:
        stop(procs)
        raise
    return procs


def stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        p.send_signal(signal.SIGTERM)
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-mcp", type=int, default=9101)
    ap.add_argument("--base-agent", type=int, default=8101)
    ap.add_argument("--supervisor-port", type=int, default=8000)
    ap.add_argument("--no-supervisor", action="store_true")
    a = ap.parse_args()
    ps = start(a.base_mcp, a.base_agent, None if a.no_supervisor else a.supervisor_port)
    print(f"stack up: {len(ps)} services. Supervisor: http://127.0.0.1:{a.supervisor_port}/docs  (Ctrl-C to stop)")
    try:
        signal.pause()
    except KeyboardInterrupt:
        pass
    finally:
        stop(ps)
