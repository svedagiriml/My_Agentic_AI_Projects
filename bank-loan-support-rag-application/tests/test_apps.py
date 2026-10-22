"""HTTP-level smoke tests of the FastAPI apps that need no external services."""
from fastapi.testclient import TestClient

from common.app_factory import create_app
from common.config import Settings


def test_guardrail_http_contract():
    from services.guardrail.app import app

    with TestClient(app) as c:
        r = c.post("/check", json={"text": "my pan is ABCPE1234F", "direction": "input"})
        assert r.status_code == 200 and "<PAN_1>" in r.json()["sanitized_text"]
        bad = c.post("/check", json={"text": "x", "direction": "sideways"})
        assert bad.status_code == 422                       # Pydantic rejects invalid enum
        extra = c.post("/check", json={"text": "x", "direction": "input", "admin": True})
        assert extra.status_code == 422                     # extra="forbid"
        assert c.get("/metrics").status_code == 200


def test_per_client_throttling_returns_429():
    app = create_app("t", Settings(), throttle_per_minute=2)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    with TestClient(app) as c:
        codes = [c.get("/ping").status_code for _ in range(4)]
        assert codes[:2] == [200, 200] and codes[-1] == 429
        assert c.get("/healthz").status_code == 200        # health is never throttled
