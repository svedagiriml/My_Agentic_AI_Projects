from __future__ import annotations

import os

# Deterministic, offline configuration for the whole test session (set before bankcore imports).
os.environ.update({
    "ENVIRONMENT": "test",
    "LLM_PROVIDER": "fake",
    "LLM_FALLBACK_PROVIDERS": "[]",
    "EMBEDDING_BACKEND": "hash",
    "LANGFUSE_ENABLED": "false",
    "SEMANTIC_CACHE_ENABLED": "true",
    "LOG_LEVEL": "WARNING",
})

import pytest  # noqa: E402

from bankcore.config import get_settings  # noqa: E402
from bankcore.resilience.circuit_breaker import reset_breakers  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_state():
    reset_breakers()
    yield
    reset_breakers()


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
async def harness(settings):
    from bankcore.testing.harness import build_harness
    return await build_harness(settings)


@pytest.fixture
async def client(harness):
    import httpx
    transport = httpx.ASGITransport(app=harness.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
