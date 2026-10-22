"""bankcore — shared building blocks for the agentic bank customer-support platform.

Every microservice (supervisor, sub-agents, MCP servers, memory worker) depends on
this package so that schemas, security, guardrails and resilience primitives are
identical everywhere. Changing a handoff schema in one place changes it for all.
"""

__version__ = "1.0.0"
