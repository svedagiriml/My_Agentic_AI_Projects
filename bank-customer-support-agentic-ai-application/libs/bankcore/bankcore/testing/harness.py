"""In-process wiring of the full system (supervisor + 8 ReAct agents + 8 MCP servers).

Used by integration/trajectory tests and offline evaluation. Production wiring differs only in
transport: HTTP between Cloud Run services instead of in-process calls.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from bankcore.agents.react import ReActAgent
from bankcore.agents.specs import SPECS
from bankcore.config import LLMProvider, Settings
from bankcore.core_banking.gateway import MockCoreBanking
from bankcore.llm.embeddings import HashEmbedder
from bankcore.llm.factory import ResilientLLM, set_fake_model_factory
from bankcore.mcp.client import MCPToolClient
from bankcore.schemas.common import AgentName
from bankcore.testing.offline_llm import offline_model_factory

MCP_MODULES = {
    AgentName.ACCOUNT: "mcp_servers.accounts.server", AgentName.PAYMENTS: "mcp_servers.payments.server",
    AgentName.CARDS: "mcp_servers.cards.server", AgentName.FRAUD: "mcp_servers.fraud.server",
    AgentName.LOANS: "mcp_servers.loans.server", AgentName.INVESTMENTS: "mcp_servers.investments.server",
    AgentName.KNOWLEDGE: "mcp_servers.knowledge.server", AgentName.COMPLIANCE: "mcp_servers.compliance.server",
}


@dataclass
class Harness:
    settings: Settings
    gateway: MockCoreBanking
    mcp_servers: dict[AgentName, Any] = field(default_factory=dict)
    agents: dict[AgentName, ReActAgent] = field(default_factory=dict)
    container: Any = None
    app: Any = None


async def build_harness(settings: Settings, *, llm_overrides: dict[str, Any] | None = None,
                        offline: bool = True) -> Harness:
    """offline=True -> deterministic rule-based models; offline=False -> real (free) LLMs from settings."""
    from supervisor.app.agent_client import InProcessSubAgentClient
    from supervisor.app.container import build_container
    from supervisor.app.main import create_app

    gw = MockCoreBanking()
    h = Harness(settings=settings, gateway=gw)
    for name, mod in MCP_MODULES.items():
        module = importlib.import_module(mod)
        if name == AgentName.KNOWLEDGE:
            from mcp_servers.knowledge.kb_index import InMemoryKnowledgeIndex
            from mcp_servers.knowledge.server import KB_DIR
            idx = InMemoryKnowledgeIndex(HashEmbedder(settings.embedding_dim))
            await idx.ingest_dir(KB_DIR)
            h.mcp_servers[name] = module.create(index=idx, settings=settings)
        elif name == AgentName.COMPLIANCE:
            h.mcp_servers[name] = module.create(settings=settings)
        else:
            h.mcp_servers[name] = module.create(gateway=gw, settings=settings)

    overrides = llm_overrides or {}
    for name, spec in SPECS.items():
        if offline:
            set_fake_model_factory(lambda role, n=name.value: overrides.get(n) or offline_model_factory(role, n))
            llm = ResilientLLM("subagent", settings, providers=[LLMProvider.FAKE])
        else:
            llm = ResilientLLM("subagent", settings)
        mcp = MCPToolClient(agent=name.value, domain=spec.mcp_domain, target=h.mcp_servers[name].server,
                            allowed_tools=set(spec.tools), settings=settings)
        h.agents[name] = ReActAgent(spec, llm, mcp, settings=settings)
    if offline:
        set_fake_model_factory(lambda role: overrides.get("supervisor") or offline_model_factory(role))
        sup_llm = ResilientLLM("supervisor", settings, providers=[LLMProvider.FAKE])
    else:
        sup_llm = ResilientLLM("supervisor", settings)
    h.container = await build_container(settings, agents=InProcessSubAgentClient(h.agents, settings), llm=sup_llm,
                                        use_external=False)
    h.app = create_app(container=h.container, settings=settings)
    return h
