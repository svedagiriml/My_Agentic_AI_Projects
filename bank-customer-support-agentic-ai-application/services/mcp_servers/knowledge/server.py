"""MCP server: Core Banking FAQ & General Knowledge (RAG). No customer data is touched."""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

from bankcore.errors import NotFound
from bankcore.llm.embeddings import get_embedder
from bankcore.mcp.server import BankMCPServer, ToolContext
from bankcore.schemas import tools as T

from mcp_servers.knowledge.kb_index import InMemoryKnowledgeIndex

KB_DIR = Path(os.getenv("KB_DIR", Path(__file__).resolve().parents[3] / "data" / "kb"))

BRANCHES = [
    {"branch_id": "BR-001", "name": "Horizon Trust — Springfield Downtown", "kind": "branch",
     "address": "100 Main St, Springfield, IL 62701", "lat": 39.7990, "lon": -89.6440,
     "hours": "Mon-Fri 9:00-17:00, Sat 9:00-13:00"},
    {"branch_id": "ATM-014", "name": "Horizon ATM — Capitol Ave", "kind": "atm",
     "address": "455 Capitol Ave, Springfield, IL 62701", "lat": 39.7980, "lon": -89.6550, "hours": "24/7"},
    {"branch_id": "BR-002", "name": "Horizon Trust — Austin Congress Ave", "kind": "branch",
     "address": "700 Congress Ave, Austin, TX 73301", "lat": 30.2690, "lon": -97.7420,
     "hours": "Mon-Fri 9:00-18:00, Sat 10:00-14:00"},
    {"branch_id": "ATM-031", "name": "Horizon ATM — South Lamar", "kind": "atm",
     "address": "1500 S Lamar Blvd, Austin, TX 78704", "lat": 30.2500, "lon": -97.7660, "hours": "24/7"},
    {"branch_id": "BR-003", "name": "Horizon Trust — Chicago Loop", "kind": "branch",
     "address": "20 W Madison St, Chicago, IL 60602", "lat": 41.8820, "lon": -87.6290,
     "hours": "Mon-Fri 8:30-17:30"},
]
ZIP_COORDS = {"62701": (39.8017, -89.6436), "73301": (30.2672, -97.7431), "78704": (30.2430, -97.7650),
              "60602": (41.8830, -87.6290), "60601": (41.8860, -87.6230)}


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    r = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def parse_origin(s: str) -> tuple[float, float]:
    m = re.match(r"^\s*(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)\s*$", s)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon
    z = re.search(r"\b(\d{5})\b", s)
    if z and z.group(1) in ZIP_COORDS:
        return ZIP_COORDS[z.group(1)]
    raise NotFound("unknown location", user_message="I couldn't find that ZIP code; try 'lat,lon' coordinates.")


def create(index: InMemoryKnowledgeIndex | None = None, **kwargs) -> BankMCPServer:
    srv = BankMCPServer("knowledge", "Bank policies, fees, hours and locations. Answers are general, not "
                                     "account-specific.", **kwargs)
    state = {"index": index}

    async def get_index() -> InMemoryKnowledgeIndex:
        if state["index"] is None:
            idx = InMemoryKnowledgeIndex(get_embedder())
            await idx.ingest_dir(KB_DIR)
            state["index"] = idx
        return state["index"]

    @srv.tool(name="query_knowledge_base", input_model=T.KBQueryInput, output_model=T.KBQueryOutput,
              description="Hybrid semantic + keyword search over official bank policy documents (fees, overdraft, "
                          "routing/wires, disputes, cards, savings/CDs, mortgages, privacy, digital banking).")
    async def query_knowledge_base(ctx: ToolContext, semantic_search_string: str, top_k: int = 4) -> dict:
        idx = await get_index()
        hits = await idx.search(semantic_search_string, top_k=top_k)
        return {"query": semantic_search_string, "chunks": [
            {"doc_id": c.doc_id, "title": c.title, "content": c.content, "score": s, "source": c.source}
            for c, s in hits]}

    @srv.tool(name="locate_nearest_atm_or_branch", input_model=T.LocateInput, output_model=T.LocateOutput,
              description="Find the nearest Horizon Trust branches/ATMs to a US ZIP code or 'lat,lon' coordinates.")
    async def locate_nearest_atm_or_branch(ctx: ToolContext, zip_code_or_coords: str) -> dict:
        origin = parse_origin(zip_code_or_coords)
        ranked = sorted(BRANCHES, key=lambda b: haversine_km(origin, (b["lat"], b["lon"])))[:3]
        return {"origin": zip_code_or_coords, "results": [
            {"branch_id": b["branch_id"], "name": b["name"], "kind": b["kind"], "address": b["address"],
             "distance_km": round(haversine_km(origin, (b["lat"], b["lon"])), 2), "hours": b["hours"]}
            for b in ranked]}

    return srv
