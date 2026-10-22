"""Offline RAG evaluation with RAGAS.

For every golden question we run the *same* production path as the gateway
(rewrite -> hybrid retrieve -> generate), but bypass the semantic cache so each
sample is a real retrieval + generation, then score with RAGAS:

| Metric                                   | Measures                                                   |
|------------------------------------------|------------------------------------------------------------|
| faithfulness                             | are the answer's claims supported by the retrieved context |
| answer_relevancy (ResponseRelevancy)     | does the answer address the question                       |
| context_precision (LLM, with reference)  | are relevant chunks ranked above irrelevant ones           |
| context_recall (LLM)                     | does the context contain everything the reference needs    |

The judge LLM is free (Groq llama-3.3-70b by default) and the embeddings are the
same local BGE model, so evaluation costs nothing. Results are written to
evaluation/results/ and, if LangFuse is configured, pushed as scores on a
dedicated trace per sample. `--min-*` thresholds make it a CI quality gate.

Usage:
    python -m evaluation.run_ragas --limit 14 --min-faithfulness 0.7
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
from datetime import UTC, datetime

from common.config import get_settings
from common.http_client import ServiceClient
from common.observability import new_trace_id
from common.schemas import (
    GenerateRequest,
    GenerateResponse,
    LoanProduct,
    RetrieveRequest,
    RetrieveResponse,
    RewriteRequest,
    RewrittenQuery,
)

ROOT = pathlib.Path(__file__).parent
METRIC_COLS = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]


async def collect(limit: int) -> list[dict]:
    s = get_settings()
    kw = dict(use_id_token=s.use_gcp_id_token, timeout_s=120)
    gen = ServiceClient("generator", str(s.generator_url), **kw)
    ret = ServiceClient("retriever", str(s.retriever_url), **kw)
    rows = [json.loads(line) for line in (ROOT / "golden_dataset.jsonl").read_text().splitlines() if line.strip()]
    out = []
    for row in rows[:limit]:
        tid = new_trace_id()
        hdr = {"X-Trace-Id": tid}
        rw = await gen.post_model("/rewrite", RewriteRequest(query=row["question"], trace_id=tid), RewrittenQuery, hdr)
        prod = None if rw.product is LoanProduct.general else rw.product
        rr = await ret.post_model("/retrieve", RetrieveRequest(query=rw.rewritten_query, product_filter=prod,
                                                               top_k=s.final_top_k, trace_id=tid), RetrieveResponse, hdr)
        g = await gen.post_model("/generate", GenerateRequest(query=row["question"], rewritten_query=rw.rewritten_query,
                                                             chunks=rr.chunks, trace_id=tid), GenerateResponse, hdr)
        out.append({"user_input": row["question"], "retrieved_contexts": [c.display_text for c in rr.chunks],
                    "response": g.answer, "reference": row["reference"], "trace_id": tid,
                    "retrieved_ids": [c.chunk_id for c in rr.chunks], "latency_ms": g.latency_ms,
                    "tokens": g.usage.total})
        print(f"[collected] {row['question'][:70]}")
    await gen.aclose()
    await ret.aclose()
    return out


def score(samples: list[dict]):  # type: ignore[no-untyped-def]
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    s = get_settings()
    judge = LangchainLLMWrapper(ChatGroq(model=s.groq_model, api_key=s.groq_api_key.get_secret_value(),  # type: ignore[union-attr]
                                         temperature=0))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=s.embedding_model,
                                                           encode_kwargs={"normalize_embeddings": True}))
    ds = EvaluationDataset.from_list([{k: x[k] for k in ("user_input", "retrieved_contexts", "response", "reference")}
                                      for x in samples])
    # Free-tier friendly: few workers, patient retries with backoff.
    rc = RunConfig(max_workers=2, max_retries=8, max_wait=60, timeout=240)
    result = evaluate(ds, metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(),
                                   LLMContextRecall()], llm=judge, embeddings=emb, run_config=rc)
    return result.to_pandas()


def push_to_langfuse(df, samples: list[dict]) -> None:  # type: ignore[no-untyped-def]
    s = get_settings()
    if not s.langfuse_enabled:
        return
    from langfuse import Langfuse

    lf = Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key.get_secret_value(),  # type: ignore[union-attr]
                  host=s.langfuse_host)
    for (_, row), sample in zip(df.iterrows(), samples):
        for m in METRIC_COLS:
            if m in row and row[m] == row[m]:  # skip NaN
                lf.create_score(trace_id=sample["trace_id"], name=f"ragas_{m}", value=float(row[m]))
    lf.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        ap.add_argument(f"--min-{m.replace('_', '-')}", type=float, default=0.0)
    args = ap.parse_args()

    samples = asyncio.run(collect(args.limit))
    df = score(samples)
    df["trace_id"] = [x["trace_id"] for x in samples]
    df["latency_ms"] = [x["latency_ms"] for x in samples]
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    df.to_csv(out_dir / f"ragas_{stamp}.csv", index=False)
    summary = {c: round(float(df[c].mean()), 4) for c in METRIC_COLS if c in df}
    summary["n"] = len(df)
    (out_dir / f"ragas_{stamp}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    push_to_langfuse(df, samples)

    gates = {"faithfulness": args.min_faithfulness, "answer_relevancy": args.min_answer_relevancy,
             "llm_context_precision_with_reference": args.min_context_precision,
             "context_recall": args.min_context_recall}
    failed = [f"{k}={summary.get(k)} < {v}" for k, v in gates.items() if v and summary.get(k, 0) < v]
    if failed:
        print("QUALITY GATE FAILED:", "; ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
