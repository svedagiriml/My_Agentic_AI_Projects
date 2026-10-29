# Evaluation report

Cases: 33  |  Gates passed: **True**

| Metric | Value |
|---|---|
| routing_accuracy | 1.000 |
| tool_precision | 1.000 |
| tool_recall | 1.000 |
| tool_f1 | 1.000 |
| trajectory_exact | 1.000 |
| trajectory_in_order | 1.000 |
| fact_recall | 1.000 |
| forbidden_content_rate | 0.000 |
| hallucination_rate | 0.000 |
| status_accuracy | 1.000 |
| guardrail_block_recall | 1.000 |
| guardrail_false_positive_rate | 0.000 |
| hitl_trigger_accuracy | 1.000 |
| latency_p50_s | 0.059 |
| latency_p95_s | 0.082 |
| retrieval_recall@4 | 1.000 |
| retrieval_precision@4 | 0.667 |
| retrieval_mrr | 1.000 |
| retrieval_ndcg@4 | 1.000 |

## Quality gates

| Gate | Pass |
|---|---|
| routing_accuracy | ✅ |
| tool_f1 | ✅ |
| trajectory_in_order | ✅ |
| fact_recall | ✅ |
| hallucination_rate | ✅ |
| guardrail_block_recall | ✅ |
| guardrail_false_positive | ✅ |
| hitl_trigger_accuracy | ✅ |
| retrieval_recall@4 | ✅ |

## Per-category score (fact_recall × tool_f1)

- account: 1.0
- cards: 1.0
- cards_hitl: 1.0
- context_switching: 1.0
- fraud: 1.0
- fraud_hitl: 1.0
- investments: 1.0
- investments_hitl: 1.0
- knowledge: 1.0
- loans: 1.0
- payments: 1.0
- payments_hitl: 1.0
- security: 1.0

## Failing cases

