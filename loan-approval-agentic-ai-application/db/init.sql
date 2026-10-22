-- Long-term memory store (episodic / semantic / procedural) on Postgres + PGVector.
-- LangGraph checkpoint tables are created automatically by AsyncPostgresSaver.setup().
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS long_term_memory (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL CHECK (kind IN ('episodic', 'semantic', 'procedural')),
    namespace   TEXT NOT NULL DEFAULT 'global',          -- 'global' | 'applicant:A1001' | 'agent:assess_risk'
    content     TEXT NOT NULL,                           -- PII-redacted, gate-approved
    source      TEXT NOT NULL,                           -- provenance (rule_engine, human_review, policy_admin, ...)
    confidence  REAL NOT NULL DEFAULT 0.9,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding   vector(384) NOT NULL
);
CREATE INDEX IF NOT EXISTS ltm_embedding_hnsw ON long_term_memory USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS ltm_kind_ns ON long_term_memory (kind, namespace);

-- Memory-poisoning quarantine: rejected writes wait here for human review.
CREATE TABLE IF NOT EXISTS memory_quarantine (
    id          TEXT PRIMARY KEY,
    record      JSONB NOT NULL,
    reasons     TEXT[] NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_by TEXT,
    released    BOOLEAN NOT NULL DEFAULT FALSE
);

-- Append-only decision audit log (hash-chained). Revoke UPDATE/DELETE for the app role.
CREATE TABLE IF NOT EXISTS decision_audit (
    audit_id    TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    decision    JSONB NOT NULL,
    prev_hash   TEXT NOT NULL,
    chain_hash  TEXT NOT NULL,
    stored_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- REVOKE UPDATE, DELETE ON decision_audit FROM loanflow_app;
