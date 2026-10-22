-- Horizon Trust agentic support — Postgres 16 + pgvector schema
-- Long-term memory (episodic / semantic / procedural), RAG knowledge base, audit log.
-- LangGraph checkpoint tables are created automatically by AsyncPostgresSaver.setup().

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------- memories --
-- Common shape for all three memory kinds. Tenant isolation: user_id is mandatory for
-- episodic/semantic (enforced by CHECK) and NULL for global procedural memory.
CREATE TABLE IF NOT EXISTS episodic_memory (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    content         TEXT NOT NULL CHECK (char_length(content) <= 2000),
    metadata        JSONB NOT NULL DEFAULT '{}',
    source          TEXT NOT NULL,
    trust_score     REAL NOT NULL CHECK (trust_score BETWEEN 0 AND 1),
    trace_id        TEXT,
    content_sha256  TEXT NOT NULL UNIQUE,
    embedding       vector(384) NOT NULL,
    hit_count       INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS semantic_memory (LIKE episodic_memory INCLUDING ALL);
CREATE TABLE IF NOT EXISTS procedural_memory (LIKE episodic_memory INCLUDING ALL);
ALTER TABLE procedural_memory ALTER COLUMN user_id DROP NOT NULL;
ALTER TABLE procedural_memory DROP CONSTRAINT IF EXISTS procedural_global;
ALTER TABLE procedural_memory ADD CONSTRAINT procedural_global CHECK (user_id IS NULL);

-- HNSW ANN indexes (cosine) + tenant filter indexes
CREATE INDEX IF NOT EXISTS episodic_embedding_hnsw ON episodic_memory USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS semantic_embedding_hnsw ON semantic_memory USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS procedural_embedding_hnsw ON procedural_memory USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS episodic_user_idx ON episodic_memory (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS semantic_user_idx ON semantic_memory (user_id);

-- Rejected writes (memory-poisoning attempts) are kept for security review, never served.
CREATE TABLE IF NOT EXISTS memory_quarantine (
    id          TEXT PRIMARY KEY,
    user_id     TEXT,
    kind        TEXT NOT NULL,
    content     TEXT NOT NULL,
    source      TEXT NOT NULL,
    reason      TEXT NOT NULL,
    trace_id    TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -------------------------------------------------------------- knowledge --
CREATE TABLE IF NOT EXISTS kb_chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    title       TEXT NOT NULL,
    content     TEXT NOT NULL,
    source      TEXT NOT NULL,
    version     TEXT NOT NULL DEFAULT 'v1',
    embedding   vector(384) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || content)) STORED,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS kb_embedding_hnsw ON kb_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS kb_tsv_gin ON kb_chunks USING gin (tsv);

-- ------------------------------------------------------------------ audit --
-- Append-only: application role has INSERT only (see grants below).
CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL PRIMARY KEY,
    event_type  TEXT NOT NULL,
    user_id     TEXT,
    trace_id    TEXT,
    payload     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS audit_user_idx ON audit_log (user_id, created_at DESC);

-- ------------------------------------------------------------ evaluation --
CREATE TABLE IF NOT EXISTS hitl_reviews (
    id          BIGSERIAL PRIMARY KEY,
    case_id     TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    correct     BOOLEAN NOT NULL,
    helpful     SMALLINT CHECK (helpful BETWEEN 1 AND 5),
    safe        BOOLEAN NOT NULL,
    notes       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Least privilege (role created by Terraform / Cloud SQL IAM in prod)
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'bank_app') THEN
    REVOKE UPDATE, DELETE ON audit_log FROM bank_app;
    GRANT INSERT, SELECT ON audit_log TO bank_app;
  END IF;
END $$;
