-- Runs once, the first time the Postgres container starts with an empty volume.
-- To re-run: docker compose down -v && docker compose up -d

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS comments (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT        NOT NULL,          -- dataset name, e.g. 'yelp_review_full'
    source_id   TEXT        NOT NULL,          -- id within that dataset
    body        TEXT        NOT NULL,
    rating      SMALLINT,                      -- optional, 1-5
    created_at  TIMESTAMPTZ,                   -- optional, when the comment was written
    loaded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding   vector(384),                   -- all-MiniLM-L6-v2 output size; NULL until embedded
    UNIQUE (source, source_id)
);

-- Cosine-distance index for the similarity search (the <=> operator).
CREATE INDEX IF NOT EXISTS comments_embedding_idx
    ON comments USING hnsw (embedding vector_cosine_ops);
