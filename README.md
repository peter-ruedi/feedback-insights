# Feedback Insights

Ask plain-English questions about thousands of open-ended comments ("What do people say about pricing?") and get themes backed by the actual comments they came from. Also exposed as an MCP server so Claude can query it.

> Work in progress: data loading and embeddings are done; the `/ask` endpoint, MCP server and UI come next.

## Run it locally

Needs Docker and Python 3.11+.

```bash
cp .env.example .env
docker compose up -d                      # Postgres 16 + pgvector, schema from db/init.sql

python -m venv .venv && source .venv/bin/activate
pip install -e ".[fetch,dev]"

python scripts/fetch_dataset.py --n 3000  # public Yelp reviews -> data/yelp_sample.csv
python -m feedback_insights.ingest data/yelp_sample.csv --source yelp_review_full
python -m feedback_insights.embed         # local all-MiniLM-L6-v2, no API key needed

pytest
```

Check it worked:

```bash
docker compose exec db psql -U feedback -c \
  "SELECT count(*) AS total, count(embedding) AS embedded FROM comments;"
```

## How it works

- **Storage:** one `comments` table in PostgreSQL with a `vector(384)` column and an HNSW index for cosine search.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, run locally and normalized to unit length.
- **Loading:** any CSV with a `text` column (plus optional `id`, `rating`, `created_at`). Reloads are idempotent.

## Built with Claude Code

Notes on how I used it go here.
