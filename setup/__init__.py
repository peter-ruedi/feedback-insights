"""One-time project setup, run in this order from the repo root:

    python -m setup.fetch_dataset --n 3000          # download sample reviews -> data/yelp_sample.csv
    python -m setup.ingest data/yelp_sample.csv --source yelp_review_full
    python -m setup.embed                           # embed comments that have none yet

Not part of the installed package: the API and MCP server never import this.
"""
