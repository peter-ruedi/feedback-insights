"""Download a random sample of public reviews from Hugging Face as a CSV.

    pip install -e ".[fetch]"
    python -m setup.fetch_dataset --n 3000 --out data/yelp_sample.csv

Default dataset: yelp_review_full (public Yelp reviews, labels 0-4 = 1-5 stars).
Swap in any dataset with a text column; adjust the column mapping below.
"""

import argparse
import csv
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Yelp/yelp_review_full")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/yelp_sample.csv"))
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    sample = ds.shuffle(seed=args.seed).select(range(args.n))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "rating"])
        writer.writeheader()
        for i, row in enumerate(sample):
            writer.writerow({"id": f"{args.split}-{i}", "text": row["text"], "rating": row["label"] + 1})

    print(f"Wrote {args.n} rows to {args.out}")


if __name__ == "__main__":
    main()
