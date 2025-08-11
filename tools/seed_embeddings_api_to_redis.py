import argparse
import csv
import json
import os
from typing import List, Tuple

import redis

from app.core.config import settings
from app.services.embedding_api import get_embedding_provider


def read_questions(csv_path: str) -> Tuple[List[int], List[str]]:
    ids: List[int] = []
    texts: List[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row.get("id") or row.get("question_id") or 0)
            txt = (row.get("text") or "").strip()
            if qid and txt:
                ids.append(qid)
                texts.append(txt)
    return ids, texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="examples/questions_sample.csv")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    prov = get_embedding_provider()
    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

    ids, texts = read_questions(args.csv)
    if not ids:
        print("No questions found in CSV; aborting.")
        return

    total = len(ids)
    for start in range(0, total, args.batch):
        end = min(start + args.batch, total)
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        vecs = prov.embed_batch(batch_texts)
        pipe = r.pipeline()
        for qid, vec in zip(batch_ids, vecs):
            pipe.set(f"embedding:question:{qid}", json.dumps([float(x) for x in vec]))
        pipe.execute()
        print(f"Seeded {end} / {total}")

    print("Done. Embeddings written to Redis.")


if __name__ == "__main__":
    main()


