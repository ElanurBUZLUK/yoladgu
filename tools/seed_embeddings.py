import argparse, json, numpy as np, redis, hashlib, csv
from app.core.config import settings
from app.services.embedding_service import EmbeddingService

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()
    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    emb = EmbeddingService(settings.EMBED_DIM)
    # Prefer examples/questions_sample.csv if exists
    try:
        texts, ids = [], []
        with open("examples/questions_sample.csv", newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = int(row["id"]) if "id" in row else int(row.get("question_id") or 0)
                txt = row.get("text") or ""
                if qid:
                    ids.append(qid); texts.append(txt)
        if texts:
            X = emb.embed_sync(texts)
            for qid, vec in zip(ids, X):
                r.set(f"embedding:question:{qid}", json.dumps(list(map(float, vec.tolist()))))
            print(f"Seeded {len(ids)} embeddings from examples/questions_sample.csv")
            return
    except Exception:
        pass
    for qid in range(1, args.n+1):
        h = hashlib.sha256(str(qid).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        vec = rng.normal(0,1,size=settings.EMBED_DIM).astype(float).tolist()
        r.set(f"embedding:question:{qid}", json.dumps(vec))
    print(f"Seeded {args.n} embeddings (fallback)")

if __name__ == "__main__":
    main()
