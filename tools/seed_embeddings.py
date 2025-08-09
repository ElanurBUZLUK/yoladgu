import argparse, json, numpy as np, redis, hashlib
from app.core.config import settings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()
    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    for qid in range(1, args.n+1):
        # deterministic random
        h = hashlib.sha256(str(qid).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        vec = rng.normal(0,1,size=settings.EMBED_DIM).astype(float).tolist()
        r.set(f"embedding:question:{qid}", json.dumps(vec))
    print(f"Seeded {args.n} embeddings")

if __name__ == "__main__":
    main()
