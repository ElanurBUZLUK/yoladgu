import json, argparse, numpy as np, redis, os
from app.core.config import settings
from app.services.vector_index_manager import VectorIndexManager

CACHE_KEY = "embedding:question:{}"

def all_ids(r: redis.Redis) -> list[int]:
    keys = r.keys(CACHE_KEY.format("*"))
    out = []
    for k in keys:
        try: out.append(int(k.split(":")[-1]))
        except: pass
    return sorted(out)

def load_from_redis(r: redis.Redis, ids: list[int]):
    pipe = r.pipeline()
    for qid in ids:
        pipe.get(CACHE_KEY.format(qid))
    raws = pipe.execute()
    vecs, kept = [], []
    for qid, raw in zip(ids, raws):
        if not raw: continue
        try:
            v = json.loads(raw)
            if len(v) == settings.EMBED_DIM:
                vecs.append(v); kept.append(qid)
        except: pass
    if not vecs:
        return np.empty((0, settings.EMBED_DIM), np.float32), np.empty((0,), np.int64)
    return np.array(vecs, np.float32), np.array(kept, np.int64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    ids = all_ids(r)
    if args.limit and args.limit>0: ids = ids[:args.limit]
    X, I = load_from_redis(r, ids)
    if X.shape[0] == 0:
        print("No embeddings found."); return
    mgr = VectorIndexManager(settings.REDIS_URL)
    info = mgr.build_on_inactive(X, I)
    print("Built:", info)

if __name__ == "__main__":
    main()
