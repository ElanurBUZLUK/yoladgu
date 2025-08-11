import os, json, argparse, random
from app.services.embedding_api import get_embedding_provider


DEMO = [
    "Trigonometrik oranlar",
    "Kuvvet ve hareket",
    "Periyodik tablo",
    "Fotosentez",
    "Limit ve süreklilik",
    "Sanayi devrimi",
    "Muson iklimi",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--out", type=str, default="data/embeddings_api.jsonl")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    prov = get_embedding_provider()

    corpus = [random.choice(DEMO) for _ in range(args.n)]
    vecs = prov.embed_batch(corpus)

    with open(args.out, "w", encoding="utf-8") as f:
        for t, v in zip(corpus, vecs):
            f.write(json.dumps({"text": t, "embedding": v}, ensure_ascii=False) + "\n")

    print("OK:", args.out)


if __name__ == "__main__":
    main()


