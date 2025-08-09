import csv, json, argparse, redis
from app.core.config import settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)

    count = 0
    with open(args.csv, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["id"])
            data = {
                "id": qid,
                "topic_id": int(row.get("topic_id") or 0),
                "text": row["text"],
                "options": [row["opt_a"], row["opt_b"], row["opt_c"], row["opt_d"]],
                "correct_index": int(row["correct_index"]),
                "difficulty_level": int(row.get("difficulty_level") or 1),
            }
            r.set(f"question:{qid}", json.dumps(data))
            count += 1
    print(f"Imported {count} questions.")


if __name__ == "__main__":
    main()


