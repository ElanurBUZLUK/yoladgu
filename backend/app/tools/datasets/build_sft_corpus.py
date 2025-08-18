import re, json
from pathlib import Path
from sqlalchemy import text
from app.core.database import get_db

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")

def scrub(x: str) -> str:
    return PHONE.sub("[PHONE]", EMAIL.sub("[EMAIL]", x or ""))

def build(limit=5000, out_path="/tmp/sft_corpus.jsonl"):
    db = next(get_db())
    rows = db.execute(text(
        """
        SELECT a.user_id, a.answer, g.stem, g.options, g.answer AS gold
        FROM attempts a
        JOIN generated_questions g ON g.id::text = a.question_id
        WHERE a.domain = 'english' AND a.answer IS NOT NULL
        ORDER BY a.created_at DESC
        LIMIT :l
        """
    ), {"l": limit}).mappings().all()

    with open(out_path, 'w', encoding='utf-8') as f:
        for r in rows:
            prompt = f"Question: {scrub(r['stem'])}\nOptions: {scrub(json.dumps(r['options']))}"
            out = scrub(json.dumps(r['answer']))
            item = {"input": prompt, "output": out, "meta": {"source":"prod", "task":"mcq"}}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} lines -> {out_path}")

if __name__ == "__main__":
    build()
```

