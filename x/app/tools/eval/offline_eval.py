import json, argparse
from typing import List
from pathlib import Path
from app.services.llm_router import LLMRouter
from .schemas import EvalItem, EvalResult

def judge(item: EvalItem, output: str) -> bool | None:
    if item.expected is None:
        return None
    gt, out = item.expected.strip().lower(), output.strip().lower()
    return (gt == out) or (gt in out)

def run_eval(input_path: str, out_path: str):
    data = [EvalItem(**json.loads(l)) for l in Path(input_path).read_text().splitlines() if l.strip()]
    router = LLMRouter()
    results: List[EvalResult] = []
    acc, n = 0, 0
    total_cost = 0.0

    for it in data:
        res = router.run(it.prompt)
        ok = judge(it, res['text'])
        if ok is not None:
            n += 1
            acc += 1 if ok else 0
        total_cost += float(res['usage'].get('cost_usd', 0.0))
        results.append(EvalResult(id=it.id, text=res['text'], correct=ok, tokens={
            'prompt': res['usage'].get('prompt_tokens',0),
            'completion': res['usage'].get('completion_tokens',0)
        }).model_json())

    Path(out_path).write_text("\n".join(results))
    summary = {
        'accept_at_1': (acc / n) if n else None,
        'items': len(data),
        'evaluated': n,
        'total_cost_usd': round(total_cost, 6)
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()
    run_eval(args.inp, args.out)