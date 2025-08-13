from __future__ import annotations

from typing import Dict, List, Tuple


def choose_target(cands: Dict[str, List[Tuple[str, float]]], mode: str) -> Dict[str, str] | None:
    pool: List[Tuple[str, str, float]] = []
    if mode in ("vocab", "mixed"):
        pool += [("vocab", k, s) for k, s in cands.get("vocab", [])]
    if mode in ("grammar", "mixed"):
        pool += [("grammar", k, s) for k, s in cands.get("grammar", [])]
    if not pool:
        return None
    pool.sort(key=lambda x: x[2], reverse=True)
    typ, key, _ = pool[0]
    return {"type": typ, "key": key}


