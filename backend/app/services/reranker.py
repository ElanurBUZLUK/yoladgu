from typing import List, Dict

class LightweightReranker:
    def __init__(self, weights: dict = None):
        self.weights = weights or {"dense": 0.6, "sparse": 0.4, "graph": 0.2}

    def rerank(self, items: List[Dict]) -> List[Dict]:
        for item in items:
            src = item.get("src", "dense")
            weight = self.weights.get(src, 0.5)
            # Adjust score based on source weight (lower dist is better)
            item["rerank_score"] = item["dist"] * (1.0 / weight)
        return sorted(items, key=lambda x: x["rerank_score"])