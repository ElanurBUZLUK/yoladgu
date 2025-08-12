from __future__ import annotations

from prometheus_client import Counter, Histogram

rec_latency = Histogram(
    "recommendation_latency_seconds",
    "Latency for recommendation ensemble endpoint",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

rec_scores = Histogram(
    "model_score",
    "Model scores by channel",
    labelnames=("channel",),
)

events_total = Counter(
    "events_total",
    "Event count by type",
    labelnames=("type",),
)


