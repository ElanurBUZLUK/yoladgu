import time
from collections import defaultdict, deque


class SlidingWindowRateLimiter:
    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls = max_calls
        self.window = window_seconds
        self.calls = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.time()
        dq = self.calls[key]
        while dq and now - dq[0] > self.window:
            dq.popleft()
        if len(dq) < self.max_calls:
            dq.append(now)
            return True
        return False


