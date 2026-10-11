from collections import defaultdict, deque
from math import ceil
from threading import Lock
from time import monotonic


_requests_by_key = defaultdict(deque)
_lock = Lock()


def consume_request(key: str, limit: int, window_seconds: int):
    now = monotonic()
    with _lock:
        timestamps = _requests_by_key[key]
        while timestamps and now - timestamps[0] >= window_seconds:
            timestamps.popleft()

        if len(timestamps) >= limit:
            return max(1, ceil(window_seconds - (now - timestamps[0])))

        timestamps.append(now)
        return None
