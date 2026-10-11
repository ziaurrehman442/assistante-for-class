import unittest
from unittest.mock import patch

from app.services.rate_limiter import consume_request


class RateLimiterTests(unittest.TestCase):
    @patch("app.services.rate_limiter.monotonic", side_effect=[100.0, 101.0, 102.0])
    def test_limits_requests_within_window_and_returns_retry_time(self, _clock):
        key = "rate-limit-test-key"
        self.assertIsNone(consume_request(key, limit=2, window_seconds=60))
        self.assertIsNone(consume_request(key, limit=2, window_seconds=60))
        self.assertEqual(consume_request(key, limit=2, window_seconds=60), 58)

    @patch(
        "app.services.rate_limiter.monotonic",
        side_effect=[200.0, 261.0],
    )
    def test_allows_request_after_window_expires(self, _clock):
        key = "rate-limit-expiration-test-key"
        self.assertIsNone(consume_request(key, limit=1, window_seconds=60))
        self.assertIsNone(consume_request(key, limit=1, window_seconds=60))


if __name__ == "__main__":
    unittest.main()
