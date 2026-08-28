"""Tests for the SEC client's politeness guard.

Offline: only `_RateLimiter` is exercised, which is pure timing logic with no
network. The fetch paths are deliberately not simulated -- a fake that returns
whatever the test wants would prove nothing about the real endpoint, and the real
endpoint has no place in a suite that must run without a network.

This guard was untested. Its failure mode is not a wrong number in a table; it is
being blocked by the SEC for exceeding their published request ceiling, which would
make the corpus unrebuildable for everyone reading this repository.
"""

from __future__ import annotations

import time

import pytest

from retrieval_ablation.config import get_settings
from retrieval_ablation.corpus.edgar import _RateLimiter


class TestRateLimiter:
    def test_requests_are_spaced_by_the_configured_interval(self):
        """Five acquisitions at 20/s must take at least the four gaps between them."""
        limiter = _RateLimiter(20.0)

        started = time.monotonic()
        for _ in range(5):
            limiter.acquire()
        elapsed = time.monotonic() - started

        # Four intervals of 50 ms. A small tolerance because the clock is real.
        assert elapsed >= 4 * 0.05 - 0.005

    def test_the_first_acquisition_does_not_wait(self):
        """A limiter nobody has used should not delay the first request."""
        limiter = _RateLimiter(1.0)

        started = time.monotonic()
        limiter.acquire()

        assert time.monotonic() - started < 0.05

    def test_a_non_positive_rate_is_refused(self):
        """Zero would divide by zero; a negative rate would mean waiting backwards."""
        for bad in (0, -1.0):
            with pytest.raises(ValueError, match="positive"):
                _RateLimiter(bad)

    def test_a_faster_rate_takes_less_time(self):
        """Otherwise the rate parameter would not be doing anything."""

        def elapsed_for(rate: float) -> float:
            limiter = _RateLimiter(rate)
            started = time.monotonic()
            for _ in range(3):
                limiter.acquire()
            return time.monotonic() - started

        assert elapsed_for(50.0) < elapsed_for(10.0)


class TestConfiguredRate:
    def test_the_default_stays_under_the_published_ceiling(self):
        """SEC publishes 10 requests per second. This project asks for less.

        Asserted rather than trusted to a comment, because the consequence of
        drifting above it is an IP block that no test failure would explain.
        """
        rate = get_settings().edgar_requests_per_second

        assert 0 < rate <= 10.0
