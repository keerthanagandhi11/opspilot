from types import SimpleNamespace

import pytest

from app.rate_limit import RateLimited, RateLimiter


def req(ip="10.0.0.9", xff=None):
    return SimpleNamespace(client=SimpleNamespace(host=ip), headers={"x-forwarded-for": xff} if xff else {})


def test_blocks_after_limit():
    rl = RateLimiter(2)
    rl.check(req())
    rl.check(req())
    with pytest.raises(RateLimited):
        rl.check(req())


def test_spoofed_header_ignored_without_trusted_proxy():
    rl = RateLimiter(1)
    rl.check(req(xff="1.1.1.1"))
    with pytest.raises(RateLimited):
        rl.check(req(xff="2.2.2.2"))  # same real client, different fake header


def test_behind_proxy_uses_entry_added_by_proxy():
    rl = RateLimiter(1, trust_proxy=True)
    rl.check(req(ip="proxy", xff="fake-1, 203.0.113.7"))
    with pytest.raises(RateLimited):
        rl.check(req(ip="proxy", xff="fake-2, 203.0.113.7"))  # visitor changes the fake part only
    rl.check(req(ip="proxy", xff="198.51.100.4"))  # a different real visitor is allowed
