"""Shared HTTP / caching primitives for ingesters.

All scrapers and dataset downloaders go through ``http_get`` so we get:
- a single User-Agent string,
- automatic retry with exponential backoff,
- on-disk response caching keyed by URL (so re-runs are cheap and offline-friendly).
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..paths import DATA_CACHE

log = logging.getLogger(__name__)

USER_AGENT = (
    "world-cup-predictions/0.1 (research; +https://github.com/SuperXingKong)"
)
DEFAULT_TIMEOUT = 30


@dataclass(frozen=True)
class CacheEntry:
    path: Path
    fresh: bool


def _cache_key(url: str) -> Path:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    return DATA_CACHE / f"{h}.bin"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20))
def _do_get(url: str, *, headers: dict | None = None, timeout: int = DEFAULT_TIMEOUT) -> bytes:
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    log.info("GET %s", url)
    r = requests.get(url, headers=h, timeout=timeout)
    r.raise_for_status()
    return r.content


def http_get(
    url: str,
    *,
    headers: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    use_cache: bool = True,
    sleep_before: float = 0.0,
) -> bytes:
    """Fetch ``url``, returning bytes. Caches on-disk by URL hash by default."""
    cache_path = _cache_key(url)
    if use_cache and cache_path.exists():
        return cache_path.read_bytes()
    if sleep_before > 0:
        time.sleep(sleep_before)
    body = _do_get(url, headers=headers, timeout=timeout)
    if use_cache:
        cache_path.write_bytes(body)
    return body


def http_get_text(url: str, **kw) -> str:
    return http_get(url, **kw).decode("utf-8", errors="replace")
