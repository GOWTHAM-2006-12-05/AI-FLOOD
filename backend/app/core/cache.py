"""
Redis cache layer — async Redis client with typed helpers.

Provides:
    • Async connection pool
    • JSON serialisation cache helpers
    • TTL-aware get/set with namespace prefixes
    • Cache decorator for ML predictions
    • Cache invalidation

Usage:
    from backend.app.core.cache import cache_get, cache_set, cache_prediction

    # Direct usage
    await cache_set("weather:13.08:80.27", data, ttl=600)
    cached = await cache_get("weather:13.08:80.27")

    # Decorator
    @cache_prediction(ttl=120)
    async def predict_flood(lat, lon):
        ...
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import wraps
from typing import Any, Callable, Optional

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# Lazy Redis client — initialised on first use
_redis_client = None


async def _get_redis():
    """Get or create async Redis client."""
    global _redis_client
    if _redis_client is None:
        try:
            import redis.asyncio as aioredis
            _redis_client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
            logger.info("Redis connected: %s", settings.REDIS_URL)
        except Exception as e:
            logger.warning("Redis unavailable: %s — caching disabled", e)
            return None
    return _redis_client


async def cache_get(key: str) -> Optional[Any]:
    """Get a cached value by key. Returns None on miss or error."""
    client = await _get_redis()
    if not client:
        return None
    try:
        raw = await client.get(key)
        if raw is not None:
            return json.loads(raw)
    except Exception as e:
        logger.warning("Cache GET error for %s: %s", key, e)
    return None


async def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Set a cached value with optional TTL (seconds)."""
    client = await _get_redis()
    if not client:
        return False
    try:
        serialised = json.dumps(value, default=str)
        await client.set(key, serialised, ex=ttl or settings.REDIS_CACHE_TTL)
        return True
    except Exception as e:
        logger.warning("Cache SET error for %s: %s", key, e)
        return False


async def cache_delete(key: str) -> bool:
    """Delete a cache key."""
    client = await _get_redis()
    if not client:
        return False
    try:
        await client.delete(key)
        return True
    except Exception as e:
        logger.warning("Cache DELETE error for %s: %s", key, e)
        return False


async def cache_clear_prefix(prefix: str) -> int:
    """Delete all keys matching a prefix pattern."""
    client = await _get_redis()
    if not client:
        return 0
    try:
        keys = []
        async for key in client.scan_iter(f"{prefix}*"):
            keys.append(key)
        if keys:
            await client.delete(*keys)
        return len(keys)
    except Exception as e:
        logger.warning("Cache CLEAR error for %s*: %s", prefix, e)
        return 0


def _make_cache_key(prefix: str, args: tuple, kwargs: dict) -> str:
    """Deterministic cache key from function arguments."""
    raw = json.dumps({"a": args, "k": kwargs}, sort_keys=True, default=str)
    digest = hashlib.md5(raw.encode()).hexdigest()[:12]
    return f"{prefix}:{digest}"


def cache_prediction(ttl: Optional[int] = None, prefix: str = "pred"):
    """
    Decorator: cache async function results in Redis.

    Usage:
        @cache_prediction(ttl=120, prefix="flood")
        async def predict_flood(lat: float, lon: float) -> dict:
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = _make_cache_key(f"{prefix}:{func.__name__}", args, kwargs)
            cached = await cache_get(key)
            if cached is not None:
                logger.debug("Cache HIT: %s", key)
                return cached

            result = await func(*args, **kwargs)
            await cache_set(key, result, ttl=ttl or settings.REDIS_PREDICTION_TTL)
            return result
        return wrapper
    return decorator


async def close_redis() -> None:
    """Close Redis connection."""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed")
