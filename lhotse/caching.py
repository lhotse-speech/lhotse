from functools import lru_cache, wraps
from typing import Any, Callable

LHOTSE_CACHING_ENABLED = False

# This dict is holding two variants for each registered method:
# the cached variant, that uses its own LRU cache,
# and a noncached variant, that will always be recomputed.
# This allows the users to enable or disable caching and
# always get the expected behavior.
LHOTSE_CACHED_METHOD_REGISTRY = {"cached": {}, "noncached": {}}


def set_caching_enabled(enabled: bool) -> None:
    global LHOTSE_CACHING_ENABLED
    global LHOTSE_CACHED_METHOD_REGISTRY
    assert isinstance(enabled, bool)
    LHOTSE_CACHING_ENABLED = enabled
    if not enabled:
        # Caching disabled: purge all caches.
        for method in LHOTSE_CACHED_METHOD_REGISTRY["cached"].values():
            method.cache_clear()


def is_caching_enabled() -> bool:
    return LHOTSE_CACHING_ENABLED


def dynamic_lru_cache(method: Callable) -> Callable:
    """
    Least-recently-used cache decorator.

    It enhances Python's built-in ``lru_cache`` with a dynamic
    lookup of whether to apply the cached, or noncached variant
    of the decorated function.

    To disable/enable caching globally in Lhotse, call::

        >>> from lhotse import set_caching_enabled
        >>> set_caching_enabled(True)   # enable
        >>> set_caching_enabled(False)  # disable

    Currently it hard-codes the cache size at 512 items
    (regardless of the array size).

    Check :meth:`functools.lru_cache` for more details.
    """
    # Create an LRU cached variant of the method and register it
    # together with the original, noncached method.
    global LHOTSE_CACHED_METHOD_REGISTRY
    name = method.__qualname__  # example: "Recording.load_audio()"
    if name in LHOTSE_CACHED_METHOD_REGISTRY["cached"]:
        raise ValueError(
            f"Method '{name}' is already cached. "
            f"We don't support caching different methods which have "
            f"the same __qualname__ attribute (i.e., class name + method name)."
        )
    LHOTSE_CACHED_METHOD_REGISTRY["noncached"][name] = method
    LHOTSE_CACHED_METHOD_REGISTRY["cached"][name] = lru_cache(maxsize=512)(method)

    @wraps(method)
    def wrapper(*args, **kwargs) -> Any:
        # Each time the user calls this function, we will
        # dynamically dispatch the cached or noncached variant
        # of the wrapped method, depending on the global settings.
        if is_caching_enabled():
            m = LHOTSE_CACHED_METHOD_REGISTRY["cached"][name]
        else:
            m = LHOTSE_CACHED_METHOD_REGISTRY["noncached"][name]
        return m(*args, **kwargs)

    return wrapper
