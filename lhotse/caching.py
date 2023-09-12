from functools import lru_cache, wraps
from threading import Lock
from typing import Any, Callable, Dict, Optional

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

    # enable cache for audio files of "command" type
    AudioCache.enable(enabled)

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


class AudioCache:
    """
    Cache of 'bytes' objects with audio data.
    It is used to cache the "command" type audio inputs.

    By default it is disabled, to enable call `set_caching_enabled(True)`
    or `AudioCache.enable()`.

    The cache size is limited to max 100 elements and 500MB of audio.

    A global dict `__cache_dict` (static member variable of class AudioCache)
    is holding the wavs as 'bytes' arrays.
    The key is the 'source' identifier (i.e. the command for loading the data).

    Thread-safety is ensured by a threading.Lock guard.
    """

    __enabled: bool = False

    max_cache_memory: int = 500 * 1e6  # 500 MB
    max_cache_elements: int = 100  # 100 audio files

    __cache_dict: Dict[str, bytes] = {}
    __lock: Lock = Lock()

    @classmethod
    def enable(cls, enabled=True):
        cls.__enabled = enabled
        if not enabled:
            cls.__clear_cache()

    @classmethod
    def enabled(cls) -> bool:
        return cls.__enabled

    @classmethod
    def try_cache(cls, key: str) -> Optional[bytes]:
        """
        Test if 'key' is in the chache. If yes return the bytes array,
        otherwise return None.
        """

        if not cls.__enabled:
            return None

        with cls.__lock:
            if key in cls.__cache_dict:
                return cls.__cache_dict[key]
            else:
                return None

    @classmethod
    def add_to_cache(cls, key: str, value: bytes):
        """
        Add the new (key,value) pair to cache.
        Possibly free some elements before adding the new pair.
        The oldest elements are removed first.
        """

        if not cls.__enabled:
            return None

        if len(value) > cls.max_cache_memory:
            return

        with cls.__lock:
            # limit cache elements
            while len(cls.__cache_dict) > cls.max_cache_elements:
                # remove oldest elements from cache
                # (dict pairs are sorted according to insertion order)
                cls.__cache_dict.pop(next(iter(cls.__cache_dict)))

            # limit cache memory
            while len(value) + AudioCache.__cache_memory() > cls.max_cache_memory:
                # remove oldest elements from cache
                # (dict pairs are sorted according to insertion order)
                cls.__cache_dict.pop(next(iter(cls.__cache_dict)))

            # store the new (key,value) pair
            cls.__cache_dict[key] = value

    @classmethod
    def __cache_memory(cls) -> int:
        """
        Return size of AudioCache values in bytes.
        (internal, not to be called from outside)
        """
        ans = 0
        for key, value in cls.__cache_dict.items():
            ans += len(value)
        return ans

    @classmethod
    def __clear_cache(cls) -> None:
        """
        Clear the cache, remove the data.
        """
        with cls.__lock:
            cls.__cache_dict.clear()
