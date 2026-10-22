"""Benchmark snapshot construction and parent reconstruction without private data or GPUs."""

import argparse
import gc
import json
import pickle
import platform
import random
import statistics
import time
import tracemalloc
from types import SimpleNamespace

from lhotse.dataset.sampling.dynamic_bucketing import DynamicBucketer
from lhotse.dataset.sampling.token_codec import unpack_bucket_tokens
from lhotse.lazy import IteratorNode


class _Source(IteratorNode):
    """Supply the graph-restorable capability without reading any data."""

    has_constant_time_access = True

    def __iter__(self):
        """No source traversal is required for snapshot capture."""
        return iter(())

    def __getitem__(self, token):
        """Snapshot benchmarks must not materialize data through the restore source."""
        raise AssertionError("Unexpected source read")


def measure(fn, repeats):
    """Time ordinary GC and report collections without tracing allocation overhead."""
    gc.collect()
    before = [s["collections"] for s in gc.get_stats()]
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        del value
        durations.append(time.perf_counter() - start)
    after = [s["collections"] for s in gc.get_stats()]
    tracemalloc.start()
    value = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del value
    return {
        "median_ms": 1000 * statistics.median(durations),
        "mean_ms": 1000 * statistics.mean(durations),
        "gc_collections": [b - a for a, b in zip(before, after)],
        "peak_python_bytes_separate_sample": peak,
    }


def main():
    """Compare the unmodified path, serialize-after-capture, and direct packing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=30000)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    bucketer = DynamicBucketer(
        [],
        duration_bins=[1, 2, 3, 4, 5],
        world_size=1,
        max_cuts=4,
        rng=random.Random(0),
        restore_sources=[_Source()],
    )
    for i in range(args.items):
        token = (i % 457, (i // 457, (0, (i % 17, (i, i % 3)))))
        bucketer.buckets[i % len(bucketer.buckets)].put(
            (SimpleNamespace(_graph_origin=token),)
        )

    def wrapped():
        return {"packed_bucket_state": pickle.dumps(bucketer.get_state(), protocol=5)}

    factories = {
        "plain": bucketer.get_state,
        "serialize_after_capture": wrapped,
        "direct_compact": lambda: bucketer.get_state(compact=True),
    }
    plain = bucketer.get_state()
    compact = bucketer.get_state(compact=True)
    assert unpack_bucket_tokens(compact["bucket_tokens"]) == plain["bucket_tokens"]
    report = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "items": args.items,
        "repeats": args.repeats,
        "gc_thresholds": gc.get_threshold(),
        "variants": {},
    }
    for name, factory in factories.items():
        payload = pickle.dumps(factory(), protocol=5)
        report["variants"][name] = {
            "wire_bytes": len(payload),
            "capture": measure(factory, args.repeats),
            "capture_and_pickle": measure(
                lambda: pickle.dumps(factory(), protocol=5), args.repeats
            ),
            "parent_unpickle": measure(lambda: pickle.loads(payload), args.repeats),
        }
    report["compact_decode_for_restore"] = measure(
        lambda: unpack_bucket_tokens(compact["bucket_tokens"]), args.repeats
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
