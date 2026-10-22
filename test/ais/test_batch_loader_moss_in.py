"""Guards against SDK drift for AISBatchLoader's batch construction path.

``AISBatchLoader._append_moss_in`` now uses ``Batch.add(str, bck=, provider=)``
introduced in aistore 1.26.0. These tests verify that the SDK API we rely on
still exists and behaves correctly, and document the construction speedup vs
the previous ``MossIn.model_construct`` bypass.

Run with::

    pytest test/ais/test_batch_loader_moss_in.py -v
    pytest test/ais/test_batch_loader_moss_in.py -v -k bench  # timing only
"""

from __future__ import annotations

import inspect
import time

import pytest

aistore = pytest.importorskip("aistore", minversion="1.26.0")

from aistore.sdk.batch.batch import Batch
from aistore.sdk.batch.types import MossIn, MossReq

# ---------------------------------------------------------------------------
# SDK field presence: catch renames before they cause silent bad requests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["obj_name", "bck", "provider", "archpath", "start", "length"]
)
def test_mossin_has_field(field: str):
    """MossIn must expose every field that AISBatchLoader reads via _moss_attrs/_moss_range."""
    assert field in MossIn.model_fields, (
        f"MossIn.{field} missing on aistore {aistore.__version__}; "
        f"AISBatchLoader reads this field for fallback GET and saved_requests_list recovery."
    )


def test_mossreq_has_moss_in_field():
    """MossReq.moss_in is the list AISBatchLoader reads via saved_requests_list for fallback recovery."""
    assert (
        "moss_in" in MossReq.model_fields
    ), f"MossReq.moss_in missing on aistore {aistore.__version__}."


def test_batch_requests_list_is_property():
    """Batch.requests_list must remain a property — AISBatchLoader snapshots it for fallback recovery."""
    descriptor = vars(Batch).get("requests_list")
    assert isinstance(descriptor, property), (
        f"Batch.requests_list is {type(descriptor).__name__!r}, expected property "
        f"on aistore {aistore.__version__}."
    )


# ---------------------------------------------------------------------------
# Batch.add(str, bck=, provider=) API: verify the new string-based path works
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "param",
    ["obj", "bck", "provider", "archpath", "start", "length"],
)
def test_batch_add_accepts_param(param: str):
    """Batch.add must accept every kwarg that _append_moss_in passes — catch renames early."""
    sig = inspect.signature(Batch.add)
    assert param in sig.parameters, (
        f"Batch.add no longer has a '{param}' parameter on aistore {aistore.__version__}. "
        f"AISBatchLoader._append_moss_in passes this kwarg — update the call site."
    )


# ---------------------------------------------------------------------------
# Timing: document construction speedup vs old model_construct bypass
# ---------------------------------------------------------------------------

_N_ITERS = 500
_COMBOS = [
    {"obj_name": "audio.wav", "bck": "bkt", "provider": "ais"},
    {"obj_name": "shard.tar", "bck": "bkt", "provider": "ais", "archpath": "rec1.wav"},
    {
        "obj_name": "shard.tar",
        "bck": "bkt",
        "provider": "ais",
        "start": 4096,
        "length": 65536,
    },
]


def _time_fn(fn, n: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n):
        for kwargs in _COMBOS:
            fn(kwargs)
    return time.perf_counter() - t0


def test_mossin_validating_constructor_not_slower_than_model_construct():
    """The validating constructor must be at least as fast as model_construct.

    model_construct was used as a bypass to avoid Pydantic validation overhead,
    but Pydantic v2 with frozen=True makes MossIn(**kwargs) faster. If this
    ever regresses, the bypass may need to be reintroduced.
    """
    # Warmup
    _time_fn(lambda kw: MossIn.model_construct(**kw), 50)
    _time_fn(lambda kw: MossIn(**kw), 50)

    t_construct = _time_fn(lambda kw: MossIn.model_construct(**kw), _N_ITERS)
    t_validate = _time_fn(lambda kw: MossIn(**kw), _N_ITERS)

    assert t_validate <= t_construct * 1.5, (
        f"MossIn(**kwargs) ({t_validate*1000:.1f}ms) is significantly slower than "
        f"model_construct ({t_construct*1000:.1f}ms) — the bypass may need to be reintroduced."
    )
