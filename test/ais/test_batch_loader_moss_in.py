"""Schema-drift guard for ``AISBatchLoader._append_moss_in``'s fast path.

The fast path bypasses Pydantic v2 validation by calling
``MossIn.model_construct(...)`` directly and appending to
``batch.request.moss_in``. If a future ``aistore`` SDK adds a required
field, renames one of the fields we pass, or restructures
``batch.request.moss_in`` away from a list, the bypass silently produces
invalid requests. These tests pin the assumption and fail loudly on drift.

Run with::

    pytest test/ais/test_batch_loader_moss_in.py -v
"""

from __future__ import annotations

import pytest

aistore = pytest.importorskip("aistore")

from aistore.sdk.batch.types import MossIn, MossReq


# ---------------------------------------------------------------------------
# Field presence: the fast path passes specific kwargs to MossIn — assert
# they're all real fields on the current SDK so a rename surfaces here.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["obj_name", "bck", "provider", "archpath", "start", "length"],
)
def test_mossin_has_required_field(field: str):
    """Every kwarg ``_append_moss_in`` passes must exist on ``MossIn``."""
    assert field in MossIn.model_fields, (
        f"AISBatchLoader._append_moss_in passes '{field}' to "
        f"MossIn.model_construct, but the field is missing on aistore "
        f"{aistore.__version__}. Update the fast path or pin the SDK."
    )


def test_mossin_obj_name_is_required():
    """``obj_name`` is the only required field today; if more get added,
    the fast path's kwargs-sparse build silently produces an invalid request."""
    required = [
        name
        for name, info in MossIn.model_fields.items()
        if info.is_required()
    ]
    assert set(required) <= {"obj_name"}, (
        f"MossIn introduced new required fields {set(required) - {'obj_name'}} "
        f"in aistore {aistore.__version__}. AISBatchLoader._append_moss_in's "
        f"sparse-kwargs build will skip them and produce invalid requests. "
        f"Update the fast path."
    )


# ---------------------------------------------------------------------------
# Round-trip equivalence: model_construct() must produce the same payload
# as the validating constructor for every (kwargs-sparse) combination the
# fast path actually emits.
# ---------------------------------------------------------------------------


_FAST_KWARG_COMBOS = [
    # (description, kwargs)
    ("url-only (e.g. recording.source.type='url')",
     {"obj_name": "audio.wav", "bck": "bkt", "provider": "ais"}),
    ("url with archpath (tar member)",
     {"obj_name": "shard.tar", "bck": "bkt", "provider": "ais", "archpath": "rec1.wav"}),
    ("shar_ptr (byte-range)",
     {"obj_name": "shard.tar", "bck": "bkt", "provider": "ais", "start": 4096, "length": 65536}),
    ("aws provider variant",
     {"obj_name": "shard.tar", "bck": "bkt", "provider": "aws"}),
]


@pytest.mark.parametrize("desc,kwargs", _FAST_KWARG_COMBOS, ids=[c[0] for c in _FAST_KWARG_COMBOS])
def test_model_construct_matches_validating_constructor(desc: str, kwargs: dict):
    """``MossIn.model_construct(**kwargs)`` must serialize identically to
    ``MossIn(**kwargs)`` — same field values, same defaults for omitted
    fields. If they ever diverge, the fast path is no longer a no-op."""
    fast = MossIn.model_construct(**kwargs)
    slow = MossIn(**kwargs)
    assert fast.model_dump() == slow.model_dump(), (
        f"MossIn.model_construct dump differs from MossIn(...) dump for {desc!r}.\n"
        f"  fast: {fast.model_dump()}\n  slow: {slow.model_dump()}"
    )


def test_model_construct_dump_json_matches_validating_constructor():
    """Wire-format equivalence: the SDK serializes via ``model_dump_json``
    or equivalent; that's the surface the AIStore proxy actually sees."""
    kwargs = {"obj_name": "x.tar", "bck": "b", "provider": "ais", "archpath": "y.wav"}
    fast = MossIn.model_construct(**kwargs).model_dump_json()
    slow = MossIn(**kwargs).model_dump_json()
    assert fast == slow, f"JSON serialization differs: fast={fast!r} slow={slow!r}"


# ---------------------------------------------------------------------------
# Container shape: the fast path appends to ``batch.requests_list`` (public
# property returning ``MossReq.moss_in``) — assert that ``MossReq.moss_in``
# exists, is list-typed, and that ``Batch.requests_list`` is the accessor.
# We can't easily instantiate ``Batch`` without an AIStore client, so we
# verify the shape at the type-system level instead.
# ---------------------------------------------------------------------------


def test_mossreq_has_moss_in_list_field():
    """``MossReq.moss_in`` is the underlying ``List[MossIn]`` we append to
    via ``batch.requests_list``. If this field name or type changes, the
    fast path breaks."""
    assert "moss_in" in MossReq.model_fields, (
        f"MossReq.moss_in missing on aistore {aistore.__version__}; "
        f"AISBatchLoader fast path uses batch.requests_list which delegates "
        f"to this field."
    )
    annotation = MossReq.model_fields["moss_in"].annotation
    # typing.List[MossIn] subscripts; check origin is list-like.
    import typing as _t
    assert _t.get_origin(annotation) is list, (
        f"MossReq.moss_in type changed from List[MossIn] to {annotation} "
        f"on aistore {aistore.__version__}. Fast path's append() may break."
    )


def test_batch_requests_list_is_public_accessor():
    """``Batch.requests_list`` must be a ``@property`` returning the
    underlying ``MossReq.moss_in`` list — that's what the fast path
    mutates. If it's renamed or turned into a getter that returns a copy,
    the fast path silently no-ops."""
    from aistore.sdk.batch.batch import Batch

    descriptor = vars(Batch).get("requests_list")
    assert isinstance(descriptor, property), (
        f"Batch.requests_list is {type(descriptor).__name__}, expected property "
        f"on aistore {aistore.__version__}."
    )
    # Sanity-check the getter body: must reach back into MossReq.moss_in.
    # Most robust check is source inspection (kept lenient — only flag a hard
    # rename of 'moss_in', anything else gets caught by the round-trip tests).
    import inspect
    src = inspect.getsource(descriptor.fget)
    assert "moss_in" in src, (
        f"Batch.requests_list source no longer references moss_in on "
        f"aistore {aistore.__version__}: {src!r}"
    )
