"""Compact bucket snapshots preserve tokens and indexed sampler continuation."""

import copy
import io
import json
import pickle
import random
import struct
from unittest.mock import patch

import pytest
import torch

from lhotse import CutSet
from lhotse.checkpoint import DataloaderCheckpoint
from lhotse.dataset import IterableDatasetWrapper
from lhotse.dataset.sampling.dynamic_bucketing import DynamicBucketingSampler
from lhotse.dataset.sampling.token_codec import (
    _MAGIC,
    pack_bucket_tokens,
    unpack_bucket_tokens,
)
from lhotse.testing.dummies import DummyManifest


def _pack(value):
    """Exercise the same writer used for direct bucket capture."""
    return pack_bucket_tokens(value)


@pytest.mark.parametrize(
    "token",
    [
        0,
        -1,
        2**63 - 1,
        -(2**63),
        2**100,
        -(2**100),
        (),
        [],
        (1, (2, 3)),
        ("source", (4, (5, 6))),
        (1, [2, (3,)]),
        "ąę日本語",
        b"\x00\xff",
        None,
        True,
        False,
        1.25,
    ],
)
def test_token_roundtrip(token):
    """Preserve scalar types and variable token shapes through pickle transport."""
    state = [[[], [token], [token, (7, token)]], []]
    packed = _pack(state)
    assert unpack_bucket_tokens(pickle.loads(pickle.dumps(packed))) == state
    stream = io.BytesIO()
    torch.save({"bucket_tokens": packed}, stream)
    stream.seek(0)
    assert torch.load(stream, weights_only=True)["bucket_tokens"] == packed


def test_random_token_trees():
    """Exercise mixed lengths and nesting without an additional test dependency."""
    rng = random.Random(17)

    def token(depth):
        if depth == 0 or rng.random() < 0.4:
            return rng.randrange(-(2**70), 2**70)
        values = [token(depth - 1) for _ in range(rng.randrange(4))]
        return values if rng.random() < 0.5 else tuple(values)

    state = [
        [[token(5) for _ in range(rng.randrange(4))] for _ in range(30)]
        for _ in range(4)
    ]
    assert unpack_bucket_tokens(_pack(state)) == state


def test_invalid_payloads():
    """Reject truncated, unknown-version, malformed and trailing payloads."""
    packed = _pack([[[(1, (2, "source"))]], []])
    for end in range(len(packed)):
        with pytest.raises(ValueError):
            unpack_bucket_tokens(packed[:end])
    for bad in [
        packed + b"x",
        b"wrong",
        _MAGIC + b"\xff",
        _MAGIC + b"\xff\xff\xff\xff",
    ]:
        with pytest.raises(ValueError):
            unpack_bucket_tokens(bad)


def test_unsupported_tokens():
    """Invalid opt-in token types fail explicitly rather than triggering sampler replay."""
    with pytest.raises(ValueError, match="Unsupported compact"):
        _pack([[[object()]]])


def test_reject_pickle_globals():
    """Externally supplied payloads cannot reconstruct arbitrary Python objects."""
    payload = _MAGIC + struct.pack("<III", 1, 1, 1) + pickle.dumps(object())
    with pytest.raises(ValueError, match="Non-primitive"):
        unpack_bucket_tokens(payload)


@pytest.mark.parametrize("state", [[], [[]], [[[]]], [[], [[]], []]])
def test_empty_containers(state):
    """Bucket, item, and token counts may all be zero."""
    assert unpack_bucket_tokens(_pack(state)) == state


def test_pickle_memo_boundaries():
    """Repeated tokens share a memo within a bucket but not across buckets."""
    token = ("source", (123, (456, 789)))
    state = [[[token, token], [token]], [[token], [token, token]]]
    restored = unpack_bucket_tokens(_pack(state))
    assert restored == state
    assert restored[0][0][0] is restored[0][1][0]
    assert restored[1][0][0] is restored[1][1][1]


@pytest.fixture
def manifest(tmp_path):
    """Use real indexed manifests without audio access."""
    path = tmp_path / "cuts.jsonl"
    DummyManifest(CutSet, begin_id=0, end_id=160).to_jsonl(path)
    return path


def _sampler(path, compact, arity=1, concurrent=False):
    """Build a nested graph that requires the indexed restore path."""
    sources = [
        CutSet.from_file(path, indexed=True).repeat(times=2) for _ in range(arity)
    ]
    return DynamicBucketingSampler(
        *sources,
        max_cuts=4,
        duration_bins=[0.5, 1.5],
        buffer_size=32,
        shuffle=True,
        seed=42,
        compact_state=compact,
        concurrent=concurrent,
    )


def _ids(batch):
    """Retain source grouping when comparing paired/triplet batches."""
    return tuple(
        tuple(c.id for c in cs)
        for cs in (batch if isinstance(batch, tuple) else (batch,))
    )


def _remaining(iterator):
    """Consume without calling iter() again, which starts a new sampler epoch."""
    batches = []
    while True:
        try:
            batches.append(_ids(next(iterator)))
        except StopIteration:
            return batches


@pytest.mark.parametrize("arity", [1, 2, 3])
@pytest.mark.parametrize(
    "save_compact,load_compact",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_indexed_continuation(manifest, arity, save_compact, load_compact):
    """Packed and legacy states restore the exact sequence without replay."""
    sampler = _sampler(manifest, save_compact, arity)
    iterator = iter(sampler)
    for _ in range(7):
        next(iterator)
    state = copy.deepcopy(sampler.state_dict())
    expected = _remaining(iterator)
    restored = _sampler(manifest, load_compact, arity)
    restored.load_state_dict(copy.deepcopy(state))
    pending = restored.state_dict()
    assert isinstance(pending["bucketer_state"]["bucket_tokens"], bytes) == load_compact
    with patch.object(
        restored, "_replay_step", side_effect=AssertionError("Unexpected replay")
    ):
        assert [_ids(batch) for batch in restored] == expected


def test_direct_capture_and_snapshot_independence(manifest):
    """The compact path does not construct the legacy bucket lists."""
    sampler = _sampler(manifest, True)
    iterator = iter(sampler)
    next(iterator)
    bucketer = sampler._bucketer
    expected = bucketer.get_state()
    with patch.object(
        bucketer,
        "_get_plain_bucket_tokens",
        side_effect=AssertionError("Legacy capture"),
    ):
        state = bucketer.get_state(compact=True)
    payload = state["bucket_tokens"]
    state["bucket_tokens"] = unpack_bucket_tokens(payload)
    assert state == expected
    for _ in range(10):
        next(iterator)
    assert unpack_bucket_tokens(payload) == expected["bucket_tokens"]


def test_capture_with_concurrent_producer(manifest):
    """Capture valid tokens while the normal producer refills bucket queues."""
    sampler = _sampler(manifest, True, concurrent=True)
    iterator = iter(sampler)
    try:
        for _ in range(12):
            next(iterator)
            tokens = unpack_bucket_tokens(
                sampler.state_dict()["bucketer_state"]["bucket_tokens"]
            )
            assert len(tokens) == 3
            for bucket in tokens:
                for item in bucket:
                    assert len(item) == 1
                    assert sampler._bucketer.restore_sources[0][item[0]].id
    finally:
        sampler.cuts_iter.close()


def test_unsupported_capture_does_not_fall_back_to_replay(manifest):
    """The sampler's fallback catches TypeError, so compact validation uses ValueError."""
    sampler = _sampler(manifest, True)
    next(iter(sampler))
    with patch.object(sampler._bucketer, "_capture_item_token", return_value=object()):
        with pytest.raises(ValueError, match="Unsupported compact"):
            sampler.state_dict()


def test_json_export_is_legacy(manifest, tmp_path):
    """JSON checkpoints remain readable without knowing the compact format."""
    sampler = _sampler(manifest, True)
    iterator = iter(sampler)
    for _ in range(5):
        next(iterator)
    state = copy.deepcopy(sampler.state_dict())
    expected = _remaining(iterator)
    path = tmp_path / "state.json"
    checkpoint = DataloaderCheckpoint(0, 1, 0, sampler_state=state)
    original_state = copy.deepcopy(state)
    checkpoint.save(path)
    assert checkpoint.sampler_state == original_state
    assert isinstance(
        json.loads(path.read_text())["sampler_state"]["bucketer_state"][
            "bucket_tokens"
        ],
        list,
    )
    restored = _sampler(manifest, False)
    # Isolate token serialization from existing JSON handling of diagnostic keys and selection RNG state.
    exported = DataloaderCheckpoint.load(path).sampler_state["bucketer_state"][
        "bucket_tokens"
    ]
    state["bucketer_state"]["bucket_tokens"] = exported
    restored.load_state_dict(state)
    assert [_ids(batch) for batch in restored] == expected


@pytest.mark.parametrize(
    "value", [b"ordinary bytes", _pack([[[1]]])], ids=["ordinary", "compact_payload"]
)
@pytest.mark.parametrize("location", ["sampler", "worker", "nested", "bucket_metadata"])
def test_json_export_rejects_unrelated_bytes(tmp_path, value, location):
    """A byte string outside the sampler's token field is not a compact snapshot."""
    checkpoint = DataloaderCheckpoint(0, 1, 0)
    if location == "sampler":
        checkpoint.sampler_state = {"custom": value}
    elif location == "worker":
        checkpoint.worker_states = [{"custom": value}]
    elif location == "nested":
        checkpoint.sampler_state = {
            "custom": {"bucketer_state": {"bucket_tokens": value}}
        }
    else:
        checkpoint.sampler_state = {
            "bucketer_state": {"bucket_tokens": _pack([[[1]]]), "custom": value}
        }

    with pytest.raises(
        TypeError, match="Object of type bytes is not JSON serializable"
    ):
        checkpoint.save(tmp_path / "state.json")


@pytest.mark.parametrize(
    "value", [b"ordinary bytes", _pack([[[1]]])], ids=["ordinary", "compact_payload"]
)
@pytest.mark.parametrize("compact", [False, True])
def test_json_export_rejects_byte_tokens(tmp_path, value, compact):
    """Opaque token bytes retain the legacy JSON error, even if they resemble a snapshot."""
    tokens = [[[("source", value)]]]
    checkpoint = DataloaderCheckpoint(
        0,
        1,
        0,
        sampler_state={
            "bucketer_state": {"bucket_tokens": _pack(tokens) if compact else tokens}
        },
    )

    with pytest.raises(
        TypeError, match="Object of type bytes is not JSON serializable"
    ):
        checkpoint.save(tmp_path / "state.json")


def test_json_export_rejects_invalid_compact_state(tmp_path):
    """Malformed bytes in the compact token field still report a decoder error."""
    checkpoint = DataloaderCheckpoint(
        0, 1, 0, sampler_state={"bucketer_state": {"bucket_tokens": b"invalid"}}
    )

    with pytest.raises(ValueError, match="Invalid compact bucket token header"):
        checkpoint.save(tmp_path / "state.json")


class _IdsDataset:
    """Avoid audio and tensor work in the multiprocessing correctness test."""

    def __getitem__(self, batch):
        return [c.id for c in batch]


@pytest.mark.parametrize(
    "save_compact,load_compact", [(False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("workers", [0, 2])
def test_stateful_dataloader(manifest, save_compact, load_compact, workers):
    """Real worker snapshots preserve continuation across representation changes."""
    StatefulDataLoader = pytest.importorskip(
        "torchdata.stateful_dataloader"
    ).StatefulDataLoader

    def loader(compact):
        return StatefulDataLoader(
            IterableDatasetWrapper(_IdsDataset(), _sampler(manifest, compact)),
            batch_size=None,
            num_workers=workers,
            **({"multiprocessing_context": "spawn", "timeout": 30} if workers else {}),
        )

    original = loader(save_compact)
    iterator = iter(original)
    try:
        for _ in range(8):
            next(iterator)
        state = copy.deepcopy(original.state_dict())
        expected = [next(iterator) for _ in range(12)]
    finally:
        if workers:
            iterator._shutdown_workers()
    restored = loader(load_compact)
    restored.load_state_dict(state)
    iterator = iter(restored)
    try:
        assert [next(iterator) for _ in range(12)] == expected
    finally:
        if workers:
            iterator._shutdown_workers()
