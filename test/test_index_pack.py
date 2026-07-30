import json
import os
import struct
from pathlib import Path

import pytest

import lhotse.packed_lazy as packed_lazy_module
from lhotse.index_pack import (
    IndexPack,
    IndexPackCollectionSpec,
    index_pack_collection_key,
    write_index_pack,
)
from lhotse.indexing import create_jsonl_index
from lhotse.lazy import (
    GraphOriginDict,
    LazyIndexedManifestIterator,
    LazyIteratorChain,
    get_graph_origin,
)
from lhotse.packed_lazy import LazyPackedManifestIterator, read_packed_range


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_index_pack_converts_existing_sidecars_and_reads_lazily(tmp_path):
    paths = [tmp_path / "shard-0.jsonl", tmp_path / "shard-1.jsonl"]
    records = [
        [{"id": "a0"}, {"id": "a1"}],
        [{"id": "b0"}, {"id": "b1"}, {"id": "b2"}],
    ]
    for path, shard_records in zip(paths, records):
        _write_jsonl(path, shard_records)
        create_jsonl_index(path)

    source_spec = str(tmp_path / "shard-__OP_0..1_CL_.jsonl")
    spec = IndexPackCollectionSpec(
        role="manifest",
        kind="jsonl",
        source_spec=source_spec,
        paths=tuple(map(str, paths)),
    )
    pack_path = tmp_path / "dataset.idxpack"
    write_index_pack(pack_path, [spec])
    assert pack_path.read_bytes()[:16] == b"IDXPACK2\x02\x00\x00\x00\x00\x01\x00\x00"

    with IndexPack(pack_path) as pack:
        collection = pack.collection(
            index_pack_collection_key("manifest", "jsonl", source_spec)
        )
        assert len(collection) == 5

        actual = []
        for idx in range(len(collection)):
            location = collection.locate(idx)
            with open(location.path, "rb") as f:
                f.seek(location.start)
                actual.append(json.loads(f.read(location.end - location.start)))

        assert actual == [record for shard in records for record in shard]


def test_index_pack_deduplicates_segments_within_dataset(tmp_path):
    path = tmp_path / "shared.jsonl"
    records = [{"id": "a"}, {"id": "b"}]
    _write_jsonl(path, records)
    create_jsonl_index(path)

    specs = [
        IndexPackCollectionSpec(
            role="manifest",
            kind="jsonl",
            source_spec=f"alias-{idx}",
            paths=(str(path),),
        )
        for idx in range(2)
    ]
    pack_path = tmp_path / "dataset.idxpack"
    write_index_pack(pack_path, specs)

    with IndexPack(pack_path) as pack:
        assert pack.num_segments == 1
        assert pack.num_collections == 2
        for spec in specs:
            collection = pack.collection(spec.key)
            assert len(collection) == 2
            assert collection.locate(1).path == str(path)


def test_index_pack_persists_arbitrary_application_kind(tmp_path):
    source = tmp_path / "records.jsonl"
    _write_jsonl(source, [{"id": "first"}])
    create_jsonl_index(source)
    spec = IndexPackCollectionSpec(
        role="primary-records",
        kind="application/vnd.example.records+json",
        source_spec={"template": "records.jsonl"},
        paths=(str(source),),
    )

    pack_path = tmp_path / "dataset.idxpack"
    write_index_pack(pack_path, [spec])

    with IndexPack(pack_path) as pack:
        collection = pack.collection(spec.key)
        assert collection.kind == "application/vnd.example.records+json"
        assert collection.locate(0).path == str(source)


def test_index_pack_path_only_collection_needs_no_sidecars(tmp_path):
    paths = (str(tmp_path / "payload-0.tar"), str(tmp_path / "payload-1.tar"))
    spec = IndexPackCollectionSpec(
        role="payload",
        kind="application/x-tar",
        source_spec="payload-{0..1}.tar",
        paths=paths,
        offsets_required=False,
    )
    pack_path = tmp_path / "dataset.idxpack"
    write_index_pack(pack_path, [spec])

    with IndexPack(pack_path) as pack:
        collection = pack.collection(spec.key)
        assert not collection.offsets_required
        assert len(collection) == 0
        assert collection.path_for_shard(-1) == paths[-1]
        with pytest.raises(IndexError):
            collection.locate(0)


def test_index_pack_opens_v2_path_only_collection_without_collection_flag(tmp_path):
    """Early v2 writers marked path-only segments but not their collection."""
    spec = IndexPackCollectionSpec(
        role="payload",
        kind="application/x-tar",
        source_spec="payload.tar",
        paths=(str(tmp_path / "payload.tar"),),
        offsets_required=False,
    )
    pack_path = tmp_path / "dataset.idxpack"
    write_index_pack(pack_path, [spec])

    collection_flags_offset = 256 + struct.calcsize("<32sQQQQI")
    with pack_path.open("r+b") as f:
        f.seek(collection_flags_offset)
        f.write(struct.pack("<I", 0))

    with IndexPack(pack_path) as pack:
        collection = pack.collection(spec.key)
        assert not collection.offsets_required
        assert collection.path_for_shard(0) == str(tmp_path / "payload.tar")


def test_index_pack_locates_directly_within_shard(tmp_path):
    paths = [tmp_path / "shard-0.jsonl", tmp_path / "shard-1.jsonl"]
    for path, records in zip(
        paths,
        ([{"id": "a"}], [{"id": "b0"}, {"id": "b1"}]),
    ):
        _write_jsonl(path, records)
        create_jsonl_index(path)
    spec = IndexPackCollectionSpec(
        role="manifest",
        kind="jsonl",
        source_spec="two-shards",
        paths=tuple(map(str, paths)),
    )
    pack_path = tmp_path / "dataset.idxpack"
    write_index_pack(pack_path, [spec])

    with IndexPack(pack_path) as pack:
        collection = pack.collection(spec.key)
        assert collection.shard_length(1) == 2
        location = collection.locate_in_shard(1, -1)
        assert (location.shard_index, location.local_index) == (1, 1)
        assert json.loads(
            read_packed_range(pack, location.path, location.start, location.end)
        ) == {"id": "b1"}


def test_index_pack_rejects_stale_jsonl_sentinel(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"id": "a-longer-value"}])
    create_jsonl_index(path)
    _write_jsonl(path, [{"id": "short"}])
    newer = path.stat().st_mtime_ns + 2_000_000_000
    os.utime(Path(str(path) + ".idx"), ns=(newer, newer))
    spec = IndexPackCollectionSpec(
        role="manifest",
        kind="jsonl",
        source_spec=str(path),
        paths=(str(path),),
    )

    with pytest.raises(ValueError, match="Invalid sentinel"):
        write_index_pack(tmp_path / "dataset.idxpack", [spec])


def test_index_pack_rejects_newer_same_size_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"id": "one"}])
    idx = create_jsonl_index(path)
    newer = idx.stat().st_mtime_ns + 2_000_000_000
    os.utime(path, ns=(newer, newer))
    spec = IndexPackCollectionSpec(
        role="manifest",
        kind="jsonl",
        source_spec=str(path),
        paths=(str(path),),
    )

    with pytest.raises(ValueError, match="newer than index"):
        write_index_pack(tmp_path / "dataset.idxpack", [spec])


def test_index_pack_rejects_truncated_file(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"id": "a"}])
    create_jsonl_index(path)
    pack_path = tmp_path / "dataset.idxpack"
    write_index_pack(
        pack_path,
        [
            IndexPackCollectionSpec(
                role="manifest",
                kind="jsonl",
                source_spec=str(path),
                paths=(str(path),),
            )
        ],
    )
    pack_path.write_bytes(pack_path.read_bytes()[:64])

    with pytest.raises(ValueError, match="truncated|header"):
        IndexPack(pack_path)


def test_index_pack_rejects_sequence_outside_segment_table(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"id": "a"}])
    create_jsonl_index(path)
    spec = IndexPackCollectionSpec(
        role="manifest", kind="jsonl", source_spec=str(path), paths=(str(path),)
    )
    pack_path = tmp_path / "dataset.idxpack"
    write_index_pack(pack_path, [spec])
    with IndexPack(pack_path) as pack:
        sequence_offset = pack.sequence_offset
        invalid_segment_id = pack.num_segments
    contents = bytearray(pack_path.read_bytes())
    struct.pack_into("<Q", contents, sequence_offset, invalid_segment_id)
    pack_path.write_bytes(contents)

    with pytest.raises(ValueError, match="corrupt sequence"):
        IndexPack(pack_path)


def test_index_pack_rejects_incorrect_cumulative_shard_count(tmp_path):
    paths = [tmp_path / "data-0.jsonl", tmp_path / "data-1.jsonl"]
    for path in paths:
        _write_jsonl(path, [{"id": "a"}])
        create_jsonl_index(path)
    spec = IndexPackCollectionSpec(
        role="manifest",
        kind="jsonl",
        source_spec="two-shards",
        paths=tuple(map(str, paths)),
    )
    pack_path = tmp_path / "dataset.idxpack"
    write_index_pack(pack_path, [spec])
    with IndexPack(pack_path) as pack:
        second_cumulative_offset = pack.sequence_offset + struct.calcsize("<QQ") + 8
    contents = bytearray(pack_path.read_bytes())
    struct.pack_into("<Q", contents, second_cumulative_offset, 1)
    pack_path.write_bytes(contents)

    with pytest.raises(ValueError, match="corrupt cumulative count"):
        IndexPack(pack_path)


def test_index_pack_refuses_replaced_file_on_reopen(tmp_path):
    path = tmp_path / "data.jsonl"
    _write_jsonl(path, [{"id": "a"}])
    create_jsonl_index(path)
    spec = IndexPackCollectionSpec(
        role="manifest", kind="jsonl", source_spec=str(path), paths=(str(path),)
    )
    pack_path = tmp_path / "dataset.idxpack"
    replacement = tmp_path / "replacement.idxpack"
    write_index_pack(pack_path, [spec])
    write_index_pack(replacement, [spec])

    pack = IndexPack(pack_path)
    pack.close()
    os.replace(replacement, pack_path)
    with pytest.raises(RuntimeError, match="changed after it was opened"):
        pack.collection(spec.key)


def test_packed_range_retries_short_pread(tmp_path, monkeypatch):
    path = tmp_path / "data.bin"
    path.write_bytes(b"abcdefgh")
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"id": "a"}])
    create_jsonl_index(source)
    spec = IndexPackCollectionSpec(
        role="manifest", kind="jsonl", source_spec=str(source), paths=(str(source),)
    )
    pack_path = tmp_path / "dataset.idxpack"
    write_index_pack(pack_path, [spec])
    real_pread = os.pread

    def short_pread(fd, size, offset):
        return real_pread(fd, min(size, 2), offset)

    monkeypatch.setattr(packed_lazy_module.os, "pread", short_pread)
    with IndexPack(pack_path) as pack:
        assert read_packed_range(pack, str(path), 1, 7) == b"bcdefg"


@pytest.mark.parametrize("shuffle", [False, True])
def test_lazy_packed_manifest_matches_legacy_sharded_iterator(tmp_path, shuffle):
    paths = [tmp_path / "shard-0.jsonl", tmp_path / "shard-1.jsonl"]
    records = [
        [{"id": "a0"}, {"id": "a1"}],
        [{"id": "b0"}, {"id": "b1"}, {"id": "b2"}],
    ]
    for path, shard_records in zip(paths, records):
        _write_jsonl(path, shard_records)
        create_jsonl_index(path)

    source_spec = str(tmp_path / "shard-__OP_0..1_CL_.jsonl")
    spec = IndexPackCollectionSpec(
        role="manifest",
        kind="jsonl",
        source_spec=source_spec,
        paths=tuple(map(str, paths)),
    )
    pack_path = tmp_path / "dataset.idxpack"
    write_index_pack(pack_path, [spec])

    legacy_sources = [
        LazyIndexedManifestIterator(path, decode=GraphOriginDict) for path in paths
    ]
    legacy = LazyIteratorChain(
        *legacy_sources,
        shuffle_iters=shuffle,
        seed=42,
    )
    packed = LazyPackedManifestIterator(
        pack_path,
        spec.key,
        shuffle_shards=shuffle,
        seed=42,
        decode=GraphOriginDict,
    )

    legacy_items = list(legacy)
    packed_items = list(packed)
    assert [item["id"] for item in packed_items] == [
        item["id"] for item in legacy_items
    ]
    assert [get_graph_origin(item) for item in packed_items] == [
        get_graph_origin(item) for item in legacy_items
    ]


def test_lazy_packed_manifest_restores_global_shuffle(tmp_path):
    paths = [tmp_path / "shard-0.jsonl", tmp_path / "shard-1.jsonl"]
    for shard, path in enumerate(paths):
        _write_jsonl(path, [{"id": f"{shard}-{idx}"} for idx in range(20)])
        create_jsonl_index(path)
    source_spec = "two-shards"
    spec = IndexPackCollectionSpec(
        role="manifest",
        kind="jsonl",
        source_spec=source_spec,
        paths=tuple(map(str, paths)),
    )
    pack_path = tmp_path / "dataset.idxpack"
    write_index_pack(pack_path, [spec])

    source = LazyPackedManifestIterator(
        pack_path,
        spec.key,
        shuffle_shards=True,
        seed=17,
        decode=GraphOriginDict,
    )
    iterator = iter(source)
    consumed = [next(iterator)["id"] for _ in range(13)]
    assert len(consumed) == 13
    state = source.state_dict()
    expected = [item["id"] for item in iterator]

    restored = LazyPackedManifestIterator(
        pack_path,
        spec.key,
        shuffle_shards=True,
        seed=17,
        decode=GraphOriginDict,
    )
    restored.load_state_dict(state)
    assert [item["id"] for item in restored] == expected


def test_lazy_packed_manifest_restores_sequential_iteration(tmp_path):
    paths = [tmp_path / "shard-0.jsonl", tmp_path / "shard-1.jsonl"]
    for shard, path in enumerate(paths):
        _write_jsonl(path, [{"id": f"{shard}-{idx}"} for idx in range(6)])
        create_jsonl_index(path)
    spec = IndexPackCollectionSpec(
        role="manifest",
        kind="jsonl",
        source_spec="two-shards",
        paths=tuple(map(str, paths)),
    )
    pack_path = tmp_path / "dataset.idxpack"
    write_index_pack(pack_path, [spec])
    source = LazyPackedManifestIterator(pack_path, spec.key, decode=GraphOriginDict)
    iterator = iter(source)
    assert [next(iterator)["id"] for _ in range(8)] == [
        "0-0",
        "0-1",
        "0-2",
        "0-3",
        "0-4",
        "0-5",
        "1-0",
        "1-1",
    ]
    state = source.state_dict()
    expected = [item["id"] for item in iterator]
    restored = LazyPackedManifestIterator(pack_path, spec.key, decode=GraphOriginDict)
    restored.load_state_dict(state)
    assert [item["id"] for item in restored] == expected


def test_lazy_packed_manifest_warns_when_skipping_decode_error(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"id": "good"}\nnot-json\n')
    create_jsonl_index(path)
    spec = IndexPackCollectionSpec(
        role="manifest", kind="jsonl", source_spec=str(path), paths=(str(path),)
    )
    pack_path = tmp_path / "dataset.idxpack"
    write_index_pack(pack_path, [spec])
    source = LazyPackedManifestIterator(
        pack_path,
        spec.key,
        decode=GraphOriginDict,
        skip_decode_errors=True,
    )

    with pytest.warns(UserWarning, match="Skipping malformed packed manifest"):
        assert [item["id"] for item in source] == ["good"]
