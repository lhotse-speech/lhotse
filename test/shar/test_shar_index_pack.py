"""Tests for reading Lhotse Shar through an ``.idxpack`` (``lhotse.shar.index_pack``)."""

import gzip
import itertools
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import lhotse.indexing
from lhotse import CutSet
from lhotse.bin.modes.cli_base import cli
from lhotse.index_pack import IndexPack, open_index_pack
from lhotse.shar import SharWriter, write_shar_index_pack
from lhotse.shar.index_pack import shar_source_spec
from lhotse.shar.readers.indexed import LazyIndexedSharIterator
from lhotse.testing.dummies import DummyManifest
from lhotse.utils import fastcopy

FIELDS = {
    "recording": "wav",
    "features": "numpy",
    "custom_embedding": "numpy",
    "custom_features": "numpy",
    "custom_indexes": "numpy",
    "custom_recording": "wav",
    "label": "jsonl",
}
NUM_CUTS = 20
SHARD_SIZE = 7  # uneven shards: 7 + 7 + 6


def _write_shar(path: Path, begin_id: int = 0, compress_jsonl: bool = False) -> Path:
    path.mkdir(parents=True)
    cuts = DummyManifest(
        CutSet, begin_id=begin_id, end_id=begin_id + NUM_CUTS, with_data=True
    )
    with SharWriter(
        path,
        fields=FIELDS,
        shard_size=SHARD_SIZE,
        compress_jsonl=compress_jsonl,
        create_index=True,
    ) as writer:
        for cut in cuts:
            cut.label = f"label-{cut.id}"
            writer.write(cut)
    return path


def _mapping(shar_dir: Path, fields=None) -> dict:
    """The per-field mapping equivalent to a Shar directory (optionally a subset of its fields)."""
    names = ["cuts", *FIELDS] if fields is None else ["cuts", *fields]
    out = {}
    for name in names:
        out[name] = sorted(
            str(p)
            for p in shar_dir.glob(f"{name}.*")
            if not p.name.endswith((".idx", ".gzidx"))
        )
    return out


def _source(shar_dir: Path, mode: str):
    """Reader keyword arguments selecting the source in ``in_dir`` or ``fields`` mode."""
    if mode == "in_dir":
        return {"in_dir": shar_dir}
    return {"fields": _mapping(shar_dir)}


def _signature(cut):
    return (
        cut.id,
        cut.recording.sources[0].source,
        cut.features.storage_key,
        cut.label,
        str(cut.shard_origin),
        cut.shar_epoch,
    )


def _pack(tmp_path, *sources, name="pack.idxpack", **kwargs) -> Path:
    path = tmp_path / name
    write_shar_index_pack(path, list(sources), **kwargs)
    return path


@pytest.fixture
def shar_dir(tmp_path):
    return _write_shar(tmp_path / "shar")


@pytest.mark.parametrize("mode", ["in_dir", "fields"])
@pytest.mark.parametrize("shuffle", [False, True])
def test_packed_matches_sidecar_reader(tmp_path, shar_dir, mode, shuffle):
    source = _source(shar_dir, mode)
    pack = _pack(tmp_path, source["in_dir"] if mode == "in_dir" else source["fields"])
    expected = LazyIndexedSharIterator(**source, shuffle=shuffle, seed=7)
    actual = LazyIndexedSharIterator(**source, shuffle=shuffle, seed=7, index_pack=pack)
    assert len(actual) == len(expected) == NUM_CUTS
    assert actual.fields == expected.fields == set(FIELDS)
    assert actual.num_shards == expected.num_shards == 3
    for _ in range(2):  # the second epoch is reshuffled identically
        a, b = list(expected), list(actual)
        assert [_signature(c) for c in a] == [_signature(c) for c in b]
        assert a[3].to_dict() == b[3].to_dict()
    if shuffle:
        assert [c.id for c in b] != sorted(c.id for c in b)
    assert expected.state_dict() == actual.state_dict()


@pytest.mark.parametrize("mode", ["in_dir", "fields"])
def test_packed_lazy_pointers(tmp_path, shar_dir, mode):
    source = _source(shar_dir, mode)
    pack = _pack(tmp_path, source["in_dir"] if mode == "in_dir" else source["fields"])
    expected = list(LazyIndexedSharIterator(**source, lazy=True))
    actual = list(LazyIndexedSharIterator(**source, lazy=True, index_pack=pack))
    assert [c.id for c in actual] == [c.id for c in expected]
    for e, a in zip(expected, actual):
        assert a.recording.sources[0].source == e.recording.sources[0].source
        assert a.features.storage_key == e.features.storage_key
        np.testing.assert_array_equal(a.load_audio(), e.load_audio())
        np.testing.assert_array_equal(a.load_features(), e.load_features())
    # Pointers into the last pair of a tar also address its trailing end-of-archive padding.
    assert actual[-1].load_audio().shape == expected[-1].load_audio().shape


def test_random_access_and_negative_index(tmp_path, shar_dir):
    pack = _pack(tmp_path, shar_dir)
    expected = LazyIndexedSharIterator(in_dir=shar_dir)
    actual = LazyIndexedSharIterator(in_dir=shar_dir, index_pack=pack)
    for idx in (0, 6, 7, 13, 14, 19, -1, -20):
        assert _signature(actual[idx]) == _signature(expected[idx])
    # Graph restore tokens are accepted as well.
    assert actual[(5, 3)].shar_epoch == 3
    with pytest.raises(IndexError):
        actual[NUM_CUTS]
    with pytest.raises(IndexError):
        actual[-NUM_CUTS - 1]


@pytest.mark.parametrize("mode", ["in_dir", "fields"])
def test_state_dict_resume_cross_restore_and_pickle(tmp_path, shar_dir, mode):
    source = _source(shar_dir, mode)
    pack = _pack(tmp_path, source["in_dir"] if mode == "in_dir" else source["fields"])

    def packed():
        return LazyIndexedSharIterator(**source, shuffle=True, seed=3, index_pack=pack)

    def sidecar():
        return LazyIndexedSharIterator(**source, shuffle=True, seed=3)

    full = [c.id for c in sidecar()]
    assert [c.id for c in packed()] == full

    for first, second in [(packed, packed), (packed, sidecar), (sidecar, packed)]:
        it = first()
        head = [c.id for c in itertools.islice(iter(it), 8)]
        state = it.state_dict()
        resumed = pickle.loads(pickle.dumps(second()))
        resumed.load_state_dict(state)
        assert head + [c.id for c in resumed] == full

    # A pickled reader that is already in use keeps working in another "process".
    reader = packed()
    next(iter(reader))
    clone = pickle.loads(pickle.dumps(reader))
    assert _signature(clone[4]) == _signature(reader[4])


def test_multi_source_pack_serves_several_readers(tmp_path):
    first = _write_shar(tmp_path / "first", begin_id=0)
    second = _write_shar(tmp_path / "second", begin_id=100)
    pack = _pack(
        tmp_path, first, str(second) + "/"
    )  # a trailing slash is not significant
    for shar_dir in (first, second):
        expected = list(LazyIndexedSharIterator(in_dir=shar_dir))
        actual = list(LazyIndexedSharIterator(in_dir=shar_dir, index_pack=pack))
        assert [_signature(c) for c in actual] == [_signature(c) for c in expected]
    ids = {c.id for c in CutSet.from_shar(in_dir=second, index_pack=pack)}
    assert ids == {c.id for c in LazyIndexedSharIterator(in_dir=second)}
    assert len(ids) == NUM_CUTS

    # One pack object is shared by the readers; mixing them is a regular CutSet.mux.
    shared = open_index_pack(pack)
    mixed = CutSet.mux(
        CutSet.from_shar(in_dir=first, index_pack=shared),
        CutSet.from_shar(in_dir=second, index_pack=shared),
        seed=0,
    )
    assert len(list(mixed.repeat(1))) > 0


def test_source_identity(tmp_path, shar_dir):
    assert shar_source_spec(in_dir=shar_dir) == shar_source_spec(in_dir=str(shar_dir))
    assert shar_source_spec(in_dir=str(shar_dir) + "/") == shar_source_spec(
        in_dir=shar_dir
    )
    mapping = _mapping(shar_dir)
    reordered = dict(reversed(list(mapping.items())))
    assert shar_source_spec(fields=mapping) == shar_source_spec(fields=reordered)
    assert shar_source_spec(fields=mapping) != shar_source_spec(in_dir=shar_dir)
    pack = _pack(tmp_path, shar_dir)
    # The pack was written from the directory, so the same files addressed as a mapping are a different source.
    with pytest.raises(ValueError, match="no Shar source"):
        LazyIndexedSharIterator(fields=mapping, index_pack=pack)
    with pytest.raises(ValueError, match="no Shar source"):
        LazyIndexedSharIterator(in_dir=tmp_path / "other", index_pack=pack)
    with pytest.raises(ValueError, match="more than once"):
        _pack(tmp_path, shar_dir, shar_dir, name="dup.idxpack")
    with pytest.raises(ValueError):
        shar_source_spec()
    with pytest.raises(ValueError):
        shar_source_spec(fields={"recording": []})


def test_metadata_only_mapping(tmp_path, shar_dir):
    cuts_only = _mapping(shar_dir, fields=[])
    assert list(cuts_only) == ["cuts"]
    pack = _pack(tmp_path, cuts_only)
    expected = list(LazyIndexedSharIterator(fields=cuts_only))
    reader = LazyIndexedSharIterator(fields=cuts_only, index_pack=pack)
    assert reader.fields == set()
    actual = list(reader)
    assert [c.to_dict() for c in actual] == [c.to_dict() for c in expected]
    assert not actual[0].recording.sources[0].type == "memory"


def test_field_subset_and_unknown_field(tmp_path, shar_dir):
    subset = _mapping(shar_dir, fields=["recording"])
    pack = _pack(
        tmp_path, shar_dir, subset
    )  # files shared with the directory are stored once
    cut = LazyIndexedSharIterator(fields=subset, index_pack=pack)[0]
    assert cut.recording.sources[0].type == "memory"
    assert cut.features.storage_type != "memory_npy"
    assert cut.label == f"label-{cut.id}"  # carried by the cuts manifest itself
    with pytest.raises(ValueError, match="no Shar source"):
        LazyIndexedSharIterator(
            fields=_mapping(shar_dir, fields=["recording", "nope"]), index_pack=pack
        )
    with pytest.raises(ValueError, match="no Shar source"):
        LazyIndexedSharIterator(
            fields=_mapping(shar_dir, fields=["features"]), index_pack=pack
        )


def test_pack_does_not_open_sidecars(tmp_path, shar_dir, monkeypatch):
    pack = _pack(tmp_path, shar_dir)
    expected = [_signature(c) for c in LazyIndexedSharIterator(in_dir=shar_dir)]
    for sidecar in shar_dir.glob("*.idx"):
        sidecar.unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("A sidecar reader was used")

    monkeypatch.setattr(lhotse.indexing, "read_index", forbidden)
    monkeypatch.setattr(
        "lhotse.shar.readers.indexed.IndexedJsonlReader.__init__", forbidden
    )
    monkeypatch.setattr(
        "lhotse.shar.readers.indexed.IndexedTarReader.__init__", forbidden
    )
    reader = LazyIndexedSharIterator(in_dir=shar_dir, index_pack=pack, shuffle=True)
    assert sorted(_signature(c) for c in reader) == sorted(expected)
    assert not list(shar_dir.glob("*.idx"))  # nothing was auto-created either


def test_pack_inside_shar_dir_is_not_a_field(tmp_path, shar_dir):
    write_shar_index_pack(shar_dir / "shar.idxpack", shar_dir)
    assert len(list(LazyIndexedSharIterator(in_dir=shar_dir))) == NUM_CUTS
    reader = LazyIndexedSharIterator(
        in_dir=shar_dir, index_pack=shar_dir / "shar.idxpack"
    )
    assert reader.fields == set(FIELDS)


@pytest.mark.parametrize("mode", ["in_dir", "fields"])
def test_gzip_cuts(tmp_path, mode):
    pytest.importorskip("indexed_gzip")
    shar_dir = _write_shar(tmp_path / "gz", compress_jsonl=True)
    assert len(list(shar_dir.glob("cuts.*.jsonl.gz"))) == 3
    source = _source(shar_dir, mode)
    pack = _pack(tmp_path, source["in_dir"] if mode == "in_dir" else source["fields"])
    expected = [_signature(c) for c in LazyIndexedSharIterator(**source, shuffle=True)]
    for sidecar in itertools.chain(shar_dir.glob("*.idx"), shar_dir.glob("*.gzidx")):
        sidecar.unlink()
    actual = LazyIndexedSharIterator(**source, shuffle=True, index_pack=pack)
    assert [_signature(c) for c in actual] == expected


def test_trailing_zero_padding_is_accepted(tmp_path, shar_dir):
    """The file carries zero padding that lies beyond the end of its sidecar."""
    expected = [_signature(c) for c in LazyIndexedSharIterator(in_dir=shar_dir)]
    tar = sorted(shar_dir.glob("recording.*.tar"))[-1]
    size = tar.stat().st_size
    assert np.fromfile(str(tar) + ".idx", dtype="<u8")[-1] == size
    with open(tar, "ab") as f:
        f.write(b"\0" * 2048)
    pack = _pack(tmp_path, shar_dir)
    reader = LazyIndexedSharIterator(in_dir=shar_dir, index_pack=pack)
    assert [_signature(c) for c in reader] == expected
    # The pack records the physical size, and the last pair's byte range extends over the padding.
    from lhotse.shar.index_pack import shar_field_collection_key

    collection = open_index_pack(pack).collection(
        shar_field_collection_key(shar_source_spec(in_dir=shar_dir), "recording", "tar")
    )
    assert collection.source_size_for_shard(-1) == size + 2048
    assert collection.locate(-1).end == size + 2048


def test_nonzero_trailing_bytes_are_rejected(tmp_path, shar_dir):
    tar = sorted(shar_dir.glob("features.*.tar"))[0]
    with open(tar, "ab") as f:
        f.write(b"\x01" * 1024)
    with pytest.raises(ValueError, match="non-zero bytes"):
        _pack(tmp_path, shar_dir)
    assert not (tmp_path / "pack.idxpack").exists()


def test_compressed_tar_is_rejected(tmp_path, shar_dir):
    tar = sorted(shar_dir.glob("recording.*.tar"))[0]
    compressed = Path(str(tar) + ".gz")
    with open(tar, "rb") as src, gzip.open(compressed, "wb") as dst:
        shutil.copyfileobj(src, dst)
    tar.unlink()
    Path(str(tar) + ".idx").unlink()
    with pytest.raises(ValueError, match="not randomly addressable"):
        _pack(tmp_path, shar_dir)
    mapping = _mapping(shar_dir)
    mapping["recording"][0] = str(compressed)
    with pytest.raises(ValueError, match="not randomly addressable"):
        _pack(tmp_path, mapping)


def test_inconsistent_sources_are_rejected(tmp_path, shar_dir):
    mapping = _mapping(shar_dir)
    # Shard counts differ across fields.
    with pytest.raises(ValueError, match="shards"):
        _pack(tmp_path, {**mapping, "features": mapping["features"][:-1]})
    # Records per shard differ (shards of 7 and 6 cuts are swapped).
    swapped = dict(mapping, features=[mapping["features"][i] for i in (2, 1, 0)])
    with pytest.raises(ValueError, match="records"):
        _pack(tmp_path, swapped)
    # Pipes do not support random access.
    with pytest.raises(ValueError, match="seekable"):
        _pack(tmp_path, {"cuts": ["pipe:cat cuts.000000.jsonl"]})
    with pytest.raises(ValueError, match="list of shards"):
        _pack(tmp_path, {"cuts": "cuts.000000.jsonl"})
    with pytest.raises(ValueError, match="No Shar cuts"):
        _pack(tmp_path, tmp_path)


def test_missing_sidecar_is_reported(tmp_path, shar_dir):
    next(iter(shar_dir.glob("features.*.idx"))).unlink()
    with pytest.raises(FileNotFoundError, match="lhotse index shar"):
        _pack(tmp_path, shar_dir)


def test_stale_sidecar_is_rejected(tmp_path, shar_dir):
    import os
    import time

    tar = sorted(shar_dir.glob("features.*.tar"))[0]
    future = time.time() + 100
    os.utime(tar, (future, future))
    with pytest.raises(ValueError, match="newer than"):
        _pack(tmp_path, shar_dir)


def test_sidecars_in_indexes_root(tmp_path, shar_dir):
    root = tmp_path / "idx"
    for sidecar in list(shar_dir.glob("*.idx")):
        target = root / sidecar.relative_to(sidecar.anchor)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(sidecar, target)
    pack = _pack(tmp_path, shar_dir, indexes_root=root)
    expected = [
        c.id for c in LazyIndexedSharIterator(in_dir=shar_dir, indexes_root=root)
    ]
    actual = LazyIndexedSharIterator(in_dir=shar_dir, index_pack=pack)
    assert [c.id for c in actual] == expected


def test_index_pack_is_exclusive_with_sidecar_options(tmp_path, shar_dir):
    pack = _pack(tmp_path, shar_dir)
    with pytest.raises(ValueError, match="mutually exclusive"):
        LazyIndexedSharIterator(in_dir=shar_dir, index_pack=pack, index_path=tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        LazyIndexedSharIterator(in_dir=shar_dir, index_pack=pack, indexes_root=tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        CutSet.from_shar(in_dir=shar_dir, index_pack=pack, index_path=tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        CutSet.from_shar(in_dir=shar_dir, index_pack=pack, indexes_root=tmp_path)
    with pytest.raises(ValueError, match="indexed=False"):
        CutSet.from_shar(in_dir=shar_dir, index_pack=pack, indexed=False)
    with pytest.raises(ValueError, match="max_open_files"):
        LazyIndexedSharIterator(
            in_dir=shar_dir, index_pack=pack, index_pack_max_open_files=0
        )
    with pytest.raises(AssertionError):
        LazyIndexedSharIterator(index_pack=pack)


@pytest.mark.parametrize("mode", ["in_dir", "fields"])
def test_from_shar_plumbing(tmp_path, shar_dir, mode):
    source = _source(shar_dir, mode)
    pack = _pack(tmp_path, source["in_dir"] if mode == "in_dir" else source["fields"])
    cuts = CutSet.from_shar(**source, index_pack=pack, shuffle_shards=True, seed=5)
    assert isinstance(cuts.data, LazyIndexedSharIterator)
    assert cuts.is_indexed
    assert cuts.data._packed is not None and cuts.data.shuffle
    expected = CutSet.from_shar(**source, indexed=True, shuffle_shards=True, seed=5)
    assert [c.id for c in cuts] == [c.id for c in expected]

    # An open IndexPack, the file-descriptor bound and the graph-restore contract.
    cuts = CutSet.from_shar(
        **source, index_pack=IndexPack(pack), index_pack_max_open_files=1
    )
    assert cuts.data._packed.max_open_files == 1
    assert (
        len(list(cuts)) == NUM_CUTS
    )  # every shard is visited with a single open descriptor

    with pytest.raises(ValueError, match="cut_map_fns"):
        CutSet.from_shar(**source, index_pack=pack, cut_map_fns=[None])


def test_supports_configuration(tmp_path, shar_dir):
    pack = _pack(tmp_path, shar_dir)
    assert LazyIndexedSharIterator.supports_configuration(
        in_dir=shar_dir, index_pack=pack
    )
    assert not LazyIndexedSharIterator.supports_configuration(
        in_dir=tmp_path / "other", index_pack=pack
    )
    assert not LazyIndexedSharIterator.supports_configuration(
        in_dir=shar_dir, index_pack=pack, index_path=tmp_path
    )
    assert not LazyIndexedSharIterator.supports_configuration(
        in_dir=shar_dir, index_pack=tmp_path / "missing.idxpack"
    )


def test_cli_shar_pack(tmp_path):
    first = _write_shar(tmp_path / "first", begin_id=0)
    second = _write_shar(tmp_path / "second", begin_id=100)
    pack = tmp_path / "cli.idxpack"
    result = CliRunner().invoke(
        cli, ["index", "shar-pack", str(pack), str(first), str(second)]
    )
    assert result.exit_code == 0, result.output
    assert pack.exists()
    assert len(CutSet.from_shar(in_dir=second, index_pack=pack)) == NUM_CUTS
    # An existing pack is only replaced on request.
    result = CliRunner().invoke(cli, ["index", "shar-pack", str(pack), str(first)])
    assert result.exit_code != 0
    result = CliRunner().invoke(
        cli, ["index", "shar-pack", "--overwrite", str(pack), str(first)]
    )
    assert result.exit_code == 0, result.output
    with pytest.raises(ValueError, match="no Shar source"):
        CutSet.from_shar(in_dir=second, index_pack=pack)


@pytest.mark.parametrize("lazy", [False, True])
def test_fields_added_without_rewriting_cuts_and_long_unicode_ids(tmp_path, lazy):
    """Cuts without placeholders (fields appended later) and long ids stored in extended tar headers."""
    cuts = list(DummyManifest(CutSet, begin_id=0, end_id=5, with_data=True))
    for cut in cuts:
        cut.id = f"nested/{cut.id}-" + "音声" * 70
    with SharWriter(tmp_path, fields={}, shard_size=3, compress_jsonl=False) as writer:
        for cut in cuts:
            writer.write(fastcopy(cut, recording=None, features=None, custom=None))
    with SharWriter(
        tmp_path,
        fields={"recording": "wav", "features": "numpy"},
        shard_size=3,
        compress_jsonl=False,
        include_cuts=False,
        warn_unused_fields=False,
        create_index=True,
    ) as writer:
        for cut in cuts:
            writer.write(cut)
    for path in tmp_path.glob("cuts.*.jsonl"):
        lhotse.indexing.create_jsonl_index(path)
    pack = tmp_path.parent / f"{tmp_path.name}.idxpack"
    write_shar_index_pack(pack, tmp_path)
    expected = list(LazyIndexedSharIterator(in_dir=tmp_path, lazy=lazy))
    actual = list(LazyIndexedSharIterator(in_dir=tmp_path, lazy=lazy, index_pack=pack))
    assert [c.id for c in actual] == [c.id for c in expected] == [c.id for c in cuts]
    for e, a in zip(expected, actual):
        assert a.recording.sources[0].source == e.recording.sources[0].source
        assert a.features.storage_key == e.features.storage_key
        np.testing.assert_array_equal(a.load_audio(), e.load_audio())
