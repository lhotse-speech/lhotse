"""
Index-pack reading of Shar sets holding many field types and layouts, compared payload by payload
against the sidecar-based reader.
"""

import itertools
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from lhotse import CutSet
from lhotse.array import Array, TemporalArray
from lhotse.audio import Recording
from lhotse.cut import MultiCut
from lhotse.shar import SharWriter, write_shar_index_pack
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


def _dummy_cuts(n, begin_id=0, with_label=True):
    cuts = list(
        DummyManifest(CutSet, begin_id=begin_id, end_id=begin_id + n, with_data=True)
    )
    if with_label:
        for cut in cuts:
            cut.label = {"text": f"label-{cut.id}", "n": len(cut.id)}
    return cuts


def _write(
    path: Path, cuts, fields, shard_size=4, compress_jsonl=False, **kwargs
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("warn_unused_fields", False)
    with SharWriter(
        path,
        fields=fields,
        shard_size=shard_size,
        compress_jsonl=compress_jsonl,
        create_index=True,
        **kwargs,
    ) as writer:
        for cut in cuts:
            writer.write(cut)
    return path


def _mapping(shar_dir: Path, fields=None) -> dict:
    names = None if fields is None else ["cuts", *fields]
    out = {}
    for p in sorted(shar_dir.iterdir()):
        if p.name.endswith((".idx", ".gzidx", ".idxpack")):
            continue
        name = p.name.split(".")[0]
        if names is None or name in names:
            out.setdefault(name, []).append(str(p))
    return out


def _source(shar_dir: Path, mode: str, fields=None) -> dict:
    return (
        {"in_dir": shar_dir}
        if mode == "in_dir"
        else {"fields": _mapping(shar_dir, fields)}
    )


def _pack(tmp_path: Path, source: dict, name="pack.idxpack", **kwargs) -> Path:
    path = tmp_path / name
    write_shar_index_pack(path, [source.get("in_dir", source.get("fields"))], **kwargs)
    return path


def _assert_same_payloads(expected, actual):
    expected, actual = list(expected), list(actual)
    assert [c.id for c in actual] == [c.id for c in expected]
    for e, a in zip(expected, actual):
        assert type(a) is type(e)
        assert str(a.shard_origin) == str(e.shard_origin)
        assert a.shar_epoch == e.shar_epoch
        assert a.to_dict() == e.to_dict()
        if e.has_recording:
            _assert_same_load(a.load_audio, e.load_audio)
        if e.has_features:
            _assert_same_load(a.load_features, e.load_features)
        for name, value in (e.custom or {}).items():
            if isinstance(value, (Array, TemporalArray, Recording)):
                _assert_same_load(
                    lambda: a.load_custom(name), lambda: e.load_custom(name)
                )
            else:
                assert getattr(a, name) == value


def _assert_same_load(actual_fn, expected_fn):
    """Payloads are equal; a field that is not loaded (e.g. outside a field subset) fails identically."""
    try:
        expected = expected_fn()
    except Exception as ex:
        with pytest.raises(type(ex)):
            actual_fn()
        return
    np.testing.assert_array_equal(actual_fn(), expected)


@pytest.mark.parametrize("mode", ["in_dir", "fields"])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "fields",
    [
        FIELDS,
        {
            **FIELDS,
            "recording": "flac",
            "custom_recording": "flac",
            "features": "lilcom",
            "custom_features": "lilcom",
        },
    ],
    ids=["wav-numpy", "flac-lilcom"],
)
def test_field_types_match_sidecar_reader(tmp_path, fields, mode, shuffle, lazy):
    if "lilcom" in fields.values():
        pytest.importorskip("lilcom")
    shar_dir = _write(tmp_path / "shar", _dummy_cuts(11), fields)
    source = _source(shar_dir, mode)
    pack = _pack(tmp_path, source)
    kwargs = dict(shuffle=shuffle, seed=3, lazy=lazy)
    expected = LazyIndexedSharIterator(**source, **kwargs)
    actual = LazyIndexedSharIterator(**source, **kwargs, index_pack=pack)
    for _ in range(2):  # two epochs
        _assert_same_payloads(expected, actual)


@pytest.mark.parametrize("lazy", [False, True])
def test_shards_of_very_different_sizes_and_single_shard(tmp_path, lazy):
    sizes = [1, 9, 2, 5]
    parts = []
    begin = 0
    for i, size in enumerate(sizes):
        # shard_size=None: one unnumbered shard per file group
        parts.append(
            _write(
                tmp_path / f"part{i}", _dummy_cuts(size, begin), FIELDS, shard_size=None
            )
        )
        begin += size
    # Assemble a source whose shards have uneven lengths out of the single-shard parts.
    mapping = {}
    for part in parts:
        for name, paths in _mapping(part).items():
            mapping.setdefault(name, []).extend(paths)
    assert all(len(v) == len(sizes) for v in mapping.values())
    pack = tmp_path / "uneven.idxpack"
    write_shar_index_pack(pack, [mapping, parts[1]])
    for shuffle in (False, True):
        expected = LazyIndexedSharIterator(
            fields=mapping, shuffle=shuffle, seed=1, lazy=lazy
        )
        actual = LazyIndexedSharIterator(
            fields=mapping, shuffle=shuffle, seed=1, lazy=lazy, index_pack=pack
        )
        assert actual.num_shards == 4 and len(actual) == sum(sizes)
        _assert_same_payloads(expected, actual)
        # random access across all shard boundaries
        for idx in range(len(expected)):
            assert actual[idx].id == expected[idx].id
    # A single-shard directory, read in directory mode, with a shard that holds 9 cuts.
    expected = LazyIndexedSharIterator(in_dir=parts[1], lazy=lazy)
    actual = LazyIndexedSharIterator(in_dir=parts[1], lazy=lazy, index_pack=pack)
    assert actual.num_shards == 1 and len(actual) == 9
    _assert_same_payloads(expected, actual)


@pytest.mark.parametrize("lazy", [False, True])
def test_multichannel_recordings(tmp_path, lazy):
    cuts = []
    for cut in _dummy_cuts(6):
        audio = np.stack([cut.load_audio()[0], cut.load_audio()[0] * 0.5])
        rec = Recording.from_bytes(_wav_bytes(audio, cut.sampling_rate), cut.id)
        cuts.append(
            MultiCut(
                cut.id,
                0.0,
                rec.duration,
                channel=[0, 1],
                recording=rec,
                custom={"custom_embedding": cut.custom["custom_embedding"]},
            )
        )
    shar_dir = _write(
        tmp_path / "mc",
        cuts,
        {"recording": "flac", "custom_embedding": "numpy"},
        shard_size=4,
    )
    for source in (_source(shar_dir, "in_dir"), _source(shar_dir, "fields")):
        pack = _pack(
            tmp_path, source, name=f"mc-{len(source)}-{next(iter(source))}.idxpack"
        )
        expected = LazyIndexedSharIterator(**source, lazy=lazy)
        actual = LazyIndexedSharIterator(**source, lazy=lazy, index_pack=pack)
        out = list(actual)
        assert out[0].num_channels == 2 and out[0].load_audio().shape[0] == 2
        _assert_same_payloads(expected, out)


@pytest.mark.parametrize("mode", ["in_dir", "fields"])
@pytest.mark.parametrize("lazy", [False, True])
def test_fields_appended_later_without_cuts(tmp_path, mode, lazy):
    cuts = _dummy_cuts(10)
    shar_dir = tmp_path / "shar"
    # First write: cuts + recording only.
    stripped = [fastcopy(c, features=None, custom={"label": c.label}) for c in cuts]
    _write(shar_dir, stripped, {"recording": "wav"}, shard_size=4)
    # Appended later without rewriting cuts: arrays, a Recording attribute and JSONL.
    _write(
        shar_dir,
        cuts,
        {
            "features": "numpy",
            "custom_embedding": "numpy",
            "custom_indexes": "numpy",
            "custom_recording": "flac",
            "label": "jsonl",
        },
        shard_size=4,
        include_cuts=False,
    )
    source = _source(shar_dir, mode)
    pack = _pack(tmp_path, source)
    expected = LazyIndexedSharIterator(**source, lazy=lazy, shuffle=True, seed=2)
    actual = LazyIndexedSharIterator(
        **source, lazy=lazy, shuffle=True, seed=2, index_pack=pack
    )
    out = list(actual)
    assert (
        out[0].features.storage_type != ""
        and out[0].load_custom("custom_recording").shape[0] == 1
    )
    _assert_same_payloads(expected, out)


@pytest.mark.parametrize("lazy", [False, True])
def test_field_subsets_via_mapping(tmp_path, lazy):
    shar_dir = _write(tmp_path / "shar", _dummy_cuts(9), FIELDS)
    subsets = [
        ["recording"],
        ["features", "custom_recording"],
        ["label", "custom_indexes", "custom_embedding"],
    ]
    mappings = [_mapping(shar_dir, subset) for subset in subsets]
    pack = tmp_path / "subsets.idxpack"
    write_shar_index_pack(pack, mappings)
    for mapping in mappings:
        expected = LazyIndexedSharIterator(fields=mapping, lazy=lazy)
        actual = LazyIndexedSharIterator(fields=mapping, lazy=lazy, index_pack=pack)
        assert actual.fields == expected.fields == set(mapping) - {"cuts"}
        _assert_same_payloads(expected, actual)


@pytest.mark.parametrize("mode", ["in_dir", "fields"])
@pytest.mark.parametrize("lazy", [False, True])
def test_gzip_cuts_with_all_field_types(tmp_path, mode, lazy):
    pytest.importorskip("indexed_gzip")
    shar_dir = _write(tmp_path / "gz", _dummy_cuts(10), FIELDS, compress_jsonl=True)
    assert list(shar_dir.glob("cuts.*.jsonl.gz"))
    source = _source(shar_dir, mode)
    pack = _pack(tmp_path, source)
    expected = list(LazyIndexedSharIterator(**source, lazy=lazy, shuffle=True, seed=4))
    for sidecar in itertools.chain(shar_dir.glob("*.idx"), shar_dir.glob("*.gzidx")):
        sidecar.unlink()
    actual = LazyIndexedSharIterator(
        **source, lazy=lazy, shuffle=True, seed=4, index_pack=pack
    )
    _assert_same_payloads(expected, actual)


def test_long_unicode_ids_with_all_field_types(tmp_path):
    cuts = _dummy_cuts(7)
    for cut in cuts:
        cut.id = f"dir/{cut.id}-" + "音声" * 80  # > 100 bytes: tar extended headers
    shar_dir = _write(tmp_path / "uni", cuts, FIELDS, shard_size=3)
    pack = _pack(tmp_path, {"in_dir": shar_dir})
    for lazy in (False, True):
        expected = LazyIndexedSharIterator(in_dir=shar_dir, lazy=lazy, shuffle=True)
        actual = LazyIndexedSharIterator(
            in_dir=shar_dir, lazy=lazy, shuffle=True, index_pack=pack
        )
        _assert_same_payloads(expected, actual)


def _wav_bytes(audio, sampling_rate):
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio).T, sampling_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ----------------------------------------------------------------------------------------------------------------------
# Pack sources
# ----------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def two_dirs(tmp_path):
    first = _write(tmp_path / "first", _dummy_cuts(9, 0), FIELDS, shard_size=4)
    second = _write(tmp_path / "second", _dummy_cuts(6, 100), FIELDS, shard_size=5)
    return first, second


def test_two_directories_and_a_mapping_in_one_pack(tmp_path, two_dirs):
    first, second = two_dirs
    subset = _mapping(first, ["recording", "label"])
    pack = tmp_path / "multi.idxpack"
    write_shar_index_pack(pack, [first, second, subset])
    for source in ({"in_dir": first}, {"in_dir": second}, {"fields": subset}):
        _assert_same_payloads(
            LazyIndexedSharIterator(**source),
            LazyIndexedSharIterator(**source, index_pack=pack),
        )
    # The pack holds the files of `first` once, although two sources address them.
    solo = tmp_path / "solo.idxpack"
    write_shar_index_pack(solo, [first, second])
    assert (
        pack.stat().st_size
        < solo.stat().st_size + 2 * 4096 + 64 * len(subset["cuts"]) * 8
    )


def test_directory_and_mapping_over_the_same_files(tmp_path, two_dirs):
    first, _ = two_dirs
    mapping = _mapping(first)
    pack = tmp_path / "both.idxpack"
    write_shar_index_pack(pack, [first, mapping])
    by_dir = LazyIndexedSharIterator(
        in_dir=first, index_pack=pack, shuffle=True, seed=1
    )
    by_map = LazyIndexedSharIterator(
        fields=mapping, index_pack=pack, shuffle=True, seed=1
    )
    expected = list(LazyIndexedSharIterator(in_dir=first, shuffle=True, seed=1))
    assert [c.id for c in by_dir] == [c.id for c in by_map] == [c.id for c in expected]
    _assert_same_payloads(
        LazyIndexedSharIterator(fields=mapping, shuffle=True, seed=1),
        LazyIndexedSharIterator(fields=mapping, index_pack=pack, shuffle=True, seed=1),
    )


def test_overlapping_files_between_sources(tmp_path, two_dirs):
    first, second = two_dirs
    m1, m2 = _mapping(first), _mapping(second)
    # Source "a" = first's shards 0,1 followed by second's shard 0; source "b" = first's shard 1,2 (overlap on shard 1).
    a = {k: m1[k][:2] + m2[k][:1] for k in m1}
    b = {k: m1[k][1:] for k in m1}
    pack = tmp_path / "overlap.idxpack"
    write_shar_index_pack(pack, [a, b])
    for source in (a, b):
        _assert_same_payloads(
            LazyIndexedSharIterator(fields=source, shuffle=True),
            LazyIndexedSharIterator(fields=source, shuffle=True, index_pack=pack),
        )
    # The same shard listed twice in one source is read twice.
    twice = {k: [v[0], v[0]] for k, v in m1.items()}
    write_shar_index_pack(pack, [twice], overwrite=True)
    actual = list(LazyIndexedSharIterator(fields=twice, index_pack=pack))
    assert [c.id for c in actual] == [
        c.id for c in LazyIndexedSharIterator(fields=twice)
    ]
    assert len(actual) == 8


def test_stale_cuts_sidecar_is_rejected(tmp_path, two_dirs):
    import time

    first, _ = two_dirs
    cuts = sorted(first.glob("cuts.*.jsonl"))[0]
    future = time.time() + 100
    os.utime(cuts, (future, future))
    with pytest.raises(ValueError, match="newer than"):
        write_shar_index_pack(tmp_path / "p.idxpack", first)
    assert not (tmp_path / "p.idxpack").exists()


def test_cuts_sidecar_of_another_shard_is_rejected(tmp_path, two_dirs):
    """A sidecar copied over from a shard with a different number of records."""
    first, _ = two_dirs
    cuts = sorted(first.glob("cuts.*.jsonl"))
    shutil.copy(
        str(cuts[0]) + ".idx", str(cuts[2]) + ".idx"
    )  # shard 0 has 4 cuts, shard 2 has 1
    with pytest.raises(ValueError, match="records"):
        write_shar_index_pack(tmp_path / "p.idxpack", first)


def test_missing_cuts_sidecar_is_reported(tmp_path, two_dirs):
    first, _ = two_dirs
    Path(str(sorted(first.glob("cuts.*.jsonl"))[1]) + ".idx").unlink()
    with pytest.raises(FileNotFoundError, match="lhotse index shar"):
        write_shar_index_pack(tmp_path / "p.idxpack", first)


def test_tar_appended_after_indexing_is_rejected(tmp_path, two_dirs):
    """A real second archive member after the indexed ones is a non-zero tail, not padding."""
    import tarfile

    first, _ = two_dirs
    tar = sorted(first.glob("custom_embedding.*.tar"))[0]
    with tarfile.open(
        tar, "a"
    ) as f:  # tarfile overwrites the end-of-archive blocks, so the sidecar is stale
        info = tarfile.TarInfo("extra.bin")
        info.size = 3
        import io

        f.addfile(info, io.BytesIO(b"abc"))
    with pytest.raises(ValueError, match="non-zero bytes|newer than"):
        write_shar_index_pack(tmp_path / "p.idxpack", first)


def test_failed_packing_keeps_an_existing_pack(tmp_path, two_dirs):
    first, second = two_dirs
    pack = tmp_path / "p.idxpack"
    write_shar_index_pack(pack, first)
    before = pack.read_bytes()
    Path(str(sorted(second.glob("recording.*.tar"))[0]) + ".idx").unlink()
    with pytest.raises(FileNotFoundError):
        write_shar_index_pack(pack, [first, second], overwrite=True)
    assert pack.read_bytes() == before
    assert not [
        p for p in tmp_path.iterdir() if p.suffix not in {".idxpack"} and p.is_file()
    ]


def test_cli_relative_paths_indexes_root_and_errors(tmp_path, two_dirs, monkeypatch):
    from click.testing import CliRunner

    from lhotse.bin.modes.cli_base import cli

    first, second = two_dirs
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["index", "shar-pack", "rel.idxpack", "first", "second/"]
    )
    assert result.exit_code == 0, result.output
    # The reader must repeat the spelling it was packed with.
    cuts = CutSet.from_shar(in_dir=Path("first"), index_pack="rel.idxpack")
    assert len(cuts) == 9
    with pytest.raises(ValueError, match="no Shar source"):
        CutSet.from_shar(in_dir=first, index_pack="rel.idxpack")
    assert len(CutSet.from_shar(in_dir="second", index_pack="rel.idxpack")) == 6

    # Sidecars in a separate tree.
    root = tmp_path / "idx"
    for sidecar in list(first.glob("*.idx")):
        target = root / sidecar.relative_to(sidecar.anchor)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(sidecar, target)
    result = runner.invoke(
        cli,
        ["index", "shar-pack", "--indexes-root", str(root), "root.idxpack", str(first)],
    )
    assert result.exit_code == 0, result.output
    assert len(list(CutSet.from_shar(in_dir=first, index_pack="root.idxpack"))) == 9

    # Without the root the sidecars are missing; a nonexistent directory and a duplicate are refused.
    result = runner.invoke(cli, ["index", "shar-pack", "bad.idxpack", str(first)])
    assert result.exit_code != 0 and isinstance(result.exception, FileNotFoundError)
    result = runner.invoke(cli, ["index", "shar-pack", "bad.idxpack", "nonexistent"])
    assert result.exit_code != 0
    result = runner.invoke(
        cli, ["index", "shar-pack", "bad.idxpack", "second", "second"]
    )
    assert result.exit_code != 0 and "more than once" in str(result.exception)
    assert not Path("bad.idxpack").exists()
