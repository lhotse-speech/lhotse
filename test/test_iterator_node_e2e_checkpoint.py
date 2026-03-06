import importlib
from pathlib import Path
from typing import Callable, Dict

import pytest
import torch.utils.data

from lhotse import CutSet
from lhotse.cut.set import LazyCutMixer
from lhotse.dataset.iterable_dataset import IterableDatasetWrapper
from lhotse.dataset.sampling.base import CutSampler
from lhotse.dataset.sampling.dynamic_bucketing import DynamicBucketingSampler
from lhotse.dataset.webdataset import LazyWebdatasetIterator, export_to_webdataset
from lhotse.hf import LazyHFDatasetIterator
from lhotse.lazy import (
    IteratorNode,
    LazyFilter,
    LazyFlattener,
    LazyIndexedManifestIterator,
    LazyInfiniteApproximateMultiplexer,
    LazyIteratorChain,
    LazyIteratorMultiplexer,
    LazyJsonlIterator,
    LazyManifestIterator,
    LazyMapper,
    LazyRepeater,
    LazyShuffler,
    LazySlicer,
    LazyTxtIterator,
)
from lhotse.shar.readers.indexed import LazyIndexedSharIterator
from lhotse.shar.readers.lazy import LazySharIterator
from lhotse.testing.dummies import DummyManifest
from lhotse.utils import fastcopy, is_module_available

try:
    from torchdata.stateful_dataloader import StatefulDataLoader

    _HAS_TORCHDATA = True
except ImportError:
    _HAS_TORCHDATA = False


class _IdentityDataset(torch.utils.data.Dataset):
    def __getitem__(self, batch):
        return batch


def _even_cut_id(cut) -> bool:
    return int(cut.id.split("-")[-1]) % 2 == 0


def _resample_to_24k(cut):
    return cut.resample(24000)


def _load_indexed(path: Path) -> CutSet:
    return CutSet.from_file(path, indexed=True)


@pytest.fixture()
def iterator_node_sources(tmp_path) -> Dict[str, Path]:
    cuts_a = tmp_path / "cuts_a.jsonl"
    cuts_b = tmp_path / "cuts_b.jsonl"
    noise = tmp_path / "noise.jsonl"
    shar_0 = tmp_path / "cuts.000000.jsonl"
    shar_1 = tmp_path / "cuts.000001.jsonl"

    DummyManifest(CutSet, begin_id=0, end_id=32).to_jsonl(cuts_a)
    DummyManifest(CutSet, begin_id=100, end_id=132).to_jsonl(cuts_b)
    DummyManifest(CutSet, begin_id=1000, end_id=1010).to_jsonl(noise)
    DummyManifest(CutSet, begin_id=2000, end_id=2016).to_jsonl(shar_0)
    DummyManifest(CutSet, begin_id=3000, end_id=3016).to_jsonl(shar_1)

    webdataset_tar = tmp_path / "cuts.tar"
    if is_module_available("webdataset"):
        base_cut = CutSet.from_file("test/fixtures/libri/cuts.json")[0]
        webdataset_cuts = CutSet.from_cuts(
            [fastcopy(base_cut, id=f"{base_cut.id}-wds-{idx}") for idx in range(16)]
        )
        export_to_webdataset(
            webdataset_cuts,
            output_path=str(webdataset_tar),
            load_audio=False,
            load_features=False,
            load_custom=False,
        )

    return {
        "cuts_a": cuts_a,
        "cuts_b": cuts_b,
        "noise": noise,
        "shar_0": shar_0,
        "shar_1": shar_1,
        "webdataset_tar": webdataset_tar,
    }


def _build_lazy_manifest(sources: Dict[str, Path]) -> CutSet:
    return CutSet(LazyManifestIterator(sources["cuts_a"]))


def _build_lazy_indexed_manifest(sources: Dict[str, Path]) -> CutSet:
    return CutSet(LazyIndexedManifestIterator(sources["cuts_a"], shuffle=True, seed=19))


def _build_lazy_iterator_chain(sources: Dict[str, Path]) -> CutSet:
    return CutSet(
        LazyIteratorChain(
            _load_indexed(sources["cuts_a"]).data,
            _load_indexed(sources["cuts_b"]).data,
        )
    )


def _build_lazy_iterator_multiplexer(sources: Dict[str, Path]) -> CutSet:
    return CutSet(
        LazyIteratorMultiplexer(
            _load_indexed(sources["cuts_a"]).data,
            _load_indexed(sources["cuts_b"]).data,
            weights=[0.35, 0.65],
            seed=17,
        )
    )


def _build_lazy_filter(sources: Dict[str, Path]) -> CutSet:
    return CutSet(LazyFilter(_load_indexed(sources["cuts_a"]).data, _even_cut_id))


def _build_lazy_mapper(sources: Dict[str, Path]) -> CutSet:
    return CutSet(LazyMapper(_load_indexed(sources["cuts_a"]).data, _resample_to_24k))


def _build_lazy_repeater(sources: Dict[str, Path]) -> CutSet:
    return CutSet(
        LazyRepeater(_load_indexed(sources["cuts_a"]).data, times=2, preserve_id=False)
    )


def _build_lazy_slicer(sources: Dict[str, Path]) -> CutSet:
    return CutSet(LazySlicer(_load_indexed(sources["cuts_a"]).data, k=0, n=3))


def _build_lazy_cut_mixer(sources: Dict[str, Path]) -> CutSet:
    return CutSet(
        LazyCutMixer(
            cuts=_load_indexed(sources["cuts_a"]),
            mix_in_cuts=_load_indexed(sources["noise"]),
            mix_prob=1.0,
            seed=23,
            preserve_id="left",
        )
    )


def _build_lazy_shar_iterator(sources: Dict[str, Path]) -> CutSet:
    return CutSet(
        LazySharIterator(
            fields={"cuts": [sources["shar_0"], sources["shar_1"]]},
            shuffle_shards=True,
            seed=3,
        )
    )


def _build_lazy_indexed_shar_iterator(sources: Dict[str, Path]) -> CutSet:
    return CutSet(
        LazyIndexedSharIterator(
            fields={"cuts": [sources["shar_0"], sources["shar_1"]]},
            shuffle=True,
            seed=29,
        )
    )


def _build_lazy_iterator_chain_over_indexed_shar(sources: Dict[str, Path]) -> CutSet:
    return CutSet(
        LazyIteratorChain(
            LazyIndexedSharIterator(
                fields={"cuts": [sources["shar_0"]]},
                shuffle=True,
                seed=29,
            ),
            LazyIndexedSharIterator(
                fields={"cuts": [sources["shar_1"]]},
                shuffle=True,
                seed=31,
            ),
        )
    )


def _build_lazy_webdataset_iterator(sources: Dict[str, Path]) -> CutSet:
    pytest.importorskip("webdataset")
    return CutSet(LazyWebdatasetIterator(str(sources["webdataset_tar"])))


def _build_lazy_hf_dataset_iterator(_: Dict[str, Path]) -> CutSet:
    pytest.importorskip("datasets")

    from datasets import Dataset

    wav_bytes = Path("test/fixtures/mono_c0.wav").read_bytes()
    dataset = Dataset.from_dict(
        {
            "audio": [{"bytes": wav_bytes, "path": None}] * 8,
            "sentence": [f"text-{idx}" for idx in range(8)],
            "language": ["en"] * 8,
            "gender": ["m"] * 8,
            "speaker": [f"speaker-{idx}" for idx in range(8)],
        }
    )
    return CutSet(LazyHFDatasetIterator(dataset, gender_key="gender"))


INDEXED_E2E_CASES = [
    pytest.param(
        "LazyIndexedManifestIterator",
        _build_lazy_indexed_manifest,
        id="LazyIndexedManifestIterator",
    ),
    pytest.param(
        "LazyIteratorChain",
        _build_lazy_iterator_chain,
        id="LazyIteratorChain",
    ),
    pytest.param(
        "LazyIteratorMultiplexer",
        _build_lazy_iterator_multiplexer,
        id="LazyIteratorMultiplexer",
    ),
    pytest.param("LazyFilter", _build_lazy_filter, id="LazyFilter"),
    pytest.param(
        "LazyMapper",
        _build_lazy_mapper,
        id="LazyMapper",
    ),
    pytest.param("LazyRepeater", _build_lazy_repeater, id="LazyRepeater"),
    pytest.param("LazySlicer", _build_lazy_slicer, id="LazySlicer"),
    pytest.param(
        "LazyCutMixer",
        _build_lazy_cut_mixer,
        id="LazyCutMixer",
    ),
    pytest.param(
        "LazyIndexedSharIterator",
        _build_lazy_indexed_shar_iterator,
        id="LazyIndexedSharIterator",
    ),
]
INDEXED_E2E_NODE_NAMES = {case.values[0] for case in INDEXED_E2E_CASES}

REPLAY_E2E_CASES = [
    pytest.param(
        "LazyManifestIterator", _build_lazy_manifest, id="LazyManifestIterator"
    ),
    pytest.param("LazySharIterator", _build_lazy_shar_iterator, id="LazySharIterator"),
    pytest.param(
        "LazyWebdatasetIterator",
        _build_lazy_webdataset_iterator,
        id="LazyWebdatasetIterator",
    ),
    pytest.param(
        "LazyHFDatasetIterator",
        _build_lazy_hf_dataset_iterator,
        id="LazyHFDatasetIterator",
    ),
]
REPLAY_E2E_NODE_NAMES = {case.values[0] for case in REPLAY_E2E_CASES}

UNIT_ONLY_NODES = {
    "LazyTxtIterator",
    "LazyJsonlIterator",
}

UNSUPPORTED_EXACT_RESTORE_NODES = {
    "LazyInfiniteApproximateMultiplexer",
    "LazyShuffler",
    "LazyFlattener",
}


def _all_iterator_node_names():
    for module_name in (
        "lhotse.lazy",
        "lhotse.cut.set",
        "lhotse.shar.readers.lazy",
        "lhotse.shar.readers.indexed",
        "lhotse.dataset.webdataset",
        "lhotse.hf",
    ):
        importlib.import_module(module_name)

    queue = list(IteratorNode.__subclasses__())
    seen = set()
    while queue:
        cls = queue.pop()
        if cls in seen:
            continue
        seen.add(cls)
        queue.extend(cls.__subclasses__())
    return {
        cls.__name__
        for cls in seen
        if not cls.__name__.startswith("_") and not cls.__module__.startswith("test.")
    }


def _strip_runtime_fields(value):
    if isinstance(value, dict):
        return {
            k: _strip_runtime_fields(v)
            for k, v in value.items()
            if k not in {"_origin", "dataloading_info"}
        }
    if isinstance(value, list):
        return [_strip_runtime_fields(v) for v in value]
    return value


def _batch_signature(batch):
    return [_strip_runtime_fields(cut.to_dict()) for cut in batch]


def _make_wrapper(cuts: CutSet) -> IterableDatasetWrapper:
    sampler = DynamicBucketingSampler(
        cuts,
        max_cuts=4,
        shuffle=False,
        seed=0,
        num_buckets=2,
    )
    return IterableDatasetWrapper(_IdentityDataset(), sampler)


def _assert_exact_restore_with_wrapper(
    make_cuts: Callable[[Dict[str, Path]], CutSet],
    sources: Dict[str, Path],
    monkeypatch,
    *,
    n_consumed: int = 2,
    expect_o1: bool,
) -> None:
    def make():
        return _make_wrapper(make_cuts(sources))

    full = make()
    all_batches = [_batch_signature(batch) for batch in full]
    assert len(all_batches) > n_consumed

    wrapped_1 = make()
    iterator_1 = iter(wrapped_1)
    first_batches = [_batch_signature(next(iterator_1)) for _ in range(n_consumed)]
    state = wrapped_1.state_dict()

    wrapped_2 = make()
    wrapped_2.load_state_dict(state)

    if expect_o1:
        original_next = CutSampler.__next__

        def _fail_on_replay(self):
            raise RuntimeError("Unexpected O(N) replay during indexed restore.")

        monkeypatch.setattr(CutSampler, "__next__", _fail_on_replay)
        try:
            iter(wrapped_2)
        finally:
            monkeypatch.setattr(CutSampler, "__next__", original_next)

    remaining = [_batch_signature(batch) for batch in wrapped_2]
    assert first_batches + remaining == all_batches


def _assert_exact_restore_with_stateful_dataloader(
    make_cuts: Callable[[Dict[str, Path]], CutSet],
    sources: Dict[str, Path],
    *,
    n_consumed: int = 2,
) -> None:
    if not _HAS_TORCHDATA:
        pytest.skip("torchdata not installed")

    def make():
        return _make_wrapper(make_cuts(sources))

    full = StatefulDataLoader(make(), batch_size=None, num_workers=2)
    all_batches = [_batch_signature(batch) for batch in full]
    assert len(all_batches) > n_consumed

    loader_1 = StatefulDataLoader(make(), batch_size=None, num_workers=2)
    iterator_1 = iter(loader_1)
    first_batches = [_batch_signature(next(iterator_1)) for _ in range(n_consumed)]
    state = loader_1.state_dict()

    loader_2 = StatefulDataLoader(make(), batch_size=None, num_workers=2)
    loader_2.load_state_dict(state)
    remaining = [_batch_signature(batch) for batch in loader_2]

    assert first_batches + remaining == all_batches


def test_iterator_node_checkpoint_matrix_is_complete():
    expected = {
        *UNIT_ONLY_NODES,
        *UNSUPPORTED_EXACT_RESTORE_NODES,
        *INDEXED_E2E_NODE_NAMES,
        *REPLAY_E2E_NODE_NAMES,
    }
    assert _all_iterator_node_names() == expected


def test_iterator_node_chain_preserves_nested_graph_tokens(
    iterator_node_sources, monkeypatch
):
    _assert_exact_restore_with_wrapper(
        _build_lazy_iterator_chain_over_indexed_shar,
        iterator_node_sources,
        monkeypatch,
        expect_o1=True,
    )


@pytest.mark.parametrize(("node_name", "make_cuts"), INDEXED_E2E_CASES)
def test_iterator_node_indexed_exact_restore(
    node_name, make_cuts, iterator_node_sources, monkeypatch
):
    _assert_exact_restore_with_wrapper(
        make_cuts,
        iterator_node_sources,
        monkeypatch,
        expect_o1=True,
    )


@pytest.mark.skipif(not _HAS_TORCHDATA, reason="torchdata not installed")
@pytest.mark.parametrize(("node_name", "make_cuts"), INDEXED_E2E_CASES)
def test_iterator_node_indexed_worker_restore(
    node_name, make_cuts, iterator_node_sources
):
    _assert_exact_restore_with_stateful_dataloader(
        make_cuts,
        iterator_node_sources,
    )


@pytest.mark.parametrize(("node_name", "make_cuts"), REPLAY_E2E_CASES)
def test_iterator_node_replay_exact_restore(
    node_name, make_cuts, iterator_node_sources, monkeypatch
):
    _assert_exact_restore_with_wrapper(
        make_cuts,
        iterator_node_sources,
        monkeypatch,
        expect_o1=False,
    )


@pytest.mark.parametrize("node_name", sorted(UNIT_ONLY_NODES))
def test_iterator_node_unit_only_classes_are_explicit(node_name):
    assert node_name in UNIT_ONLY_NODES


@pytest.mark.parametrize("node_name", sorted(UNSUPPORTED_EXACT_RESTORE_NODES))
def test_iterator_node_unsupported_exact_restore_classes_are_explicit(node_name):
    assert node_name in UNSUPPORTED_EXACT_RESTORE_NODES


def test_noncheckpointable_iterator_nodes_still_report_noncheckpointable(tmp_path):
    txt_path = tmp_path / "lines.txt"
    jsonl_path = tmp_path / "items.jsonl"
    txt_path.write_text("a\nb\n")
    jsonl_path.write_text('{"id": "a"}\n{"id": "b"}\n')

    unit_only = [
        LazyTxtIterator(txt_path),
        LazyJsonlIterator(jsonl_path),
    ]
    unsupported = [
        LazyInfiniteApproximateMultiplexer([1, 2], [3, 4]),
        LazyShuffler([1, 2, 3]),
        LazyFlattener([[1, 2], [3, 4]]),
    ]
    for iterator in [*unit_only, *unsupported]:
        assert isinstance(iterator, IteratorNode)
        assert not iterator.is_checkpointable
