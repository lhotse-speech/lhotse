import pickle
from tempfile import NamedTemporaryFile

import pytest

from lhotse import CutSet
from lhotse.dataset import (
    BucketingSampler,
    CutPairsSampler,
    DynamicBucketingSampler,
    RoundRobinSampler,
    SimpleCutSampler,
    ZipSampler,
)
from lhotse.dataset.sampling.dynamic import DynamicCutSampler
from lhotse.testing.dummies import DummyManifest
from lhotse.testing.fixtures import with_dill_enabled
from lhotse.utils import is_module_available

CUTS = DummyManifest(CutSet, begin_id=0, end_id=100)
CUTS_MOD = CUTS.modify_ids(lambda cid: cid + "_alt")

SAMPLERS_TO_TEST = [
    SimpleCutSampler(CUTS, max_duration=10.0, shuffle=True, drop_last=True),
    CutPairsSampler(CUTS, CUTS, max_source_duration=10.0, shuffle=True, drop_last=True),
    RoundRobinSampler(
        SimpleCutSampler(CUTS, max_duration=10.0, shuffle=True, drop_last=True),
        SimpleCutSampler(CUTS_MOD, max_duration=10.0, shuffle=True, drop_last=True),
    ),
    ZipSampler(
        SimpleCutSampler(CUTS, max_duration=10.0, shuffle=True, drop_last=True),
        SimpleCutSampler(CUTS_MOD, max_duration=10.0, shuffle=True, drop_last=True),
    ),
    ZipSampler(
        CutPairsSampler(
            CUTS, CUTS, max_source_duration=10.0, shuffle=True, drop_last=True
        ),
        CutPairsSampler(
            CUTS_MOD, CUTS_MOD, max_source_duration=10.0, shuffle=True, drop_last=True
        ),
    ),
    BucketingSampler(
        CUTS, max_duration=10.0, shuffle=True, drop_last=True, num_buckets=2
    ),
    BucketingSampler(
        CUTS,
        CUTS,
        max_source_duration=10.0,
        shuffle=True,
        drop_last=True,
        num_buckets=2,
        sampler_type=CutPairsSampler,
    ),
    DynamicBucketingSampler(
        CUTS, max_duration=10.0, shuffle=True, drop_last=True, num_buckets=2
    ),
    DynamicBucketingSampler(
        CUTS, CUTS_MOD, max_duration=10.0, shuffle=True, drop_last=True, num_buckets=2
    ),
    DynamicCutSampler(CUTS, max_duration=10.0, shuffle=True, drop_last=True),
    DynamicCutSampler(CUTS, CUTS, max_duration=10.0, shuffle=True, drop_last=True),
]


@pytest.mark.parametrize("sampler", SAMPLERS_TO_TEST)
def test_sampler_pickling(sampler):
    with NamedTemporaryFile(mode="w+b", suffix=".pkl") as f:
        pickle.dump(sampler, f)
        f.flush()
        f.seek(0)
        restored = pickle.load(f)

    assert sampler.state_dict() == restored.state_dict()


def create_samplers_to_test_filter():
    return [
        SimpleCutSampler(CUTS, max_duration=10.0, shuffle=True),
        RoundRobinSampler(
            SimpleCutSampler(CUTS, max_duration=10.0, shuffle=True),
            SimpleCutSampler(CUTS_MOD, max_duration=10.0, shuffle=True),
        ),
        pytest.param(
            ZipSampler(
                SimpleCutSampler(CUTS, max_duration=10.0, shuffle=True),
                SimpleCutSampler(CUTS_MOD, max_duration=10.0, shuffle=True),
            ),
            marks=pytest.mark.skip(
                reason="ZipSampler exits early if any of the samplers is depleted (here the second sampler is)."
            ),
        ),
        BucketingSampler(CUTS, max_duration=10.0, shuffle=True, num_buckets=2),
        DynamicBucketingSampler(CUTS, max_duration=10.0, shuffle=True, num_buckets=2),
        DynamicCutSampler(CUTS, max_duration=10.0, shuffle=True),
    ]


def dummy_filter_global(cut) -> bool:
    return cut.id == "dummy-mono-cut-0000"


@pytest.mark.parametrize("sampler", create_samplers_to_test_filter())
def test_sampler_pickling_with_filter(sampler):
    sampler.filter(dummy_filter_global)

    with NamedTemporaryFile(mode="w+b", suffix=".pkl") as f:
        pickle.dump(sampler, f)
        f.flush()
        f.seek(0)
        restored = pickle.load(f)

    assert sampler.state_dict() == restored.state_dict()

    batches_original = [b for b in sampler]
    assert len(batches_original) == 1
    assert len(batches_original[0]) == 1
    assert batches_original[0][0].id == "dummy-mono-cut-0000"

    batches_restored = [b for b in restored]
    assert len(batches_restored) == 1
    assert len(batches_restored[0]) == 1
    assert batches_restored[0][0].id == "dummy-mono-cut-0000"


@pytest.mark.xfail(
    not is_module_available("dill"),
    reason="This test will fail when 'dill' module is not installed as it won't be able to pickle a closure.",
    raises=AttributeError,
)
@pytest.mark.parametrize("sampler", create_samplers_to_test_filter())
def test_sampler_pickling_with_filter_local_closure(with_dill_enabled, sampler):

    selected_id = "dummy-mono-cut-0000"

    def dummy_filter_local(cut) -> bool:
        return cut.id == selected_id

    sampler.filter(dummy_filter_local)

    with NamedTemporaryFile(mode="w+b", suffix=".pkl") as f:
        pickle.dump(sampler, f)
        f.flush()
        f.seek(0)
        restored = pickle.load(f)

    assert sampler.state_dict() == restored.state_dict()

    batches_original = [b for b in sampler]
    assert len(batches_original) == 1
    assert len(batches_original[0]) == 1
    assert batches_original[0][0].id == "dummy-mono-cut-0000"

    batches_restored = [b for b in restored]
    assert len(batches_restored) == 1
    assert len(batches_restored[0]) == 1
    assert batches_restored[0][0].id == "dummy-mono-cut-0000"
