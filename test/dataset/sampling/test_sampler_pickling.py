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
