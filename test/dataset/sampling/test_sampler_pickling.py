import pickle
from tempfile import NamedTemporaryFile

import pytest

from lhotse import CutSet
from lhotse.dataset import (
    BucketingSampler,
    CutPairsSampler,
    DynamicBucketingSampler,
    SingleCutSampler,
    ZipSampler,
)
from lhotse.testing.dummies import DummyManifest

CUTS = DummyManifest(CutSet, begin_id=0, end_id=100)
CUTS_MOD = CUTS.modify_ids(lambda cid: cid + "_alt")

SAMPLERS_TO_TEST = [
    SingleCutSampler(CUTS, max_duration=10.0, shuffle=True, drop_last=True),
    CutPairsSampler(CUTS, CUTS, max_source_duration=10.0, shuffle=True, drop_last=True),
    ZipSampler(  # CUTS_SHUF just to randomize the order of the zipped-in cutset
        SingleCutSampler(CUTS, max_duration=10.0, shuffle=True, drop_last=True),
        SingleCutSampler(CUTS_MOD, max_duration=10.0, shuffle=True, drop_last=True),
    ),
    # Differently initialized ZipSampler with the same CUTS (cut pairs)
    ZipSampler(
        CutPairsSampler(
            CUTS, CUTS, max_source_duration=10.0, shuffle=True, drop_last=True
        ),
        CutPairsSampler(
            CUTS_MOD, CUTS_MOD, max_source_duration=10.0, shuffle=True, drop_last=True
        ),
    ),
    # Differently initialized BucketingSampler with the same CUTS
    BucketingSampler(
        CUTS, max_duration=10.0, shuffle=True, drop_last=True, num_buckets=2
    ),
    # Differently initialized BucketingSampler (using CutPairsSampler) with the same CUTS
    BucketingSampler(
        CUTS,
        CUTS,
        max_source_duration=10.0,
        shuffle=True,
        drop_last=True,
        num_buckets=2,
        sampler_type=CutPairsSampler,
    ),
    # pytest.param(
    DynamicBucketingSampler(
        CUTS, max_duration=10.0, shuffle=True, drop_last=True, num_buckets=2
    ),
    DynamicBucketingSampler(
        CUTS, CUTS_MOD, max_duration=10.0, shuffle=True, drop_last=True, num_buckets=2
    ),
    #     marks=pytest.mark.xfail(reason='DynamicBucketingSampler does not support resumption yet.')
    # )
]


@pytest.mark.parametrize("sampler", SAMPLERS_TO_TEST)
def test_sampler_pickling(sampler):
    with NamedTemporaryFile(mode="w+b", suffix='.pkl') as f:
        pickle.dump(sampler, f)
        f.flush()
        f.seek(0)
        restored = pickle.load(f)
