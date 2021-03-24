import random
from math import isclose

from lhotse import CutSet
from lhotse.cut import MixedCut
from lhotse.dataset import CutMix
from lhotse.dataset import PerturbSpeed
from lhotse.testing.dummies import DummyManifest


def test_perturb_speed():
    tfnm = PerturbSpeed(factors=[0.9, 1.1], p=0.5, randgen=random.Random(42))
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    cuts_sp = tfnm(cuts)
    print(set(c.duration for c in cuts_sp))
    assert all(
        # The duration will not be exactly 0.9 and 1.1 because perturb speed
        # will round to a physically-viable duration based on the sampling_rate
        # (i.e. round to the nearest sample count).
        any(isclose(cut.duration, v, abs_tol=0.0125) for v in [0.9, 1.0, 1.1])
        for cut in cuts_sp
    )


def test_cutmix():
    speech_cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for c in speech_cuts:
        c.duration = 10.0

    noise_cuts = DummyManifest(CutSet, begin_id=100, end_id=102)
    for c in noise_cuts:
        c.duration = 1.5

    tfnm = CutMix(noise_cuts, snr=None, prob=1.0)

    tfnm_cuts = tfnm(speech_cuts)
    for c in tfnm_cuts:
        assert isinstance(c, MixedCut)
        assert c.tracks[0].cut.duration == 10.0
        assert sum(t.cut.duration for t in c.tracks[1:]) == 10.0
