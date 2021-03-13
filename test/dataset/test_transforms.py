import random
from math import isclose

from lhotse import CutSet
from lhotse.dataset.cut_transforms.perturb_speed import PerturbSpeed
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
