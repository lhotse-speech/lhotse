import random
from math import isclose

import pytest

from lhotse import CutSet
from lhotse.cut import MixedCut
from lhotse.dataset import CutMix, ExtraPadding
from lhotse.dataset import PerturbSpeed, PerturbVolume
from lhotse.testing.dummies import DummyManifest


def test_perturb_speed():
    tfnm = PerturbSpeed(factors=[0.9, 1.1], p=0.5, randgen=random.Random(42))
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    cuts_sp = tfnm(cuts)

    assert all(
        # The duration will not be exactly 0.9 and 1.1 because perturb speed
        # will round to a physically-viable duration based on the sampling_rate
        # (i.e. round to the nearest sample count).
        any(isclose(cut.duration, v, abs_tol=0.0125) for v in [0.9, 1.0, 1.1])
        for cut in cuts_sp
    )


def test_perturb_volume():
    tfnm = PerturbVolume(factors=[0.125, 2.], p=0.5, randgen=random.Random(42))
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    cuts_vp = tfnm(cuts)

    assert all(
        cut.duration == 1. and
        cut.start == 0. and
        cut.recording.sampling_rate == 16000 and
        cut.recording.num_samples == 16000 and
        cut.recording.duration == 1.0 for cut in cuts_vp
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


@pytest.mark.parametrize('randomized', [False, True])
def test_extra_padding_frames(randomized):
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    transform = ExtraPadding(
        extra_frames=4,
        randomized=randomized
    )
    padded_cuts = transform(cuts)

    # Non-randomized test -- check that all cuts are processed
    # in the same way.
    if not randomized:
        for cut, padded in zip(cuts, padded_cuts):
            # first track is for padding
            assert padded.tracks[0].cut.num_frames == 2
            # second track is for padding
            assert padded.tracks[-1].cut.num_frames == 2
            # total num frames is OK
            assert padded.num_frames == cut.num_frames + 4

    # Randomized test -- check that cuts have different properties.
    if randomized:
        nums_frames = [c.num_frames for c in padded_cuts]
        assert len(set(nums_frames)) > 1


@pytest.mark.parametrize('randomized', [False, True])
def test_extra_padding_samples(randomized):
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    transform = ExtraPadding(
        extra_samples=320,
        randomized=randomized
    )
    padded_cuts = transform(cuts)

    # Non-randomized test -- check that all cuts are processed
    # in the same way.
    if not randomized:
        for cut, padded in zip(cuts, padded_cuts):
            # first track is for padding
            assert padded.tracks[0].cut.num_samples == 160
            # second track is for padding
            assert padded.tracks[-1].cut.num_samples == 160
            # total num frames is OK
            assert padded.num_samples == cut.num_samples + 320

    # Randomized test -- check that cuts have different properties.
    if randomized:
        nums_samples = [c.num_samples for c in padded_cuts]
        assert len(set(nums_samples)) > 1


@pytest.mark.parametrize('randomized', [False, True])
def test_extra_padding_seconds(randomized):
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    transform = ExtraPadding(
        extra_seconds=0.04,
        randomized=randomized
    )
    padded_cuts = transform(cuts)

    # Non-randomized test -- check that all cuts are processed
    # in the same way.
    if not randomized:
        for cut, padded in zip(cuts, padded_cuts):
            # first track is for padding
            assert padded.tracks[0].cut.duration == 0.02
            # second track is for padding
            assert padded.tracks[-1].cut.duration == 0.02
            # total num frames is OK
            assert isclose(padded.duration, cut.duration + 0.04)

    # Randomized test -- check that cuts have different properties.
    if randomized:
        durations = [c.duration for c in padded_cuts]
        assert len(set(durations)) > 1
