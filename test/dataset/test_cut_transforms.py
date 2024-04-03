import random
from math import isclose

import numpy as np
import pytest

from lhotse import CutSet
from lhotse.cut import MixedCut
from lhotse.dataset import (
    CutMix,
    ExtraPadding,
    PerturbSpeed,
    PerturbTempo,
    PerturbVolume,
)
from lhotse.testing.dummies import DummyManifest


@pytest.mark.parametrize("preserve_id", [False, True])
def test_perturb_speed(preserve_id: bool):
    tfnm = PerturbSpeed(
        factors=[0.9, 1.1], p=0.5, randgen=random.Random(42), preserve_id=preserve_id
    )
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    cuts_sp = tfnm(cuts)

    assert all(
        # The duration will not be exactly 0.9 and 1.1 because perturb speed
        # will round to a physically-viable duration based on the sampling_rate
        # (i.e. round to the nearest sample count).
        any(isclose(cut.duration, v, abs_tol=0.0125) for v in [0.9, 1.0, 1.1])
        for cut in cuts_sp
    )

    if preserve_id:
        assert all(cut.id == cut_sp.id for cut, cut_sp in zip(cuts, cuts_sp))
    else:
        # Note: not using all() because PerturbSpeed has p=0.5
        assert any(cut.id != cut_sp.id for cut, cut_sp in zip(cuts, cuts_sp))


@pytest.mark.parametrize("preserve_id", [False, True])
def test_perturb_tempo(preserve_id: bool):
    tfnm = PerturbTempo(
        factors=[0.9, 1.1], p=0.5, randgen=random.Random(42), preserve_id=preserve_id
    )
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    cuts_tp = tfnm(cuts)

    assert all(
        # The duration will not be exactly 0.9 and 1.1 because perturb speed
        # will round to a physically-viable duration based on the sampling_rate
        # (i.e. round to the nearest sample count).
        any(isclose(cut.duration, v, abs_tol=0.0125) for v in [0.9, 1.0, 1.1])
        for cut in cuts_tp
    )

    if preserve_id:
        assert all(cut.id == cut_tp.id for cut, cut_tp in zip(cuts, cuts_tp))
    else:
        # Note: not using all() because PerturbTempo has p=0.5
        assert any(cut.id != cut_tp.id for cut, cut_tp in zip(cuts, cuts_tp))


@pytest.mark.parametrize("preserve_id", [False, True])
def test_perturb_volume(preserve_id: bool):
    tfnm = PerturbVolume(
        scale_low=0.125,
        scale_high=2.0,
        p=0.5,
        randgen=random.Random(42),
        preserve_id=preserve_id,
    )
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    cuts_vp = tfnm(cuts)

    assert all(
        cut.duration == 1.0
        and cut.start == 0.0
        and cut.recording.sampling_rate == 16000
        and cut.recording.num_samples == 16000
        and cut.recording.duration == 1.0
        for cut in cuts_vp
    )

    if preserve_id:
        assert all(cut.id == cut_vp.id for cut, cut_vp in zip(cuts, cuts_vp))
    else:
        # Note: not using all() because PerturbVolume has p=0.5
        assert any(cut.id != cut_vp.id for cut, cut_vp in zip(cuts, cuts_vp))


@pytest.mark.parametrize("preserve_id", [False, True])
def test_cutmix(preserve_id: bool):
    speech_cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    for c in speech_cuts:
        c.duration = 10.0

    noise_cuts = DummyManifest(CutSet, begin_id=100, end_id=102)
    for c in noise_cuts:
        c.duration = 1.5

    tfnm = CutMix(noise_cuts, snr=None, p=1.0, preserve_id=preserve_id)

    tfnm_cuts = tfnm(speech_cuts)
    for c in tfnm_cuts:
        assert isinstance(c, MixedCut)
        assert c.tracks[0].cut.duration == 10.0
        assert sum(t.cut.duration for t in c.tracks[1:]) == 10.0

    if preserve_id:
        assert all(
            cut.id == cut_noisy.id for cut, cut_noisy in zip(speech_cuts, tfnm_cuts)
        )
    else:
        assert all(
            cut.id != cut_noisy.id for cut, cut_noisy in zip(speech_cuts, tfnm_cuts)
        )


def test_cut_mix_is_stateful():
    speech_cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    noise_cuts = DummyManifest(CutSet, begin_id=100, end_id=102)

    # called twice on the same input, expecting different results
    tnfm = CutMix(noise_cuts, snr=None, p=1.0, seed=0, preserve_id=True)
    out1 = tnfm(speech_cuts)
    out2 = tnfm(speech_cuts)
    assert list(out1) != list(out2)


def test_cutmix_random_mix_offset():
    speech_cuts = CutSet.from_json("test/fixtures/ljspeech/cuts.json").resample(16000)
    noise_cuts = CutSet.from_json("test/fixtures/libri/cuts.json")
    normal_tfnm = CutMix(noise_cuts, p=1.0)
    random_tfnm = CutMix(noise_cuts, p=1.0, random_mix_offset=True)
    for a, b in zip(normal_tfnm(speech_cuts), random_tfnm(speech_cuts)):
        assert not np.array_equal(a.load_audio(), b.load_audio())


@pytest.mark.parametrize("randomized", [False, True])
@pytest.mark.parametrize("preserve_id", [False, True])
def test_extra_padding_frames(randomized: bool, preserve_id: bool):
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    transform = ExtraPadding(
        extra_frames=4, randomized=randomized, preserve_id=preserve_id
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

    if preserve_id:
        assert all(cut.id == cut_pad.id for cut, cut_pad in zip(cuts, padded_cuts))
    else:
        # Note: using any(), not all(), since some cuts may be unaffected
        #       as the transform may be randomized.
        assert any(cut.id != cut_pad.id for cut, cut_pad in zip(cuts, padded_cuts))


@pytest.mark.parametrize("randomized", [False, True])
def test_extra_padding_samples(randomized):
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    transform = ExtraPadding(extra_samples=320, randomized=randomized)
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


@pytest.mark.parametrize("randomized", [False, True])
def test_extra_padding_seconds(randomized):
    cuts = DummyManifest(CutSet, begin_id=0, end_id=10)
    transform = ExtraPadding(extra_seconds=0.04, randomized=randomized)
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
