import pytest

from lhotse.cut import MixedCut
from lhotse.testing.fixtures import random_cut_set


@pytest.fixture
def speech_cuts():
    return random_cut_set(n_cuts=200)


@pytest.fixture
def noise_cuts():
    return random_cut_set(n_cuts=30)


def test_cut_set_mixing_default(speech_cuts, noise_cuts):
    mixed_cuts = speech_cuts.mix(noise_cuts)
    for orig_cut, mix_cut in zip(speech_cuts, mixed_cuts):
        assert orig_cut.duration == mix_cut.duration
        if isinstance(mix_cut, MixedCut):
            assert all(t.snr == 20 for t in mix_cut.tracks[1:])


@pytest.mark.parametrize('duration', [1.0, 5.1, 17.5, 30.0])
def test_cut_set_mixing_with_duration(speech_cuts, noise_cuts, duration):
    mixed_cuts = speech_cuts.mix(noise_cuts, duration=duration)
    for orig_cut, mix_cut in zip(speech_cuts, mixed_cuts):
        assert mix_cut.duration == duration


@pytest.mark.parametrize('snr', [17.5, None])
def test_cut_set_mixing_with_snr_value(speech_cuts, noise_cuts, snr):
    mixed_cuts = speech_cuts.mix(noise_cuts, snr=snr)
    for orig_cut, mix_cut in zip(speech_cuts, mixed_cuts):
        if isinstance(mix_cut, MixedCut):
            assert all(t.snr == snr for t in mix_cut.tracks[1:])


def test_cut_set_mixing_with_snr_range(speech_cuts, noise_cuts):
    mixed_cuts = speech_cuts.mix(noise_cuts, snr=[-5, 25])
    for orig_cut, mix_cut in zip(speech_cuts, mixed_cuts):
        if isinstance(mix_cut, MixedCut):
            assert all(-5 <= t.snr <= 25 for t in mix_cut.tracks[1:])


def test_cut_set_mixing_with_prob(speech_cuts, noise_cuts):
    mixed_cuts = speech_cuts.mix(noise_cuts, mix_prob=0.5, duration=1000)
    was_mixed = [c.duration == 1000 for c in mixed_cuts]
    assert any(was_mixed) and not all(was_mixed)
