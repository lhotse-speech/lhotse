from math import isclose

import numpy as np
import pytest

from lhotse.cut import CutSet, MixedCut, MonoCut
from lhotse.supervision import SupervisionSegment
from lhotse.testing.dummies import remove_spaces_from_segment_text
from lhotse.utils import nullcontext as does_not_raise

# Note:
# Definitions for `cut1`, `cut2` and `cut_set` parameters are standard Pytest fixtures located in test/cut/conftest.py


def test_append_cut_duration_and_supervisions(cut1, cut2):
    appended_cut = cut1.append(cut2)

    assert isinstance(appended_cut, MixedCut)
    assert appended_cut.duration == 20.0
    assert appended_cut.supervisions == [
        SupervisionSegment(
            id="sup-1", recording_id="irrelevant", start=0.5, duration=6.0
        ),
        SupervisionSegment(
            id="sup-2", recording_id="irrelevant", start=7.0, duration=2.0
        ),
        SupervisionSegment(
            id="sup-3", recording_id="irrelevant", start=13.0, duration=2.5
        ),
    ]


@pytest.mark.parametrize(
    ["offset", "allow_padding", "expected_duration", "exception_expectation"],
    [
        (0, False, 10.0, does_not_raise()),
        (1, False, 11.0, does_not_raise()),
        (5, False, 15.0, does_not_raise()),
        (10, False, 20.0, does_not_raise()),
        (100, False, "irrelevant", pytest.raises(AssertionError)),
        (100, True, 110.0, does_not_raise()),
    ],
)
def test_overlay_cut_duration_and_supervisions(
    offset, allow_padding, expected_duration, exception_expectation, cut1, cut2
):
    with exception_expectation:
        mixed_cut = cut1.mix(cut2, offset_other_by=offset, allow_padding=allow_padding)

        assert isinstance(mixed_cut, MixedCut)
        assert mixed_cut.duration == expected_duration
        assert mixed_cut.supervisions == [
            SupervisionSegment(
                id="sup-1", recording_id="irrelevant", start=0.5, duration=6.0
            ),
            SupervisionSegment(
                id="sup-2", recording_id="irrelevant", start=7.0, duration=2.0
            ),
            SupervisionSegment(
                id="sup-3", recording_id="irrelevant", start=3.0 + offset, duration=2.5
            ),
        ]


@pytest.fixture
def mixed_feature_cut() -> MixedCut:
    cut_set = CutSet.from_json("test/fixtures/mix_cut_test/overlayed_cut_manifest.json")
    mixed_cut = cut_set["mixed-cut-id"]
    assert mixed_cut.num_frames == 1360
    assert isclose(mixed_cut.duration, 13.595)
    return mixed_cut


def test_mixed_cut_load_features_mixed(mixed_feature_cut):
    feats = mixed_feature_cut.load_features()
    assert feats.shape[0] == 1360


def test_mixed_cut_load_features_unmixed(mixed_feature_cut):
    feats = mixed_feature_cut.load_features(mixed=False)
    assert feats.shape[0] == 2
    assert feats.shape[1] == 1360


def test_mixed_cut_map_supervisions(mixed_feature_cut):
    for s in mixed_feature_cut.map_supervisions(
        remove_spaces_from_segment_text
    ).supervisions:
        if s.text is not None:
            assert " " not in s.text


@pytest.fixture
def mixed_audio_cut() -> MixedCut:
    cut_set = CutSet.from_json(
        "test/fixtures/mix_cut_test/overlayed_audio_cut_manifest.json"
    )
    mixed_cut = cut_set["mixed-cut-id"]
    assert isclose(mixed_cut.duration, 14.4)
    return mixed_cut


def test_mixed_cut_load_audio_mixed(mixed_audio_cut):
    audio = mixed_audio_cut.load_audio()
    assert audio.shape == (1, 230400)


def test_mixed_cut_load_audio_unmixed(mixed_audio_cut):
    audio = mixed_audio_cut.load_audio(mixed=False)
    assert isinstance(audio, list)
    assert len(audio) == 2
    assert audio[0].shape == (1, 230400)
    assert audio[1].shape == (1, 230400)


@pytest.fixture
def libri_cut_set():
    return CutSet.from_json("test/fixtures/libri/cuts.json")


@pytest.fixture
def libri_cut(libri_cut_set) -> MonoCut:
    return libri_cut_set["e3e70682-c209-4cac-629f-6fbed82c07cd"]


def E(x):
    if x.shape[0] == 1:
        # audio
        return np.sum(x**2)
    # fbank
    return np.sum(np.exp(x))


def test_mix_cut_snr(libri_cut):
    mixed = libri_cut.pad(duration=20).mix(libri_cut, offset_other_by=10)
    mixed_snr = libri_cut.pad(duration=20).mix(libri_cut, offset_other_by=10, snr=10)

    assert len(mixed.tracks) == 3
    assert len(mixed_snr.tracks) == 3

    audio = mixed.load_audio()
    audio_snr = mixed_snr.load_audio()
    feats = mixed.load_features()
    feats_snr = mixed_snr.load_features()

    for item in (audio, audio_snr, feats, feats_snr):
        assert E(item) > 0

    # Cuts mixed without SNR specified should have a higher energy in feature and audio domains.
    assert E(audio) > E(audio_snr)
    assert E(feats) > E(feats_snr)


def test_mix_cut_snr_truncate_snr_reference(libri_cut):
    mixed = libri_cut.pad(duration=20).mix(libri_cut, offset_other_by=10)
    mixed_snr = libri_cut.pad(duration=20).mix(libri_cut, offset_other_by=10, snr=10)

    # truncate enough to remove the first cut
    mixed = mixed.truncate(offset=18)
    mixed_snr = mixed_snr.truncate(offset=18)

    assert len(mixed.tracks) == 2
    assert len(mixed_snr.tracks) == 2

    audio = mixed.load_audio()
    audio_snr = mixed_snr.load_audio()
    feats = mixed.load_features()
    feats_snr = mixed_snr.load_features()

    for item in (audio, audio_snr, feats, feats_snr):
        assert E(item) > 0

    # Both cuts with have identical energies, as the SNR reference was removed in `mixed_snr`,
    # and the only remaining non-padding cut is the one that was mixed in.
    assert E(audio) == E(audio_snr)
    assert E(feats) == E(feats_snr)


def test_mix_cut_snr_pad_both(libri_cut):
    # Pad from both sides, then mix in some "noise" at the beginning.
    # The SNR should refer to the original cut, and not to the mixed in noise.
    padded = libri_cut.pad(duration=20, direction="both")
    mixed = padded.mix(libri_cut)
    mixed_snr = padded.mix(libri_cut, snr=10)

    assert isinstance(padded, MixedCut)
    assert len(padded.tracks) == 3
    assert len(mixed.tracks) == 4
    assert len(mixed_snr.tracks) == 4

    audio = padded.load_audio()
    audio_nosnr = mixed.load_audio()
    audio_snr = mixed_snr.load_audio()
    feats = padded.load_features()
    feats_nosnr = mixed.load_features()
    feats_snr = mixed_snr.load_features()

    for item in (audio, audio_nosnr, audio_snr, feats, feats_nosnr, feats_snr):
        assert E(item) > 0

    # Cuts mixed without SNR specified should have a higher energy in feature and audio domains.
    # Note: if any of those are equal, it means some operation had no effect
    #       (a bug this test is preventing against).
    assert E(audio_snr) > E(audio)
    assert E(audio_nosnr) > E(audio)
    assert E(audio_nosnr) > E(audio_snr)
    assert E(feats_snr) > E(feats)
    assert E(feats_nosnr) > E(feats)
    assert E(feats_nosnr) > E(feats_snr)
