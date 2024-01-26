from math import isclose

import numpy as np
import pytest

from lhotse.audio import Recording
from lhotse.cut import CutSet, MixedCut, MonoCut, MultiCut
from lhotse.supervision import SupervisionSegment
from lhotse.testing.dummies import DummyManifest, remove_spaces_from_segment_text
from lhotse.utils import nullcontext as does_not_raise

# Note:
# Definitions for `cut1`, `cut2`, `multi_cut1`, `multi_cut2`, and `cut_set` parameters are
# standard Pytest fixtures located in test/cut/conftest.py


@pytest.fixture
def stereo_cut():
    return MultiCut(
        id="multi-cut-1",
        start=0.0,
        duration=1.0,
        channel=[0, 1],
        recording=Recording.from_file(
            "test/fixtures/stereo.wav", recording_id="irrelevant"
        ),
        supervisions=[
            SupervisionSegment(
                id="sup-1", recording_id="irrelevant", start=0.1, duration=0.5
            ),
            SupervisionSegment(
                id="sup-2", recording_id="irrelevant", start=0.7, duration=0.2
            ),
        ],
    )


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


def test_append_multi_cut_with_same_channels(multi_cut1, multi_cut2):
    appended_cut = multi_cut1.append(multi_cut2)

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


def test_append_multi_cut_with_different_channels(multi_cut1, multi_cut3):
    with pytest.raises(AssertionError):
        _ = multi_cut1.append(multi_cut3)


def test_append_mono_cut_with_multi_cut(cut1, multi_cut2):
    appended_cut = cut1.append(multi_cut2)

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


def test_multi_cut_downmix(stereo_cut):
    mono_cut = stereo_cut.to_mono(mono_downmix=True)
    assert isinstance(mono_cut, MonoCut)
    assert mono_cut.num_channels == 1
    assert mono_cut.num_samples == 8000
    assert mono_cut.duration == 1.0
    assert mono_cut.supervisions == [
        SupervisionSegment(
            id="sup-1", recording_id="irrelevant", start=0.1, duration=0.5
        ),
        SupervisionSegment(
            id="sup-2", recording_id="irrelevant", start=0.7, duration=0.2
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


@pytest.fixture
def offseted_mixed_audio_cut() -> MixedCut:
    cut_set = CutSet.from_json(
        "test/fixtures/mix_cut_test/offseted_audio_cut_manifest.json"
    )
    mixed_cut = cut_set["mixed-cut-id"]
    assert isclose(mixed_cut.duration, 16.66)
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


def test_mixed_cut_load_offseted_mixed(offseted_mixed_audio_cut):
    audio = offseted_mixed_audio_cut.load_audio()
    assert audio.shape == (1, 266560)


@pytest.mark.parametrize(
    "mixed, mono_downmix",
    [
        (True, True),
        (True, False),
        pytest.param(
            False,
            True,
            marks=pytest.mark.filterwarnings("ignore::UserWarning"),
        ),
        (False, False),
    ],
)
def test_mixed_cut_with_multi_cut_load_audio_mixed1(mixed, mono_downmix):
    mono_cut1 = Recording.from_file(
        "test/fixtures/mix_cut_test/audio/storage/2412-153948-0000.flac"
    ).to_cut()  # 11.66s
    mono_cut2 = Recording.from_file(
        "test/fixtures/mix_cut_test/audio/storage/2412-153948-0001.flac"
    ).to_cut()  # 10.51s
    rir = Recording.from_file("test/fixtures/rir/real_8ch.wav")
    multi_cut = mono_cut1.reverb_rir(
        rir, rir_channels=[0, 1, 2, 3, 4, 5, 6, 7]
    )  # 11.66s
    mixed_cut = multi_cut.mix(mono_cut2, offset_other_by=5.0)  # 15.51s
    assert mixed_cut.duration == 15.51
    mixed_cut = mixed_cut.pad(duration=20.0)
    assert mixed_cut.duration == 20.0
    audio = mixed_cut.load_audio(mixed=mixed, mono_downmix=mono_downmix)
    if mixed and mono_downmix:
        assert audio.shape == (1, 320000)
    elif mixed and not mono_downmix:
        assert audio.shape == (8, 320000)
    else:
        assert isinstance(audio, list) and len(audio) == 3


@pytest.mark.parametrize(
    "mixed, mono_downmix",
    [
        (True, True),
        (True, False),
        pytest.param(
            False, True, marks=pytest.mark.filterwarnings("ignore::UserWarning")
        ),
        (False, False),
    ],
)
def test_mixed_cut_with_multi_cut_load_audio_mixed2(mixed, mono_downmix):
    mono_cut1 = Recording.from_file(
        "test/fixtures/mix_cut_test/audio/storage/2412-153948-0000.flac"
    ).to_cut()
    mono_cut2 = Recording.from_file(
        "test/fixtures/mix_cut_test/audio/storage/2412-153948-0001.flac"
    ).to_cut()
    rir = Recording.from_file("test/fixtures/rir/real_8ch.wav")
    multi_cut1 = mono_cut1.reverb_rir(rir, rir_channels=[0, 1, 2, 3])
    multi_cut2 = mono_cut2.reverb_rir(rir, rir_channels=[0, 1, 2, 3])
    mixed_cut = multi_cut1.append(multi_cut2)
    assert mixed_cut.duration == 22.17
    mixed_cut = mixed_cut.pad(duration=25.0)
    assert mixed_cut.duration == 25.0
    audio = mixed_cut.load_audio(mixed=mixed, mono_downmix=mono_downmix)
    if mixed and mono_downmix:
        assert audio.shape == (1, 400000)
    elif mixed and not mono_downmix:
        assert audio.shape == (4, 400000)
    else:
        assert isinstance(audio, list) and len(audio) == 3


def test_mixed_cut_with_multi_cut_incompatible():
    mono_cut1 = Recording.from_file(
        "test/fixtures/mix_cut_test/audio/storage/2412-153948-0000.flac"
    ).to_cut()
    mono_cut2 = Recording.from_file(
        "test/fixtures/mix_cut_test/audio/storage/2412-153948-0001.flac"
    ).to_cut()
    rir = Recording.from_file("test/fixtures/rir/real_8ch.wav")
    multi_cut1 = mono_cut1.reverb_rir(rir, rir_channels=[0, 1, 2, 3])
    multi_cut2 = mono_cut2.reverb_rir(rir, rir_channels=[0, 1, 2])
    mixed_cut = multi_cut1.pad(duration=20.0)
    assert mixed_cut.duration == 20.0

    # check 1
    with pytest.raises(AssertionError):
        mixed_cut.append(multi_cut2)

    # check 2
    with pytest.raises(AssertionError):
        multi_cut2.append(mixed_cut)


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


def test_mix_cut_with_other_raises_error(libri_cut):
    libri_cut = libri_cut.drop_features()
    with pytest.raises(ValueError):
        _ = libri_cut.mix(libri_cut.recording)


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


@pytest.mark.parametrize("mix_first", [True, False])
def test_mix_cut_with_transform(libri_cut, mix_first):
    # Create original mixed cut
    padded = libri_cut.pad(duration=20, direction="right")
    # Create transformed mixed cut
    padded = padded.reverb_rir(mix_first=mix_first)
    # Mix another cut
    mixed1 = padded.mix(libri_cut)
    mixed2 = libri_cut.mix(padded)

    assert isinstance(padded, MixedCut)
    assert len(padded.tracks) == 2
    assert isinstance(mixed1, MixedCut)
    assert isinstance(mixed2, MixedCut)
    if mix_first:
        assert len(mixed1.tracks) == 2
        assert len(mixed2.tracks) == 2
    else:
        assert len(mixed1.tracks) == 3
        assert len(mixed2.tracks) == 3


def test_cut_set_mix_snr_is_deterministic():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=2)

    mixed = cuts.mix(cuts, snr=10, mix_prob=1.0, seed=0)
    assert len(mixed) == 2

    c0 = mixed[0]
    assert isinstance(c0, MixedCut)
    assert len(c0.tracks) == 2
    assert c0.tracks[0].snr is None
    assert c0.tracks[1].snr == 10

    c1 = mixed[1]
    assert isinstance(c1, MixedCut)
    assert len(c1.tracks) == 2
    assert c1.tracks[0].snr is None
    assert c1.tracks[1].snr == 10

    # redundant but make it obvious
    assert c0.tracks[1].snr == c1.tracks[1].snr


def test_cut_set_mix_snr_is_randomized():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=2)

    mixed = cuts.mix(cuts, snr=[0, 10], mix_prob=1.0, seed=0)
    assert len(mixed) == 2

    c0 = mixed[0]
    assert isinstance(c0, MixedCut)
    assert len(c0.tracks) == 2
    assert c0.tracks[0].snr is None
    assert 0 <= c0.tracks[1].snr <= 10

    c1 = mixed[1]
    assert isinstance(c1, MixedCut)
    assert len(c1.tracks) == 2
    assert c1.tracks[0].snr is None
    assert 0 <= c1.tracks[1].snr <= 10

    assert c0.tracks[1].snr != c1.tracks[1].snr


def test_cut_set_mix_is_lazy():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=2)

    mixed = cuts.mix(cuts, snr=10, mix_prob=1.0, seed=0)

    assert mixed.is_lazy


def test_cut_set_mix_size_is_not_growing():
    cuts = DummyManifest(CutSet, begin_id=0, end_id=100)
    noise_cuts = DummyManifest(CutSet, begin_id=10, end_id=20)

    mixed_cuts = cuts.mix(
        cuts=noise_cuts,
        duration=None,
        snr=10,
        mix_prob=0.1,
        preserve_id=None,
        seed=42,
        random_mix_offset=True,
    ).to_eager()

    assert len(mixed_cuts) == len(cuts)
