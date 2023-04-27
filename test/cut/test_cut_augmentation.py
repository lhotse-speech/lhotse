import numpy as np
import pytest
import torch

from lhotse import AudioSource, CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.audio import RecordingSet
from lhotse.cut import PaddingCut
from lhotse.testing.dummies import dummy_cut, dummy_multi_cut
from lhotse.utils import fastcopy, is_module_available


@pytest.fixture
def file_source():
    return AudioSource(type="file", channels=[0], source="test/fixtures/mono_c0.wav")


@pytest.fixture
def recording(file_source):
    return Recording(
        id="rec",
        sources=[file_source],
        sampling_rate=8000,
        num_samples=4000,
        duration=0.5,
    )


@pytest.fixture
def rir():
    return Recording.from_file("test/fixtures/rir/sim_1ch.wav")


@pytest.fixture
def multi_channel_rir():
    return Recording.from_file("test/fixtures/rir/real_8ch.wav")


@pytest.fixture
def libri_recording_orig():
    return Recording.from_file("test/fixtures/libri/libri-1088-134315-0000.wav")


@pytest.fixture
def libri_recording_rvb():
    return Recording.from_file("test/fixtures/libri/libri-1088-134315-0000_rvb.wav")


@pytest.fixture
def cut_with_supervision(recording):
    return MonoCut(
        id="cut",
        start=0.0,
        duration=0.5,
        channel=0,
        supervisions=[
            SupervisionSegment(id="sup", recording_id="rec", start=0.0, duration=0.5)
        ],
        recording=recording,
    )


@pytest.fixture
def libri_cut_with_supervision(libri_recording_orig):
    return MonoCut(
        id="libri_cut_1",
        start=0,
        duration=libri_recording_orig.duration,
        channel=0,
        supervisions=[
            SupervisionSegment(
                id="sup",
                recording_id="rec",
                start=0,
                duration=libri_recording_orig.duration,
            )
        ],
        recording=libri_recording_orig,
    )


def test_cut_perturb_speed11(cut_with_supervision):
    cut_sp = cut_with_supervision.perturb_speed(1.1)
    assert cut_sp.start == 0.0
    assert cut_sp.duration == 0.4545
    assert cut_sp.end == 0.4545
    assert cut_sp.num_samples == 3636

    assert cut_sp.recording.duration == 0.4545
    assert cut_sp.recording.num_samples == 3636

    assert cut_sp.supervisions[0].start == 0.0
    assert cut_sp.supervisions[0].duration == 0.4545
    assert cut_sp.supervisions[0].end == 0.4545

    cut_samples = cut_sp.load_audio()
    assert cut_samples.shape[0] == 1
    assert cut_samples.shape[1] == 3636

    recording_samples = cut_sp.recording.load_audio()
    assert recording_samples.shape[0] == 1
    assert recording_samples.shape[1] == 3636


def test_cut_perturb_speed09(cut_with_supervision):
    cut_sp = cut_with_supervision.perturb_speed(0.9)
    assert cut_sp.start == 0.0
    assert cut_sp.duration == 0.5555
    assert cut_sp.end == 0.5555
    assert cut_sp.num_samples == 4444

    assert cut_sp.recording.duration == 0.5555
    assert cut_sp.recording.num_samples == 4444

    assert cut_sp.supervisions[0].start == 0.0
    assert cut_sp.supervisions[0].duration == 0.5555
    assert cut_sp.supervisions[0].end == 0.5555

    cut_samples = cut_sp.load_audio()
    assert cut_samples.shape[0] == 1
    assert cut_samples.shape[1] == 4444

    recording_samples = cut_sp.recording.load_audio()
    assert recording_samples.shape[0] == 1
    assert recording_samples.shape[1] == 4444


def test_cut_perturb_tempo09(cut_with_supervision):
    cut_tp = cut_with_supervision.perturb_tempo(0.9)
    assert cut_tp.start == 0.0
    assert cut_tp.duration == 0.5555
    assert cut_tp.end == 0.5555
    assert cut_tp.num_samples == 4444

    assert cut_tp.recording.duration == 0.5555
    assert cut_tp.recording.num_samples == 4444

    assert cut_tp.supervisions[0].start == 0.0
    assert cut_tp.supervisions[0].duration == 0.5555
    assert cut_tp.supervisions[0].end == 0.5555

    cut_samples = cut_tp.load_audio()
    assert cut_samples.shape[0] == 1
    assert cut_samples.shape[1] == 4444

    recording_samples = cut_tp.recording.load_audio()
    assert recording_samples.shape[0] == 1
    assert recording_samples.shape[1] == 4444


def test_cut_perturb_tempo11(cut_with_supervision):
    cut_tp = cut_with_supervision.perturb_tempo(1.1)
    assert cut_tp.start == 0.0
    assert cut_tp.duration == 0.4545
    assert cut_tp.end == 0.4545
    assert cut_tp.num_samples == 3636

    assert cut_tp.recording.duration == 0.4545
    assert cut_tp.recording.num_samples == 3636

    assert cut_tp.supervisions[0].start == 0.0
    assert cut_tp.supervisions[0].duration == 0.4545
    assert cut_tp.supervisions[0].end == 0.4545

    cut_samples = cut_tp.load_audio()
    assert cut_samples.shape[0] == 1
    assert cut_samples.shape[1] == 3636

    recording_samples = cut_tp.recording.load_audio()
    assert recording_samples.shape[0] == 1
    assert recording_samples.shape[1] == 3636


def test_cut_set_perturb_speed_doesnt_duplicate_transforms(cut_with_supervision):
    cuts = CutSet.from_cuts(
        [cut_with_supervision, cut_with_supervision.with_id("other-id")]
    )
    cuts_sp = cuts.perturb_speed(1.1)
    for cut in cuts_sp:
        # This prevents a bug regression where multiple cuts referencing the same recording would
        # attach transforms to the same manifest
        assert len(cut.recording.transforms) == 1


def test_cut_set_perturb_volume_doesnt_duplicate_transforms(cut_with_supervision):
    cuts = CutSet.from_cuts(
        [cut_with_supervision, cut_with_supervision.with_id("other-id")]
    )
    cuts_vp = cuts.perturb_volume(2.0)
    for cut in cuts_vp:
        # This prevents a bug regression where multiple cuts referencing the same recording would
        # attach transforms to the same manifest
        assert len(cut.recording.transforms) == 1


def test_cut_set_reverb_rir_doesnt_duplicate_transforms(cut_with_supervision, rir):
    rirs = RecordingSet.from_recordings([rir]).resample(8000)
    cuts = CutSet.from_cuts(
        [cut_with_supervision, cut_with_supervision.with_id("other-id")]
    )
    cuts_vp = cuts.reverb_rir(rir_recordings=rirs)
    for cut in cuts_vp:
        # This prevents a bug regression where multiple cuts referencing the same recording would
        # attach transforms to the same manifest
        assert len(cut.recording.transforms) == 1


def test_cut_set_resample_doesnt_duplicate_transforms(cut_with_supervision):
    cuts = CutSet.from_cuts(
        [cut_with_supervision, cut_with_supervision.with_id("other-id")]
    )
    cuts_res = cuts.resample(44100)
    for cut in cuts_res:
        # This prevents a bug regression where multiple cuts referencing the same recording would
        # attach transforms to the same manifest
        assert len(cut.recording.transforms) == 1


@pytest.fixture
def cut_with_supervision_start01(recording):
    return MonoCut(
        id="cut_start01",
        start=0.1,
        duration=0.4,
        channel=0,
        supervisions=[
            SupervisionSegment(id="sup", recording_id="rec", start=0.1, duration=0.3)
        ],
        recording=recording,
    )


def test_cut_start01_perturb_speed11(cut_with_supervision_start01):
    cut_sp = cut_with_supervision_start01.perturb_speed(1.1)
    assert cut_sp.start == 0.090875
    assert cut_sp.duration == 0.363625
    assert cut_sp.end == 0.4545
    assert cut_sp.num_samples == 2909

    assert cut_sp.recording.duration == 0.4545
    assert cut_sp.recording.num_samples == 3636

    assert cut_sp.supervisions[0].start == 0.090875
    assert cut_sp.supervisions[0].duration == 0.27275
    assert cut_sp.supervisions[0].end == 0.363625

    cut_samples = cut_sp.load_audio()
    assert cut_samples.shape[0] == 1
    assert cut_samples.shape[1] == 2909

    recording_samples = cut_sp.recording.load_audio()
    assert recording_samples.shape[0] == 1
    assert recording_samples.shape[1] == 3636


def test_cut_start01_perturb_speed09(cut_with_supervision_start01):
    cut_sp = cut_with_supervision_start01.perturb_speed(0.9)
    assert cut_sp.start == 0.111125
    assert cut_sp.duration == 0.4445
    assert cut_sp.end == 0.555625
    assert cut_sp.num_samples == 3556

    assert cut_sp.recording.duration == 0.5555
    assert cut_sp.recording.num_samples == 4444

    assert cut_sp.supervisions[0].start == 0.111125
    assert cut_sp.supervisions[0].duration == 0.333375
    assert cut_sp.supervisions[0].end == 0.4445

    cut_samples = cut_sp.load_audio()
    assert cut_samples.shape[0] == 1
    assert cut_samples.shape[1] == 3556

    recording_samples = cut_sp.recording.load_audio()
    assert recording_samples.shape[0] == 1
    assert recording_samples.shape[1] == 4444


def test_mixed_cut_start01_perturb_speed(cut_with_supervision_start01):
    mixed_sp = cut_with_supervision_start01.append(
        cut_with_supervision_start01
    ).perturb_speed(1.1)
    assert mixed_sp.start == 0  # MixedCut always starts at 0
    assert mixed_sp.duration == 0.363625 * 2
    assert mixed_sp.end == 0.363625 * 2
    assert mixed_sp.num_samples == 2909 * 2

    assert mixed_sp.supervisions[0].start == 0.090875
    assert mixed_sp.supervisions[0].duration == 0.27275
    assert mixed_sp.supervisions[0].end == 0.363625
    assert (
        mixed_sp.supervisions[1].start == 0.4545
    )  # round(0.363625 + 0.090875, ndigits=8)
    assert mixed_sp.supervisions[1].duration == 0.27275
    assert mixed_sp.supervisions[1].end == 0.363625 * 2

    cut_samples = mixed_sp.load_audio()
    assert cut_samples.shape[0] == 1
    assert cut_samples.shape[1] == 2909 * 2


def test_mixed_cut_start01_perturb_volume(cut_with_supervision_start01):
    mixed_vp = cut_with_supervision_start01.append(
        cut_with_supervision_start01
    ).perturb_volume(0.125)
    assert mixed_vp.start == 0  # MixedCut always starts at 0
    assert mixed_vp.duration == cut_with_supervision_start01.duration * 2
    assert mixed_vp.end == cut_with_supervision_start01.duration * 2
    assert mixed_vp.num_samples == cut_with_supervision_start01.num_samples * 2

    assert (
        mixed_vp.supervisions[0].start
        == cut_with_supervision_start01.supervisions[0].start
    )
    assert (
        mixed_vp.supervisions[0].duration
        == cut_with_supervision_start01.supervisions[0].duration
    )
    assert (
        mixed_vp.supervisions[0].end == cut_with_supervision_start01.supervisions[0].end
    )
    assert mixed_vp.supervisions[1].start == (
        cut_with_supervision_start01.duration
        + cut_with_supervision_start01.supervisions[0].start
    )
    assert (
        mixed_vp.supervisions[1].duration
        == cut_with_supervision_start01.supervisions[0].duration
    )
    assert mixed_vp.supervisions[1].end == (
        cut_with_supervision_start01.duration
        + cut_with_supervision_start01.supervisions[0].end
    )

    cut_samples = mixed_vp.load_audio()
    cut_with_supervision_start01_samples = cut_with_supervision_start01.load_audio()
    assert (
        cut_samples.shape[0] == cut_with_supervision_start01_samples.shape[0]
        and cut_samples.shape[1] == cut_with_supervision_start01_samples.shape[1] * 2
    )
    np.testing.assert_array_almost_equal(
        cut_samples,
        np.hstack(
            (cut_with_supervision_start01_samples, cut_with_supervision_start01_samples)
        )
        * 0.125,
    )


def test_mixed_cut_start01_reverb_rir(cut_with_supervision_start01, rir):
    rir = rir.resample(8000)
    mixed_rvb = cut_with_supervision_start01.append(
        cut_with_supervision_start01
    ).reverb_rir(rir_recording=rir, mix_first=False)
    assert mixed_rvb.start == 0  # MixedCut always starts at 0
    assert mixed_rvb.duration == cut_with_supervision_start01.duration * 2
    assert mixed_rvb.end == cut_with_supervision_start01.duration * 2
    assert mixed_rvb.num_samples == cut_with_supervision_start01.num_samples * 2

    assert (
        mixed_rvb.supervisions[0].start
        == cut_with_supervision_start01.supervisions[0].start
    )
    assert (
        mixed_rvb.supervisions[0].duration
        == cut_with_supervision_start01.supervisions[0].duration
    )
    assert (
        mixed_rvb.supervisions[0].end
        == cut_with_supervision_start01.supervisions[0].end
    )
    assert mixed_rvb.supervisions[1].start == (
        cut_with_supervision_start01.duration
        + cut_with_supervision_start01.supervisions[0].start
    )
    assert (
        mixed_rvb.supervisions[1].duration
        == cut_with_supervision_start01.supervisions[0].duration
    )
    assert mixed_rvb.supervisions[1].end == (
        cut_with_supervision_start01.duration
        + cut_with_supervision_start01.supervisions[0].end
    )

    cut_samples = mixed_rvb.load_audio()
    cut_with_supervision_start01_samples = cut_with_supervision_start01.reverb_rir(
        rir_recording=rir
    ).load_audio()
    assert (
        cut_samples.shape[0] == cut_with_supervision_start01_samples.shape[0]
        and cut_samples.shape[1] == cut_with_supervision_start01_samples.shape[1] * 2
    )
    np.testing.assert_array_almost_equal(
        cut_samples,
        np.hstack(
            (cut_with_supervision_start01_samples, cut_with_supervision_start01_samples)
        ),
    )


def test_mixed_cut_start01_reverb_rir_mix_first(cut_with_supervision_start01, rir):
    mixed_rvb = cut_with_supervision_start01.pad(duration=0.5).reverb_rir(
        rir_recording=rir, mix_first=True
    )
    assert mixed_rvb.start == 0  # MixedCut always starts at 0
    assert mixed_rvb.duration == 0.5
    assert mixed_rvb.end == 0.5
    assert mixed_rvb.num_samples == 4000

    # Check that the padding part should not be all zeros afte
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_almost_equal,
        mixed_rvb.load_audio()[:, 3200:],
        np.zeros((1, 800)),
    )


def test_mixed_cut_start01_reverb_rir_with_fast_random(
    cut_with_supervision_start01, rir
):
    mixed_rvb = cut_with_supervision_start01.append(
        cut_with_supervision_start01
    ).reverb_rir()
    assert mixed_rvb.start == 0  # MixedCut always starts at 0
    assert mixed_rvb.duration == cut_with_supervision_start01.duration * 2
    assert mixed_rvb.end == cut_with_supervision_start01.duration * 2
    assert mixed_rvb.num_samples == cut_with_supervision_start01.num_samples * 2

    assert (
        mixed_rvb.supervisions[0].start
        == cut_with_supervision_start01.supervisions[0].start
    )
    assert (
        mixed_rvb.supervisions[0].duration
        == cut_with_supervision_start01.supervisions[0].duration
    )
    assert (
        mixed_rvb.supervisions[0].end
        == cut_with_supervision_start01.supervisions[0].end
    )
    assert mixed_rvb.supervisions[1].start == (
        cut_with_supervision_start01.duration
        + cut_with_supervision_start01.supervisions[0].start
    )
    assert (
        mixed_rvb.supervisions[1].duration
        == cut_with_supervision_start01.supervisions[0].duration
    )
    assert mixed_rvb.supervisions[1].end == (
        cut_with_supervision_start01.duration
        + cut_with_supervision_start01.supervisions[0].end
    )


@pytest.mark.parametrize(
    "rir_channels, expected_num_tracks",
    [([0], 2), ([0, 1], 2), ([0, 1, 2], None)],
)
def test_mixed_cut_start01_reverb_rir_multi_channel(
    cut_with_supervision_start01, multi_channel_rir, rir_channels, expected_num_tracks
):
    mixed_cut = cut_with_supervision_start01.append(cut_with_supervision_start01)
    multi_channel_rir = multi_channel_rir.resample(8000)
    if expected_num_tracks is not None:
        mixed_rvb = mixed_cut.reverb_rir(multi_channel_rir, rir_channels=rir_channels)
        assert len(mixed_rvb.tracks) == expected_num_tracks
    else:
        with pytest.raises(AssertionError):
            mixed_cut.reverb_rir(multi_channel_rir, rir_channels=rir_channels)


@pytest.mark.skipif(
    not is_module_available("pyloudnorm"),
    reason="This test requires pyloudnorm to be installed.",
)
@pytest.mark.parametrize(
    "target, mix_first", [(-15.0, True), (-20.0, True), (-25.0, False)]
)
def test_mixed_cut_normalize_loudness(cut_with_supervision_start01, target, mix_first):
    mixed_cut = cut_with_supervision_start01.append(cut_with_supervision_start01)
    mixed_cut_ln = mixed_cut.normalize_loudness(target, mix_first=mix_first)

    import pyloudnorm as pyln

    # check if loudness is correct
    meter = pyln.Meter(mixed_cut_ln.sampling_rate)  # create BS.1770 meter
    loudness = meter.integrated_loudness(mixed_cut_ln.load_audio().T)
    if mix_first:
        assert loudness == pytest.approx(target, abs=0.5)


@pytest.mark.skipif(
    not is_module_available("nara_wpe"),
    reason="This test requires nara_wpe to be installed.",
)
@pytest.mark.parametrize("affix_id", [True, False])
def test_mono_cut_dereverb_wpe(affix_id):
    cut = dummy_cut(0, with_data=True)
    cut_wpe = cut.dereverb_wpe(affix_id=affix_id)
    if affix_id:
        assert cut_wpe.id == f"{cut.id}_wpe"
    else:
        assert cut_wpe.id == cut.id
    samples = cut.load_audio()
    samples_wpe = cut_wpe.load_audio()
    assert samples_wpe.shape[0] == cut_wpe.num_channels
    assert samples_wpe.shape[1] == cut_wpe.num_samples
    assert (samples != samples_wpe).any()


@pytest.mark.skipif(
    not is_module_available("nara_wpe"),
    reason="This test requires nara_wpe to be installed.",
)
@pytest.mark.parametrize("affix_id", [True, False])
def test_multi_cut_dereverb_wpe(affix_id):
    cut = dummy_multi_cut(0, with_data=True)
    cut_wpe = cut.dereverb_wpe(affix_id=affix_id)
    if affix_id:
        assert cut_wpe.id == f"{cut.id}_wpe"
    else:
        assert cut_wpe.id == cut.id
    samples = cut.load_audio()
    samples_wpe = cut_wpe.load_audio()
    assert samples_wpe.shape[0] == cut_wpe.num_channels
    assert samples_wpe.shape[1] == cut_wpe.num_samples
    assert (samples != samples_wpe).any()


def test_padding_cut_perturb_speed():
    cut = PaddingCut(
        id="cut",
        duration=5.75,
        sampling_rate=16000,
        feat_value=1e-10,
        num_samples=92000,
    )
    cut_sp = cut.perturb_speed(1.1)
    assert cut_sp.num_samples == 83636
    assert cut_sp.duration == 5.22725


def test_padding_cut_perturb_volume():
    cut = PaddingCut(
        id="cut",
        duration=5.75,
        sampling_rate=16000,
        feat_value=1e-10,
        num_samples=92000,
    )
    cut_vp = cut.perturb_volume(0.125)
    assert cut_vp.num_samples == cut.num_samples
    assert cut_vp.duration == cut.duration
    np.testing.assert_array_almost_equal(cut_vp.load_audio(), cut.load_audio())


def test_padding_cut_reverb_rir(rir):
    cut = PaddingCut(
        id="cut",
        duration=5.75,
        sampling_rate=16000,
        feat_value=1e-10,
        num_samples=92000,
    )
    cut_rvb = cut.reverb_rir(rir_recording=rir)
    assert cut_rvb.num_samples == cut.num_samples
    assert cut_rvb.duration == cut.duration
    np.testing.assert_array_almost_equal(cut_rvb.load_audio(), cut.load_audio())


def test_cut_set_perturb_speed(cut_with_supervision, cut_with_supervision_start01):
    cut_set = CutSet.from_cuts([cut_with_supervision, cut_with_supervision_start01])
    cs_sp = cut_set.perturb_speed(1.1)
    for cut_sp, cut in zip(cs_sp, cut_set):
        samples = cut_sp.load_audio()
        assert samples.shape[1] == cut_sp.num_samples
        assert samples.shape[1] < cut.num_samples


@pytest.fixture()
def cut_set(cut_with_supervision, cut_with_supervision_start01):
    return CutSet.from_cuts([cut_with_supervision, cut_with_supervision_start01])


@pytest.fixture()
def libri_cut_set(libri_cut_with_supervision):
    cut1 = libri_cut_with_supervision
    cut2 = fastcopy(cut1, id="libri_cut_2")
    return CutSet.from_cuts([cut1, cut2])


@pytest.mark.parametrize("cut_id", ["cut", "cut_start01"])
def test_resample_cut(cut_set, cut_id):
    original = cut_set[cut_id]
    resampled = original.resample(16000)
    assert original.sampling_rate == 8000
    assert resampled.sampling_rate == 16000
    assert resampled.num_samples == 2 * original.num_samples
    samples = resampled.load_audio()
    assert samples.shape[1] == resampled.num_samples


@pytest.mark.parametrize("cut_id", ["cut", "cut_start01"])
@pytest.mark.parametrize("scale", [0.125, 2.0])
def test_cut_perturb_volume(cut_set, cut_id, scale):

    cut = cut_set[cut_id]
    cut_vp = cut.perturb_volume(scale)
    assert cut_vp.start == cut.start
    assert cut_vp.duration == cut.duration
    assert cut_vp.end == cut.end
    assert cut_vp.num_samples == cut.num_samples

    assert cut_vp.recording.duration == cut.recording.duration
    assert cut_vp.recording.num_samples == cut.recording.num_samples

    assert cut_vp.supervisions[0].start == cut.supervisions[0].start
    assert cut_vp.supervisions[0].duration == cut.supervisions[0].duration
    assert cut_vp.supervisions[0].end == cut.supervisions[0].end

    assert cut_vp.load_audio().shape == cut.load_audio().shape
    assert cut_vp.recording.load_audio().shape == cut.recording.load_audio().shape

    np.testing.assert_array_almost_equal(cut_vp.load_audio(), cut.load_audio() * scale)
    np.testing.assert_array_almost_equal(
        cut_vp.recording.load_audio(), cut.recording.load_audio() * scale
    )


@pytest.mark.skipif(
    not is_module_available("pyloudnorm"),
    reason="This test requires pyloudnorm to be installed.",
)
@pytest.mark.parametrize("target", [-15.0, -20.0, -25.0])
def test_cut_normalize_loudness(libri_cut_set, target):
    cut_set_ln = libri_cut_set.normalize_loudness(target)

    import pyloudnorm as pyln

    # check if loudness is correct
    for c in cut_set_ln:
        meter = pyln.Meter(c.sampling_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(c.load_audio().T)
        assert loudness == pytest.approx(target, abs=0.5)


def test_cut_reverb_rir(libri_cut_with_supervision, libri_recording_rvb, rir):

    cut = libri_cut_with_supervision
    cut_rvb = cut.reverb_rir(rir)
    assert cut_rvb.start == cut.start
    assert cut_rvb.duration == cut.duration
    assert cut_rvb.end == cut.end
    assert cut_rvb.num_samples == cut.num_samples

    assert cut_rvb.recording.duration == cut.recording.duration
    assert cut_rvb.recording.num_samples == cut.recording.num_samples

    assert cut_rvb.supervisions[0].start == cut.supervisions[0].start
    assert cut_rvb.supervisions[0].duration == cut.supervisions[0].duration
    assert cut_rvb.supervisions[0].end == cut.supervisions[0].end

    assert cut_rvb.load_audio().shape == cut.load_audio().shape
    assert cut_rvb.recording.load_audio().shape == cut.recording.load_audio().shape

    rvb_audio_from_fixture = libri_recording_rvb.load_audio()

    np.testing.assert_array_almost_equal(cut_rvb.load_audio(), rvb_audio_from_fixture)


def test_cut_reverb_rir_assert_sampling_rate(libri_cut_with_supervision, rir):
    cut = libri_cut_with_supervision
    rir_new = rir.resample(8000)
    with pytest.raises(AssertionError):
        cut = cut.reverb_rir(rir_new)
        _ = cut.load_audio()


def test_cut_reverb_fast_rir(libri_cut_with_supervision):
    cut = libri_cut_with_supervision
    cut_rvb = cut.reverb_rir(rir_recording=None)
    assert cut_rvb.start == cut.start
    assert cut_rvb.duration == cut.duration
    assert cut_rvb.end == cut.end
    assert cut_rvb.num_samples == cut.num_samples

    assert cut_rvb.recording.duration == cut.recording.duration
    assert cut_rvb.recording.num_samples == cut.recording.num_samples

    assert cut_rvb.supervisions[0].start == cut.supervisions[0].start
    assert cut_rvb.supervisions[0].duration == cut.supervisions[0].duration
    assert cut_rvb.supervisions[0].end == cut.supervisions[0].end

    assert cut_rvb.load_audio().shape == cut.load_audio().shape
    assert cut_rvb.recording.load_audio().shape == cut.recording.load_audio().shape


@pytest.mark.parametrize(
    "rir_channels, expected_type, expected_num_tracks",
    [
        ([0], "MonoCut", 1),
        ([1], "MonoCut", 1),
        ([0, 1], "MultiCut", 2),
    ],
)
def test_cut_reverb_multi_channel_rir(
    libri_cut_with_supervision,
    multi_channel_rir,
    rir_channels,
    expected_type,
    expected_num_tracks,
):

    cut = libri_cut_with_supervision
    cut_rvb = cut.reverb_rir(multi_channel_rir, rir_channels=rir_channels)
    assert cut_rvb.to_dict()["type"] == expected_type

    if expected_type == "MixedCut":
        assert len(cut_rvb.tracks) == expected_num_tracks

        for track in cut_rvb.tracks:
            assert track.cut.start == cut.start
            assert track.cut.duration == cut.duration
            assert track.cut.end == cut.end
            assert track.cut.num_samples == cut.num_samples

        assert np.vstack(cut_rvb.load_audio(mixed=False)).shape == (
            expected_num_tracks,
            cut.num_samples,
        )
    else:
        assert cut_rvb.load_audio().shape == (expected_num_tracks, cut.num_samples)


def test_padding_cut_resample():
    original = PaddingCut(
        id="cut",
        duration=5.75,
        sampling_rate=16000,
        feat_value=1e-10,
        num_samples=92000,
    )
    resampled = original.resample(8000)
    assert resampled.sampling_rate == 8000
    assert resampled.num_samples == original.num_samples / 2
    samples = resampled.load_audio()
    assert samples.shape[1] == resampled.num_samples


def test_mixed_cut_resample(cut_with_supervision_start01):
    original = cut_with_supervision_start01.append(cut_with_supervision_start01)
    resampled = original.resample(16000)
    assert original.sampling_rate == 8000
    assert resampled.sampling_rate == 16000
    assert resampled.num_samples == 2 * original.num_samples
    samples = resampled.load_audio()
    assert samples.shape[1] == resampled.num_samples


@pytest.mark.parametrize("affix_id", [True, False])
def test_cut_set_resample(cut_set, affix_id):
    resampled_cs = cut_set.resample(16000, affix_id=affix_id)
    for original, resampled in zip(cut_set, resampled_cs):
        if affix_id:
            assert original.id != resampled.id
            assert resampled.id.endswith("_rs16000")
        else:
            assert original.id == resampled.id
        assert original.sampling_rate == 8000
        assert resampled.sampling_rate == 16000
        assert resampled.num_samples == 2 * original.num_samples
        samples = resampled.load_audio()
        assert samples.shape[1] == resampled.num_samples


@pytest.mark.parametrize("scale", [0.125, 2.0])
@pytest.mark.parametrize("affix_id", [True, False])
def test_cut_set_perturb_volume(cut_set, affix_id, scale):
    perturbed_vp_cs = cut_set.perturb_volume(scale, affix_id=affix_id)
    for original, perturbed_vp in zip(cut_set, perturbed_vp_cs):
        if affix_id:
            assert original.id != perturbed_vp.id
            assert perturbed_vp.id.endswith(f"_vp{scale}")
        else:
            assert original.id == perturbed_vp.id
        assert original.sampling_rate == perturbed_vp.sampling_rate
        assert original.num_samples == perturbed_vp.num_samples
        assert original.load_audio().shape == perturbed_vp.load_audio().shape
        np.testing.assert_array_almost_equal(
            perturbed_vp.load_audio(), original.load_audio() * scale
        )


@pytest.mark.parametrize("affix_id", [True, False])
def test_cut_set_reverb_rir(libri_cut_set, rir, affix_id):
    rirs = RecordingSet.from_recordings([rir])
    perturbed_rvb_cs = libri_cut_set.reverb_rir(rirs, affix_id=affix_id)
    for original, perturbed_rvb in zip(libri_cut_set, perturbed_rvb_cs):
        if affix_id:
            assert original.id != perturbed_rvb.id
            assert perturbed_rvb.id.endswith(f"_rvb")
        else:
            assert original.id == perturbed_rvb.id
        assert original.sampling_rate == perturbed_rvb.sampling_rate
        assert original.num_samples == perturbed_rvb.num_samples
        assert original.load_audio().shape == perturbed_rvb.load_audio().shape
