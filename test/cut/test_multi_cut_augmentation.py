import numpy as np
import pytest

from lhotse import MultiCut, Recording, SupervisionSegment


@pytest.fixture
def recording():
    return Recording.from_file("test/fixtures/libri/libri-1088-134315-0000_8ch.wav")


@pytest.fixture
def mono_rir():
    return Recording.from_file("test/fixtures/rir/sim_1ch.wav")


@pytest.fixture
def multi_channel_rir():
    return Recording.from_file("test/fixtures/rir/real_8ch.wav")


@pytest.fixture
def cut_with_supervision(recording, cut_channels=None, sup_channels=None):
    if cut_channels is None:
        cut_channels = [0, 1, 2, 3, 4, 5, 6, 7]
    if sup_channels is None:
        sup_channels = [0, 1, 2, 3, 4, 5, 6, 7]
    return MultiCut(
        id="cut",
        start=0.0,
        duration=1.0,
        channel=cut_channels,
        supervisions=[
            SupervisionSegment(
                id="sup",
                recording_id="rec",
                start=0.0,
                duration=1.0,
                channel=sup_channels,
            )
        ],
        recording=recording,
    )


def test_cut_perturb_speed11(cut_with_supervision):
    cut_sp = cut_with_supervision.perturb_speed(1.1)
    assert cut_sp.start == 0.0
    assert cut_sp.duration == 0.9090625
    assert cut_sp.end == 0.9090625
    assert cut_sp.num_samples == 14545

    assert cut_sp.recording.duration == 14.5818125
    assert cut_sp.recording.num_samples == 233309

    assert cut_sp.supervisions[0].start == 0.0
    assert cut_sp.supervisions[0].duration == 0.9090625
    assert cut_sp.supervisions[0].end == 0.9090625

    cut_samples = cut_sp.load_audio()
    assert cut_samples.shape[0] == 8
    assert cut_samples.shape[1] == 14545

    recording_samples = cut_sp.recording.load_audio()
    assert recording_samples.shape[0] == 8
    assert recording_samples.shape[1] == 233309


def test_cut_perturb_speed09(cut_with_supervision):
    cut_sp = cut_with_supervision.perturb_speed(0.9)
    assert cut_sp.start == 0.0
    assert cut_sp.duration == 1.111125
    assert cut_sp.end == 1.111125
    assert cut_sp.num_samples == 17778

    assert cut_sp.recording.duration == 17.82225
    assert cut_sp.recording.num_samples == 285156

    assert cut_sp.supervisions[0].start == 0.0
    assert cut_sp.supervisions[0].duration == 1.111125
    assert cut_sp.supervisions[0].end == 1.111125

    cut_samples = cut_sp.load_audio()
    assert cut_samples.shape[0] == 8
    assert cut_samples.shape[1] == 17778

    recording_samples = cut_sp.recording.load_audio()
    assert recording_samples.shape[0] == 8
    assert recording_samples.shape[1] == 285156


@pytest.mark.xfail(
    reason="Torchaudio 2.2 dropped support for SoX, this effect may not be available."
)
def test_cut_perturb_tempo09(cut_with_supervision):
    cut_tp = cut_with_supervision.perturb_tempo(0.9)
    assert cut_tp.start == 0.0
    assert cut_tp.duration == 1.111125
    assert cut_tp.end == 1.111125
    assert cut_tp.num_samples == 17778

    assert cut_tp.recording.duration == 17.82225
    assert cut_tp.recording.num_samples == 285156

    assert cut_tp.supervisions[0].start == 0.0
    assert cut_tp.supervisions[0].duration == 1.111125
    assert cut_tp.supervisions[0].end == 1.111125

    cut_samples = cut_tp.load_audio()
    assert cut_samples.shape[0] == 8
    assert cut_samples.shape[1] == 17778

    recording_samples = cut_tp.recording.load_audio()
    assert recording_samples.shape[0] == 8
    assert recording_samples.shape[1] == 285156


@pytest.mark.xfail(
    reason="Torchaudio 2.2 dropped support for SoX, this effect may not be available."
)
def test_cut_perturb_tempo11(cut_with_supervision):
    cut_tp = cut_with_supervision.perturb_tempo(1.1)
    assert cut_tp.start == 0.0
    assert cut_tp.duration == 0.9090625
    assert cut_tp.end == 0.9090625
    assert cut_tp.num_samples == 14545

    assert cut_tp.recording.duration == 14.5818125
    assert cut_tp.recording.num_samples == 233309

    assert cut_tp.supervisions[0].start == 0.0
    assert cut_tp.supervisions[0].duration == 0.9090625
    assert cut_tp.supervisions[0].end == 0.9090625

    cut_samples = cut_tp.load_audio()
    assert cut_samples.shape[0] == 8
    assert cut_samples.shape[1] == 14545

    recording_samples = cut_tp.recording.load_audio()
    assert recording_samples.shape[0] == 8
    assert recording_samples.shape[1] == 233309


def test_resample_cut(cut_with_supervision):
    resampled = cut_with_supervision.resample(8000)
    assert cut_with_supervision.sampling_rate == 16000
    assert resampled.sampling_rate == 8000
    assert cut_with_supervision.num_samples == 2 * resampled.num_samples
    samples = resampled.load_audio()
    assert samples.shape[1] == resampled.num_samples


@pytest.mark.parametrize("scale", [0.125, 2.0])
def test_cut_perturb_volume(cut_with_supervision, scale):

    cut_vp = cut_with_supervision.perturb_volume(scale)
    assert cut_vp.start == cut_with_supervision.start
    assert cut_vp.duration == cut_with_supervision.duration
    assert cut_vp.end == cut_with_supervision.end
    assert cut_vp.num_samples == cut_with_supervision.num_samples

    assert cut_vp.recording.duration == cut_with_supervision.recording.duration
    assert cut_vp.recording.num_samples == cut_with_supervision.recording.num_samples

    assert cut_vp.supervisions[0].start == cut_with_supervision.supervisions[0].start
    assert (
        cut_vp.supervisions[0].duration == cut_with_supervision.supervisions[0].duration
    )
    assert cut_vp.supervisions[0].end == cut_with_supervision.supervisions[0].end

    assert cut_vp.load_audio().shape == cut_with_supervision.load_audio().shape
    assert (
        cut_vp.recording.load_audio().shape
        == cut_with_supervision.recording.load_audio().shape
    )

    np.testing.assert_array_almost_equal(
        cut_vp.load_audio(), cut_with_supervision.load_audio() * scale
    )
    np.testing.assert_array_almost_equal(
        cut_vp.recording.load_audio(),
        cut_with_supervision.recording.load_audio() * scale,
    )


@pytest.mark.parametrize(
    "rir, rir_channels, expected_channels",
    [
        ("mono_rir", [0], [0, 1, 2, 3, 4, 5, 6, 7]),
        pytest.param("mono_rir", [1], None, marks=pytest.mark.xfail),
        ("multi_channel_rir", [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]),
        ("multi_channel_rir", [0], [0, 1, 2, 3, 4, 5, 6, 7]),
        ("multi_channel_rir", [1], [0, 1, 2, 3, 4, 5, 6, 7]),
        pytest.param("multi_channel_rir", [0, 1], None, marks=pytest.mark.xfail),
    ],
)
def test_cut_reverb_rir(
    cut_with_supervision, rir, rir_channels, expected_channels, request
):
    rir = request.getfixturevalue(rir)
    cut = cut_with_supervision
    cut_rvb = cut.reverb_rir(rir, rir_channels=rir_channels)
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

    assert cut_rvb.channel == expected_channels


def test_cut_reverb_fast_rir(cut_with_supervision):
    cut = cut_with_supervision
    with pytest.raises(AssertionError):
        cut_rvb = cut.reverb_rir(rir_recording=None)
