import pytest

from lhotse import AudioSource, Cut, CutSet, Recording, SupervisionSegment
from lhotse.cut import PaddingCut


@pytest.fixture
def file_source():
    return AudioSource(type='file', channels=[0], source='test/fixtures/mono_c0.wav')


@pytest.fixture
def recording(file_source):
    return Recording(id='rec', sources=[file_source], sampling_rate=8000, num_samples=4000, duration=0.5)


@pytest.fixture
def cut_with_supervision(recording):
    return Cut(
        id='cut',
        start=0.0,
        duration=0.5,
        channel=0,
        supervisions=[
            SupervisionSegment(
                id='sup',
                recording_id='rec',
                start=0.0,
                duration=0.5
            )
        ],
        recording=recording
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


@pytest.fixture
def cut_with_supervision_start01(recording):
    return Cut(
        id='cut_start01',
        start=0.1,
        duration=0.4,
        channel=0,
        supervisions=[
            SupervisionSegment(
                id='sup',
                recording_id='rec',
                start=0.1,
                duration=0.3
            )
        ],
        recording=recording
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


def test_mixed_cut_start01_perturb(cut_with_supervision_start01):
    mixed_sp = (
        cut_with_supervision_start01
            .append(cut_with_supervision_start01)
            .perturb_speed(1.1)
    )
    assert mixed_sp.start == 0  # MixedCut always starts at 0
    assert mixed_sp.duration == 0.363625 * 2
    assert mixed_sp.end == 0.363625 * 2
    assert mixed_sp.num_samples == 2909 * 2

    assert mixed_sp.supervisions[0].start == 0.090875
    assert mixed_sp.supervisions[0].duration == 0.27275
    assert mixed_sp.supervisions[0].end == 0.363625
    assert mixed_sp.supervisions[1].start == 0.4545  # round(0.363625 + 0.090875, ndigits=8)
    assert mixed_sp.supervisions[1].duration == .27275
    assert mixed_sp.supervisions[1].end == 0.363625 * 2

    cut_samples = mixed_sp.load_audio()
    assert cut_samples.shape[0] == 1
    assert cut_samples.shape[1] == 2909 * 2


def test_padding_cut_perturb():
    cut = PaddingCut(id='cut', duration=5.75, sampling_rate=16000, use_log_energy=True, num_samples=92000)
    cut_sp = cut.perturb_speed(1.1)
    assert cut_sp.num_samples == 83636
    assert cut_sp.duration == 5.22725


def test_cut_set_perturb(cut_with_supervision, cut_with_supervision_start01):
    cut_set = CutSet.from_cuts([cut_with_supervision, cut_with_supervision_start01])
    cs_sp = cut_set.perturb_speed(1.1)
    for cut_sp, cut in zip(cs_sp, cut_set):
        samples = cut_sp.load_audio()
        assert samples.shape[1] == cut_sp.num_samples
        assert samples.shape[1] < cut.num_samples
