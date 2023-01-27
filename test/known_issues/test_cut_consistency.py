import pytest

from lhotse import CutSet, Fbank, compute_num_frames
from lhotse.dataset import OnTheFlyFeatures
from lhotse.testing.dummies import dummy_cut, dummy_recording
from lhotse.testing.fixtures import RandomCutTestCase


class TestConsistentNumFramesAndSamples(RandomCutTestCase):
    @pytest.mark.parametrize(
        ["sampling_rate", "num_samples"],
        [
            (16000, 15995),
            (16000, 15996),
            (16000, 15997),
            (16000, 15998),
            (16000, 15999),
            (16000, 16000),
            (16000, 16001),
            (16000, 16002),
            (16000, 16003),
            (16000, 16004),
            (16000, 16005),
        ],
    )
    def test_simple_cut_num_frames_and_samples_are_consistent(
        self, sampling_rate, num_samples
    ):
        cut = self.with_cut(sampling_rate, num_samples)
        feats = cut.load_features()
        samples = cut.load_audio()

        assert cut.has_features
        assert feats.shape[0] == cut.features.num_frames
        assert feats.shape[0] == cut.num_frames
        assert feats.shape[1] == cut.features.num_features
        assert feats.shape[1] == cut.num_features

        assert cut.has_recording
        assert samples.shape[0] == 1
        assert samples.shape[1] == cut.recording.num_samples
        assert samples.shape[1] == cut.num_samples

    @pytest.mark.parametrize(
        ["sampling_rate", "num_samples", "padded_duration"],
        [
            (16000, 15995, 1.5),
            (16000, 15996, 1.5),
            (16000, 15997, 1.5),
            (16000, 15998, 1.5),
            (16000, 15999, 1.5),
            (16000, 16000, 1.5),
            (16000, 16001, 1.5),
            (16000, 16002, 1.5),
            (16000, 16003, 1.5),
            (16000, 16004, 1.5),
            (16000, 16005, 1.5),
        ],
    )
    def test_padded_cut_num_frames_and_samples_are_consistent(
        self, sampling_rate, num_samples, padded_duration
    ):
        cut = self.with_cut(sampling_rate, num_samples)
        cut = cut.pad(padded_duration)
        feats = cut.load_features()
        samples = cut.load_audio()

        assert cut.has_features
        assert feats.shape[0] == cut.num_frames
        assert feats.shape[1] == cut.num_features

        assert cut.has_recording
        assert samples.shape[0] == 1
        assert samples.shape[1] == cut.num_samples


def test_num_frames_on_the_fly_extractor_consistent_lens():
    # This is an actual case of failure
    sampling_rate = 24000
    frame_shift = 0.01

    fbank = Fbank.from_dict(
        {
            "feature_type": "kaldi-fbank",
            "sampling_rate": sampling_rate,
            "frame_shift": frame_shift,
            "frame_length": 0.05,
        }
    )
    extractor = OnTheFlyFeatures(fbank)

    cut = dummy_cut(
        0,
        duration=4.694979166666666,
        recording=dummy_recording(
            0, duration=4.694979166666666, sampling_rate=48000, with_data=True
        ),
    )
    # audio = cut.load_audio()
    # assert audio.shape[1] == 225359
    cut = cut.resample(sampling_rate)

    feats, feats_lens = extractor(CutSet.from_cuts([cut]))
    assert feats_lens[0] == 470
    assert feats.shape[1] == 470


@pytest.mark.xfail(
    reason="There is an incosistency between compute_num_frames and feature extractor in some rare conditions..."
)
def test_num_frames_mismatch_with_fbank():
    # This is an actual case of failure
    sampling_rate = 24000
    frame_shift = 0.01

    fbank = Fbank.from_dict(
        {
            "feature_type": "kaldi-fbank",
            "sampling_rate": sampling_rate,
            "frame_shift": frame_shift,
            "frame_length": 0.05,
        }
    )

    cut = dummy_cut(
        0,
        duration=4.694979166666666,
        recording=dummy_recording(
            0, duration=4.694979166666666, sampling_rate=48000, with_data=True
        ),
    )
    cut = cut.resample(sampling_rate)

    expected_num_frames = compute_num_frames(
        cut.duration, frame_shift=frame_shift, sampling_rate=sampling_rate
    )
    feats = fbank.extract(cut.load_audio(), sampling_rate)
    num_frames = feats.shape[0]

    assert num_frames == expected_num_frames
