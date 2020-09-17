from unittest.mock import Mock

import pytest

from lhotse import Cut, SupervisionSegment
from lhotse.cut import PaddingCut


class TestMasksWithoutSupervisions:
    def test_cut_audio_mask(self):
        cut = Cut('cut', start=0, duration=2, channel=0, recording=Mock(sampling_rate=16000))
        mask = cut.supervisions_audio_mask()
        assert mask.sum() == 0

    def test_cut_features_mask(self):
        cut = Cut('cut', start=0, duration=2, channel=0, features=Mock(sampling_rate=16000, frame_shift=0.01))
        mask = cut.supervisions_feature_mask()
        assert mask.sum() == 0

    def test_padding_cut_audio_mask(self):
        cut = PaddingCut('cut', duration=2, sampling_rate=16000, use_log_energy=True, num_samples=32000)
        mask = cut.supervisions_audio_mask()
        assert mask.sum() == 0

    def test_padding_cut_features_mask(self):
        cut = PaddingCut('cut', duration=2, sampling_rate=16000, use_log_energy=True, num_frames=2000, num_features=13)
        mask = cut.supervisions_feature_mask()
        assert mask.sum() == 0

    def test_mixed_cut_audio_mask(self):
        cut = Cut('cut', start=0, duration=2, channel=0, recording=Mock(sampling_rate=16000))
        mixed_cut = cut.append(cut)
        mask = mixed_cut.supervisions_audio_mask()
        assert mask.sum() == 0

    def test_mixed_cut_features_mask(self):
        cut = Cut('cut', start=0, duration=2, channel=0, features=Mock(sampling_rate=16000, frame_shift=0.01))
        mixed_cut = cut.append(cut)
        mask = mixed_cut.supervisions_feature_mask()
        assert mask.sum() == 0


@pytest.fixture
def supervisions():
    return [SupervisionSegment('sup', 'rec', start=0, duration=0.5)]


class TestMasksWithSupervisions:
    def test_cut_audio_mask(self, supervisions):
        cut = Cut('cut', start=0, duration=2, channel=0, recording=Mock(sampling_rate=16000),
                  supervisions=supervisions)
        mask = cut.supervisions_audio_mask()
        assert (mask[:8000] == 1).all()
        assert (mask[8000:] == 0).all()

    def test_cut_features_mask(self, supervisions):
        cut = Cut('cut', start=0, duration=2, channel=0, features=Mock(sampling_rate=16000, frame_shift=0.01),
                  supervisions=supervisions)
        mask = cut.supervisions_feature_mask()
        assert (mask[:50] == 1).all()
        assert (mask[50:] == 0).all()

    def test_mixed_cut_audio_mask(self, supervisions):
        cut = Cut('cut', start=0, duration=2, channel=0, recording=Mock(sampling_rate=16000),
                  supervisions=supervisions)
        mixed_cut = cut.append(cut)
        mask = mixed_cut.supervisions_audio_mask()
        assert (mask[:8000] == 1).all()
        assert (mask[8000:32000] == 0).all()
        assert (mask[32000:40000] == 1).all()
        assert (mask[40000:] == 0).all()

    def test_mixed_cut_features_mask(self, supervisions):
        cut = Cut('cut', start=0, duration=2, channel=0, features=Mock(sampling_rate=16000, frame_shift=0.01),
                  supervisions=supervisions)
        mixed_cut = cut.append(cut)
        mask = mixed_cut.supervisions_feature_mask()
        assert (mask[:50] == 1).all()
        assert (mask[50:200] == 0).all()
        assert (mask[200:250] == 1).all()
        assert (mask[250:] == 0).all()
