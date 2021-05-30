from unittest.mock import Mock
from itertools import chain

import pytest
import numpy as np

from lhotse import Cut, SupervisionSegment
from lhotse.cut import PaddingCut
from lhotse.supervision import AlignmentItem
from lhotse.utils import LOG_EPSILON


class TestMasksWithoutSupervisions:
    def test_cut_audio_mask(self):
        cut = Cut('cut', start=0, duration=2, channel=0, recording=Mock(sampling_rate=16000))
        mask = cut.supervisions_audio_mask()
        assert mask.sum() == 0

    def test_cut_features_mask(self):
        cut = Cut('cut', start=0, duration=2, channel=0,
                  features=Mock(sampling_rate=16000, frame_shift=0.01, num_frames=2000))
        mask = cut.supervisions_feature_mask()
        assert mask.sum() == 0

    def test_padding_cut_audio_mask(self):
        cut = PaddingCut('cut', duration=2, sampling_rate=16000, feat_value=LOG_EPSILON, num_samples=32000)
        mask = cut.supervisions_audio_mask()
        assert mask.sum() == 0

    def test_padding_cut_features_mask(self):
        cut = PaddingCut('cut', duration=2, sampling_rate=16000, feat_value=LOG_EPSILON, num_frames=2000,
                         num_features=13)
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
    return [
        SupervisionSegment('sup', 'rec', start=0, duration=0.5,
            alignment={
                'word': [
                    AlignmentItem(symbol='a', start=0, duration=0.1),
                    AlignmentItem(symbol='b', start=0.2, duration=0.2)
                ]
            }
        )
    ]


class TestMasksWithSupervisions:
    @pytest.mark.parametrize('alignment', [None, 'word'])
    def test_cut_audio_mask(self, supervisions, alignment):
        cut = Cut('cut', start=0, duration=2, channel=0, recording=Mock(sampling_rate=16000),
                  supervisions=supervisions)
        mask = cut.supervisions_audio_mask(use_alignment_if_exists=alignment)
        if alignment == "word":
            ones = np.index_exp[list(chain(range(0,1600), range(3200,6400)))]
            zeros = np.index_exp[list(chain(range(1600,3200), range(6400,1600)))]
        else:
            ones = np.index_exp[range(0,8000)]
            zeros = np.index_exp[range(8000,16000)]
        assert (mask[ones] == 1).all()
        assert (mask[zeros] == 0).all()

    @pytest.mark.parametrize('alignment', [None, 'word'])
    def test_cut_features_mask(self, supervisions, alignment):
        cut = Cut('cut', start=0, duration=2, channel=0,
                  features=Mock(sampling_rate=16000, frame_shift=0.01, num_frames=2000),
                  supervisions=supervisions)
        mask = cut.supervisions_feature_mask(use_alignment_if_exists=alignment)
        if alignment == "word":
            ones = np.index_exp[list(chain(range(0,10), range(20,40)))]
            zeros = np.index_exp[list(chain(range(10,20), range(40,200)))]
        else:
            ones = np.index_exp[range(0,50)]
            zeros = np.index_exp[range(50,200)]
        assert (mask[ones] == 1).all()
        assert (mask[zeros] == 0).all()

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
