from itertools import chain
from unittest.mock import Mock

import numpy as np
import pytest

from lhotse import MonoCut, SupervisionSegment
from lhotse.cut import PaddingCut
from lhotse.supervision import AlignmentItem
from lhotse.utils import LOG_EPSILON


class TestMasksWithoutSupervisions:
    def test_cut_audio_mask(self):
        cut = MonoCut(
            "cut", start=0, duration=2, channel=0, recording=Mock(sampling_rate=16000)
        )
        mask = cut.supervisions_audio_mask()
        assert mask.sum() == 0

    def test_cut_features_mask(self):
        cut = MonoCut(
            "cut",
            start=0,
            duration=2,
            channel=0,
            features=Mock(sampling_rate=16000, frame_shift=0.01, num_frames=2000),
        )
        mask = cut.supervisions_feature_mask()
        assert mask.sum() == 0

    def test_padding_cut_audio_mask(self):
        cut = PaddingCut(
            "cut",
            duration=2,
            sampling_rate=16000,
            feat_value=LOG_EPSILON,
            num_samples=32000,
        )
        mask = cut.supervisions_audio_mask()
        assert mask.sum() == 0

    def test_padding_cut_features_mask(self):
        cut = PaddingCut(
            "cut",
            duration=2,
            sampling_rate=16000,
            feat_value=LOG_EPSILON,
            num_frames=2000,
            num_features=13,
        )
        mask = cut.supervisions_feature_mask()
        assert mask.sum() == 0

    def test_mixed_cut_audio_mask(self):
        cut = MonoCut(
            "cut", start=0, duration=2, channel=0, recording=Mock(sampling_rate=16000)
        )
        mixed_cut = cut.append(cut)
        mask = mixed_cut.supervisions_audio_mask()
        assert mask.sum() == 0

    def test_mixed_cut_features_mask(self):
        cut = MonoCut(
            "cut",
            start=0,
            duration=2,
            channel=0,
            features=Mock(sampling_rate=16000, frame_shift=0.01),
        )
        mixed_cut = cut.append(cut)
        mask = mixed_cut.supervisions_feature_mask()
        assert mask.sum() == 0


@pytest.fixture
def supervisions():
    return [
        SupervisionSegment(
            "sup",
            "rec",
            start=0,
            duration=0.5,
            speaker="SpkA",
            alignment={
                "word": [
                    AlignmentItem(symbol="a", start=0, duration=0.1),
                    AlignmentItem(symbol="b", start=0.2, duration=0.2),
                ]
            },
        ),
        SupervisionSegment(
            "sup",
            "rec",
            start=0.6,
            duration=0.2,
            speaker="SpkB",
            alignment={
                "word": [
                    AlignmentItem(symbol="a", start=0.6, duration=0.2),
                ]
            },
        ),
    ]


class TestMasksWithSupervisions:
    @pytest.mark.parametrize("alignment", [None, "word"])
    def test_cut_audio_mask(self, supervisions, alignment):
        cut = MonoCut(
            "cut",
            start=0,
            duration=2,
            channel=0,
            recording=Mock(sampling_rate=16000),
            supervisions=supervisions,
        )
        mask = cut.supervisions_audio_mask(use_alignment_if_exists=alignment)
        if alignment == "word":
            ones = np.index_exp[
                list(chain(range(0, 1600), range(3200, 6400), range(9600, 12800)))
            ]
            zeros = np.index_exp[
                list(chain(range(1600, 3200), range(6400, 9600), range(12800, 32000)))
            ]
        else:
            ones = np.index_exp[list(chain(range(0, 8000), range(9600, 12800)))]
            zeros = np.index_exp[list(chain(range(8000, 9600), range(12800, 32000)))]
        assert (mask[ones] == 1).all()
        assert (mask[zeros] == 0).all()

    @pytest.mark.parametrize("alignment", [None, "word"])
    def test_cut_features_mask(self, supervisions, alignment):
        cut = MonoCut(
            "cut",
            start=0,
            duration=2,
            channel=0,
            features=Mock(sampling_rate=16000, frame_shift=0.01, num_frames=2000),
            supervisions=supervisions,
        )
        mask = cut.supervisions_feature_mask(use_alignment_if_exists=alignment)
        if alignment == "word":
            ones = np.index_exp[list(chain(range(0, 10), range(20, 40), range(60, 80)))]
            zeros = np.index_exp[
                list(chain(range(10, 20), range(40, 60), range(80, 200)))
            ]
        else:
            ones = np.index_exp[list(chain(range(0, 50), range(60, 80)))]
            zeros = np.index_exp[list(chain(range(50, 60), range(80, 200)))]
        assert (mask[ones] == 1).all()
        assert (mask[zeros] == 0).all()

    @pytest.mark.parametrize("alignment", [None, "word"])
    def test_cut_speakers_audio_mask(self, supervisions, alignment):
        cut = MonoCut(
            "cut",
            start=0,
            duration=2,
            channel=0,
            recording=Mock(sampling_rate=16000),
            supervisions=supervisions,
        )
        mask = cut.speakers_audio_mask(use_alignment_if_exists=alignment)
        if alignment == "word":
            ones = [
                np.index_exp[list(chain(range(0, 1600), range(3200, 6400)))],
                np.index_exp[list(chain(range(9600, 12800)))],
            ]
            zeros = [
                np.index_exp[list(chain(range(1600, 3200), range(6400, 32000)))],
                np.index_exp[list(chain(range(0, 9600), range(12800, 32000)))],
            ]
        else:
            ones = [np.index_exp[range(0, 8000)], np.index_exp[range(9600, 12800)]]
            zeros = [
                np.index_exp[list(chain(range(8000, 32000)))],
                np.index_exp[list(chain(range(0, 9600), range(12800, 32000)))],
            ]
        assert (mask[0, ones[0]] == 1).all()
        assert (mask[1, ones[1]] == 1).all()
        assert (mask[0, zeros[0]] == 0).all()
        assert (mask[1, zeros[1]] == 0).all()

    @pytest.mark.parametrize("alignment", [None, "word"])
    def test_cut_speakers_features_mask(self, supervisions, alignment):
        cut = MonoCut(
            "cut",
            start=0,
            duration=2,
            channel=0,
            features=Mock(sampling_rate=16000, frame_shift=0.01, num_frames=2000),
            supervisions=supervisions,
        )
        mask = cut.speakers_feature_mask(use_alignment_if_exists=alignment)
        if alignment == "word":
            ones = [
                np.index_exp[list(chain(range(0, 10), range(20, 40)))],
                np.index_exp[list(chain(range(60, 80)))],
            ]
            zeros = [
                np.index_exp[list(chain(range(10, 20), range(40, 200)))],
                np.index_exp[list(chain(range(0, 60), range(80, 200)))],
            ]
        else:
            ones = [
                np.index_exp[list(chain(range(0, 50)))],
                np.index_exp[list(chain(range(60, 80)))],
            ]
            zeros = [
                np.index_exp[list(chain(range(50, 200)))],
                np.index_exp[list(chain(range(0, 60), range(80, 200)))],
            ]
        assert (mask[0, ones[0]] == 1).all()
        assert (mask[1, ones[1]] == 1).all()
        assert (mask[0, zeros[0]] == 0).all()
        assert (mask[1, zeros[1]] == 0).all()

    def test_mixed_cut_audio_mask(self, supervisions):
        cut = MonoCut(
            "cut",
            start=0,
            duration=2,
            channel=0,
            recording=Mock(sampling_rate=16000),
            supervisions=supervisions,
        )
        mixed_cut = cut.append(cut)
        mask = mixed_cut.supervisions_audio_mask()
        ones = np.index_exp[
            list(
                chain(
                    range(0, 8000),
                    range(9600, 12800),
                    range(32000, 40000),
                    range(41600, 44800),
                )
            )
        ]
        zeros = np.index_exp[
            list(
                chain(
                    range(8000, 9600),
                    range(12800, 32000),
                    range(40000, 41600),
                    range(44800, 64000),
                )
            )
        ]
        assert (mask[ones] == 1).all()
        assert (mask[zeros] == 0).all()

    def test_mixed_cut_features_mask(self, supervisions):
        cut = MonoCut(
            "cut",
            start=0,
            duration=2,
            channel=0,
            features=Mock(sampling_rate=16000, frame_shift=0.01),
            supervisions=supervisions,
        )
        mixed_cut = cut.append(cut)
        mask = mixed_cut.supervisions_feature_mask()
        ones = np.index_exp[
            list(chain(range(0, 50), range(60, 80), range(200, 250), range(260, 280)))
        ]
        zeros = np.index_exp[
            list(chain(range(50, 60), range(80, 200), range(250, 260), range(280, 400)))
        ]
        assert (mask[ones] == 1).all()
        assert (mask[zeros] == 0).all()
