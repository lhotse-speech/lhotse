from math import isclose
import pytest

import torch

from lhotse.utils import time_diff_to_num_frames
from lhotse.cut import CutSet
from lhotse.dataset.vad import VadDataset


@pytest.fixture
def cut_set():
    return CutSet.from_json('test/fixtures/ami/cuts.json')


def test_vad_dataset(cut_set):
    duration = 5.0
    feature_threhold = -8.0

    cuts = list(cut_set)
    assert isclose(cuts[0].duration, 6.0)
    assert len(cuts[0].supervisions) == 2

    v1_start = time_diff_to_num_frames(cuts[0].supervisions[0].start, frame_length=0, frame_shift=0.01)
    v1_end = time_diff_to_num_frames(cuts[0].supervisions[0].end, frame_length=0, frame_shift=0.01)
    v2_start = time_diff_to_num_frames(cuts[0].supervisions[1].start, frame_length=0, frame_shift=0.01)
    v2_end = time_diff_to_num_frames(cuts[0].supervisions[1].end, frame_length=0, frame_shift=0.01)
    end = time_diff_to_num_frames(duration, frame_length=0, frame_shift=0.01)

    dataset = VadDataset(cut_set.cut_into_windows(duration=duration))
    example = dataset[0]
    assert isclose(float(torch.mean(example['is_voice'][0:v1_start])), 0)
    assert isclose(float(torch.mean(example['is_voice'][v1_start:v1_end])), 1)
    assert isclose(float(torch.mean(example['is_voice'][v1_end:v2_start])), 0)
    assert isclose(float(torch.mean(example['is_voice'][v2_start:v2_end])), 1)
    assert isclose(float(torch.mean(example['is_voice'][v2_end:end])), 0)

    assert float(torch.mean(example['features'][0:v1_start])) < feature_threhold
    assert float(torch.mean(example['features'][v1_start:v1_end])) > feature_threhold
    assert float(torch.mean(example['features'][v1_end:v2_start])) < feature_threhold
    assert float(torch.mean(example['features'][v2_start:v2_end])) > feature_threhold
    assert float(torch.mean(example['features'][v2_end:end])) < feature_threhold
