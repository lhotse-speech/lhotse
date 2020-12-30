import pytest

from lhotse.cut import CutSet
from lhotse.dataset.speech_synthesis import SpeechSynthesisDataset


@pytest.fixture
def cut_set():
    return CutSet.from_json('test/fixtures/ljspeech/cuts.json')


def test_speech_synthesis_dataset(cut_set):
    dataset = SpeechSynthesisDataset(cut_set)
    example = next(iter(dataset))
    assert example['audio'].shape[1] > 0
    assert example['features'].shape[0] > 0
    assert len(example['supervisions']['char_ids']) > 0
    assert len(example['supervisions']['char_ids_len']) > 0
