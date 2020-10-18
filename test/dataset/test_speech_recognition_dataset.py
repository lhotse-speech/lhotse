import pytest

from lhotse.cut import CutSet
from lhotse.dataset import SpeechRecognitionDataset


@pytest.fixture
def libri_cut_set():
    return CutSet.from_json('test/fixtures/ljspeech/cuts.json')


def test_speech_recognition_dataset(libri_cut_set):
    dataset = SpeechRecognitionDataset(libri_cut_set)
    assert len(dataset) == 2
    item = dataset[0]
    assert set(item) == {'features', 'text', 'supervisions_mask'}
    assert item['features'].shape == (154, 80)
    assert item['text'] == "IN EIGHTEEN THIRTEEN"
    assert item['supervisions_mask'].shape == (154,)
