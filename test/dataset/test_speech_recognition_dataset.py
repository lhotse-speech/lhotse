import pytest

from lhotse.cut import CutSet
from lhotse.dataset import SpeechRecognitionDataset
from lhotse.dataset.speech_recognition import K2DataLoader, K2SpeechRecognitionDataset


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


@pytest.fixture
def k2_cut_set(libri_cut_set):
    # Create a cut set with 4 cuts, one of them having two supervisions
    return CutSet.from_cuts([
        libri_cut_set[0],
        libri_cut_set[0].with_id('copy-1'),
        libri_cut_set[0].with_id('copy-2'),
        libri_cut_set[0].append(libri_cut_set[0])
    ]).pad()


def test_k2_speech_recognition_dataset(k2_cut_set):
    dataset = K2SpeechRecognitionDataset(k2_cut_set)
    for i in range(3):
        example = dataset[i]
        assert example['features'].shape == (308, 80)
        assert len(example['supervisions']) == 1
        assert example['supervisions'][0]['text'] == 'IN EIGHTEEN THIRTEEN'
        assert example['supervisions'][0]['example_idx'] == i
        assert example['supervisions'][0]['start_frame'] == 0
        assert example['supervisions'][0]['end_frame'] == 154
    example = dataset[3]
    assert example['features'].shape == (308, 80)
    assert len(example['supervisions']) == 2
    assert example['supervisions'][0]['text'] == 'IN EIGHTEEN THIRTEEN'
    assert example['supervisions'][0]['example_idx'] == 3
    assert example['supervisions'][0]['start_frame'] == 0
    assert example['supervisions'][0]['end_frame'] == 154
    assert example['supervisions'][1]['text'] == 'IN EIGHTEEN THIRTEEN'
    assert example['supervisions'][1]['example_idx'] == 3
    assert example['supervisions'][1]['start_frame'] == 154
    assert example['supervisions'][1]['end_frame'] == 308


def test_k2_dataloader(k2_cut_set):
    from torch import tensor
    dataset = K2SpeechRecognitionDataset(k2_cut_set)
    dloader = K2DataLoader(dataset, batch_size=4)
    batch = next(iter(dloader))
    assert batch['features'].shape == (4, 308, 80)
    # Each list has 5 items, to account for:
    # one cut with two supervisions + 3 three cuts with one supervision
    assert (batch['supervisions']['example_idx'] == tensor([0, 1, 2, 3, 3])).all()
    assert batch['supervisions']['text'] == ['IN EIGHTEEN THIRTEEN'] * 5  # a list, not tensor
    assert (batch['supervisions']['start_frame'] == tensor([0] * 4 + [154])).all()
    assert (batch['supervisions']['end_frame'] == tensor([154] * 4 + [308])).all()
