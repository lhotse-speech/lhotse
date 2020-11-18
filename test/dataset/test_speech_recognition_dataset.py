import pytest
import torch
from torch.utils.data import DataLoader

from lhotse.cut import CutSet
from lhotse.dataset import SpeechRecognitionDataset
from lhotse.dataset.speech_recognition import K2DataLoader, K2SpeechRecognitionDataset, \
    K2SpeechRecognitionIterableDataset, concat_cuts
from lhotse.testing.dummies import DummyManifest, dummy_cut


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
        assert example['supervisions'][0]['sequence_idx'] == i
        assert example['supervisions'][0]['start_frame'] == 0
        assert example['supervisions'][0]['num_frames'] == 154
    example = dataset[3]
    assert example['features'].shape == (308, 80)
    assert len(example['supervisions']) == 2
    assert example['supervisions'][0]['text'] == 'IN EIGHTEEN THIRTEEN'
    assert example['supervisions'][0]['sequence_idx'] == 3
    assert example['supervisions'][0]['start_frame'] == 0
    assert example['supervisions'][0]['num_frames'] == 154
    assert example['supervisions'][1]['text'] == 'IN EIGHTEEN THIRTEEN'
    assert example['supervisions'][1]['sequence_idx'] == 3
    assert example['supervisions'][1]['start_frame'] == 154
    assert example['supervisions'][1]['num_frames'] == 154


def test_k2_dataloader(k2_cut_set):
    from torch import tensor
    dataset = K2SpeechRecognitionDataset(k2_cut_set)
    dloader = K2DataLoader(dataset, batch_size=4)
    batch = next(iter(dloader))
    assert batch['features'].shape == (4, 308, 80)
    # Each list has 5 items, to account for:
    # one cut with two supervisions + 3 three cuts with one supervision
    assert (batch['supervisions']['sequence_idx'] == tensor([0, 1, 2, 3, 3])).all()
    assert batch['supervisions']['text'] == ['IN EIGHTEEN THIRTEEN'] * 5  # a list, not tensor
    assert (batch['supervisions']['start_frame'] == tensor([0] * 4 + [154])).all()
    assert (batch['supervisions']['num_frames'] == tensor([154] * 5)).all()


@pytest.mark.parametrize('num_workers', [0, 1])
def test_k2_speech_recognition_iterable_dataset(k2_cut_set, num_workers):
    from torch import tensor
    dataset = K2SpeechRecognitionIterableDataset(k2_cut_set, shuffle=False)
    # Note: "batch_size=None" disables the automatic batching mechanism,
    #       which is required when Dataset takes care of the collation itself.
    dloader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    batch = next(iter(dloader))
    assert batch['features'].shape == (4, 308, 80)
    # Each list has 5 items, to account for:
    # one cut with two supervisions + 3 three cuts with one supervision
    assert (batch['supervisions']['sequence_idx'] == tensor([0, 1, 2, 3, 3])).all()
    assert batch['supervisions']['text'] == ['IN EIGHTEEN THIRTEEN'] * 5  # a list, not tensor
    assert (batch['supervisions']['start_frame'] == tensor([0] * 4 + [154])).all()
    assert (batch['supervisions']['num_frames'] == tensor([154] * 5)).all()


@pytest.mark.parametrize('num_workers', [2, 3, 4])
def test_k2_speech_recognition_iterable_dataset_multiple_workers(k2_cut_set, num_workers):
    from torch import tensor
    dataset = K2SpeechRecognitionIterableDataset(k2_cut_set, shuffle=False)
    dloader = DataLoader(dataset, batch_size=None, num_workers=num_workers)

    # We expect a variable number of batches for each parametrized num_workers value,
    # because the dataset is small with 4 cuts that are partitioned across the workers.
    batches = list(dloader)

    features = torch.cat([b['features'] for b in batches])
    assert features.shape == (4, 308, 80)
    text = [t for b in batches for t in b['supervisions']['text']]
    assert text == ['IN EIGHTEEN THIRTEEN'] * 5  # a list, not tensor
    start_frame = torch.cat([b['supervisions']['start_frame'] for b in batches])
    assert (start_frame == tensor([0] * 4 + [154])).all()
    num_frames = torch.cat([b['supervisions']['num_frames'] for b in batches])
    assert (num_frames == tensor([154] * 5)).all()


def test_k2_speech_recognition_iterable_dataset_shuffling():
    # The dummy cuts have a duration of 1 second each
    cut_set = DummyManifest(CutSet, begin_id=0, end_id=100)

    dataset = K2SpeechRecognitionIterableDataset(
        cuts=cut_set,
        shuffle=True,
        # Set an effective batch size of 10 cuts, as all have 1s duration == 100 frames
        # This way we're testing that it works okay when returning multiple batches in
        # a full epoch.
        max_frames=1000
    )
    dloader = DataLoader(dataset, batch_size=None, num_workers=2)
    dloader_cut_ids = []
    batches = []
    for batch in dloader:
        batches.append(batch)
        dloader_cut_ids.extend(list(batch['supervisions']['cut_id']))

    # Invariant 1: we receive the same amount of items in a dataloader epoch as there we in the CutSet
    assert len(dloader_cut_ids) == len(cut_set)
    # Invariant 2: the items are not duplicated
    assert len(set(dloader_cut_ids)) == len(dloader_cut_ids)
    # Invariant 3: the items are shuffled, i.e. the order is different than that in the CutSet
    assert dloader_cut_ids != [c.id for c in cut_set]


def test_concat_cuts():
    cuts = [
        dummy_cut(duration=30.0),
        dummy_cut(duration=20.0),
        dummy_cut(duration=10.0),
        dummy_cut(duration=5.0),
        dummy_cut(duration=4.0),
        dummy_cut(duration=3.0),
        dummy_cut(duration=2.0),
    ]
    concat = concat_cuts(cuts, gap=1.0)
    assert [c.duration for c in concat] == [
        30.0,
        20.0 + 1.0 + 2.0 + 1.0 + 3.0,  # == 27.0
        10.0 + 1.0 + 4.0 + 1.0 + 5.0,  # == 21.0
    ]


def test_k2_speech_recognition_iterable_dataset_low_max_frames(k2_cut_set):
    dataset = K2SpeechRecognitionIterableDataset(k2_cut_set, shuffle=False, max_frames=2)
    dloader = DataLoader(dataset, batch_size=None)
    # Check that it does not crash
    for batch in dloader:
        # There will be only a single item in each batch as we're exceeding the limit each time.
        assert batch['features'].shape[0] == 1
