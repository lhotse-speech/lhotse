import pytest

from lhotse import SupervisionSegment
from lhotse.cut import CutSet
from lhotse.dataset import DiarizationDataset


@pytest.fixture
def cut_set():
    # The contents of 'test/fixtures/libri/cuts.json'
    cs = CutSet.from_dicts([
        {
            "channel": 0,
            "duration": 10.0,
            "features": {
                "channels": 0,
                "duration": 16.04,
                "num_features": 23,
                "num_frames": 1604,
                "recording_id": "recording-1",
                "sampling_rate": 16000,
                "start": 0.0,
                "storage_path": "test/fixtures/libri/storage",
                "storage_key": "30c2440c-93cb-4e83-b382-f2a59b3859b4.llc",
                "storage_type": "lilcom_files",
                "type": "fbank"
            },
            "id": "e3e70682-c209-4cac-629f-6fbed82c07cd",
            "recording": {
                "duration": 16.04,
                "id": "recording-1",
                "num_samples": 256640,
                "sampling_rate": 16000,
                "sources": [
                    {
                        "channels": [
                            0
                        ],
                        "source": "test/fixtures/libri/libri-1088-134315-0000.wav",
                        "type": "file"
                    }
                ]
            },
            "start": 0.0,
            "supervisions": [],
            "type": "Cut"
        }
    ])
    # These supervisions are artificially overwritten in a 10 seconds long LibriSpeech cut
    # to test the speaker activity matrix in the DiarizationDataset.
    cs[0].supervisions = [
        SupervisionSegment('s1', 'recording-1', 0, 3, speaker='spk1'),
        SupervisionSegment('s2', 'recording-1', 2, 4, speaker='spk2'),
        SupervisionSegment('s3', 'recording-1', 5, 2, speaker='spk3'),
        SupervisionSegment('s4', 'recording-1', 7.5, 2.5, speaker='spk4'),
    ]
    return cs


def test_diarization_dataset(cut_set):
    dataset = DiarizationDataset(cut_set)
    example = dataset[0]
    assert 'features' in example

    sa = example['speaker_activity']
    # Supervision s1
    assert (sa[0, :300] == 1).all()
    assert (sa[0, 300:] == 0).all()
    # Supervision s2
    assert (sa[1, :200] == 0).all()
    assert (sa[1, 200:600] == 1).all()
    assert (sa[1, 600:] == 0).all()
    # Supervision s3
    assert (sa[2, :500] == 0).all()
    assert (sa[2, 500:700] == 1).all()
    assert (sa[2, 700:] == 0).all()
    # Supervision s4
    assert (sa[3, :750] == 0).all()
    assert (sa[3, 750:] == 1).all()
